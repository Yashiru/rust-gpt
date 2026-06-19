"""Train the GPT on the Rust corpus: tokenizer -> dataset -> model -> generation.

Run from the repo root with `python -m rustgpt.train` (paths are relative to CWD).
"""

import math
import os
import time
from pathlib import Path

# Pin training to GPU 0 (the RTX 2080 Ti) by default, so the desktop GPU stays
# free. Must be set before torch initializes CUDA. The GTX 1070 is sm_61 and
# unsupported by this PyTorch build anyway. Override by exporting CUDA_VISIBLE_DEVICES.
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")  # number GPUs like nvidia-smi
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # fewer fragmentation OOMs

import torch
import torch.nn.functional as F

from rustgpt import Tokenizer, TOKENIZER_BACKEND  # fast Rust drop-in if built, else Python
from rustgpt.model import GPT
from rustgpt.data import load_or_encode_corpus, train_val_split, get_batch

CORPUS_PATH = Path("data/rust_corpus.txt")  # concatenated .rs files
IDS_PATH = Path("data/rust_ids.bin")        # corpus encoded to token ids (uint16 cache)
WEIGHTS_PATH = Path("weights.pt")           # trained model cache
VOCAB_PATH = Path("tokenizer.json")         # trained tokenizer cache
VOCAB_SIZE = 16384         # 256 base bytes + 16128 merges (balanced for this model size)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # train on the GPU when available
torch.backends.cudnn.benchmark = True  # fixed input shapes -> let cuDNN autotune kernels

# Modern transformer (RoPE + SwiGLU + RMSNorm + weight tying), tuned for an 11 GB
# RTX 2080 Ti. Weight tying frees ~8M params, reinvested into depth (10 layers).
N_EMBD = 512
N_HEAD = 8
N_LAYER = 10
CONTEXT_SIZE = 1024        # max positions the model supports

STEPS = 55000           # ~1 epoch over the ~900M-token (3.1 GB) corpus at batch 32 / block 512
LR = 6e-4               # peak LR; warmup + cosine decay make this safe
MIN_LR = 6e-5           # cosine floor (final LR)
WARMUP_STEPS = 1000     # ~2% of STEPS, linear warmup from 0 to peak
GRAD_CLIP = 1.0         # clip the global gradient norm (training stability)
BATCH_SIZE = 32         # measured to fit ~8.3 GB at L10/block 512 (B40+ OOMs once the desktop shares GPU 0)
BLOCK_SIZE = 512        # training context length (must be <= CONTEXT_SIZE)
VAL_FRAC = 0.1          # fraction of the token stream held out for validation
EVAL_INTERVAL = 250     # estimate train/val loss every N steps
EVAL_ITERS = 50         # how many batches to average when estimating loss
DROPOUT = 0.2           # regularization: drop this fraction of activations during training
WEIGHT_DECAY = 0.1      # AdamW L2 penalty (applied to matmul/embedding weights only)
EARLY_STOP_PATIENCE = 5 # stop if val loss hasn't improved for this many evals
TOKENIZER_SAMPLE_CHARS = 300_000_000  # train the BPE on a sample, not the whole corpus

def load_or_train_tokenizer():
    """Load the tokenizer from the JSON cache, or train it (in Rust) if absent."""
    if VOCAB_PATH.exists():
        print(f"Loading tokenizer from {VOCAB_PATH} ...")
        tok = Tokenizer.load(str(VOCAB_PATH))
        print(f"  loaded {tok.vocab_size} tokens")
        return tok

    if not CORPUS_PATH.exists():
        raise SystemExit(
            f"Corpus not found: {CORPUS_PATH}\n"
            f"Build one with `python scripts/download_corpus.py` and re-run."
        )

    # BPE frequency stats converge fast, so we train on a sample rather than the
    # whole corpus (training on multiple GB is needlessly slow). The full corpus is
    # still encoded with the resulting tokenizer (see load_or_encode_corpus).
    with open(CORPUS_PATH, encoding="utf-8") as f:
        text = f.read(TOKENIZER_SAMPLE_CHARS)
    print(f"Training BPE (vocab_size={VOCAB_SIZE}, backend={TOKENIZER_BACKEND}) on a "
          f"{len(text.encode('utf-8')):,}-byte sample of {CORPUS_PATH} ...")
    from tqdm import tqdm

    t0 = time.time()
    with tqdm(total=VOCAB_SIZE - 256, desc="BPE merges", unit="merge") as bar:
        tok = Tokenizer.train(text, VOCAB_SIZE,
                              progress=lambda step, total: bar.update(step - bar.n))
    tok.save(str(VOCAB_PATH))
    print(f"  trained {tok.vocab_size} tokens in {time.time() - t0:.1f}s, saved to {VOCAB_PATH}")
    return tok


def pick_amp_dtype():
    """Pick the fastest safe autocast dtype for the current GPU (None -> full FP32).

    Ampere+ (sm_80+) has bf16 tensor cores and needs no loss scaling; Volta/Turing
    (sm_70/75) has fp16 tensor cores (scaling required); older GPUs / CPU stay FP32.
    NB: torch.cuda.is_bf16_supported() reports True on Turing via (unaccelerated)
    emulation, so we gate on the compute capability instead.
    """
    if not torch.cuda.is_available():
        return None
    major, _ = torch.cuda.get_device_capability()
    if major >= 8:
        return torch.bfloat16
    if major >= 7:
        return torch.float16
    return None


def compute_loss(model, xb, yb, amp_dtype):
    """Forward pass + cross-entropy over every position, optionally in mixed precision."""
    with torch.autocast(device_type=DEVICE, dtype=amp_dtype, enabled=amp_dtype is not None):
        logits = model(xb)
        return F.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1))


@torch.no_grad()
def estimate_loss(model, train_data, val_data, amp_dtype):
    """Average loss over a few fixed batches of train and val, for a stable read."""
    model.eval()  # no dropout/batchnorm here, but this is the idiomatic toggle
    means = []
    for data in (train_data, val_data):
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            xb, yb = get_batch(data, BLOCK_SIZE, BATCH_SIZE, DEVICE)
            losses[k] = compute_loss(model, xb, yb, amp_dtype).item()
        means.append(losses.mean().item())
    model.train()
    return means[0], means[1]


def generate_sample(model, tokenizer, prompt, max_new_tokens=200,
                    temperature=0.8, top_k=50):
    """Encode `prompt`, let the model continue it, and decode back to Rust text."""
    model.eval()
    ids = tokenizer.encode(prompt)
    idx = torch.tensor([ids], dtype=torch.int64, device=DEVICE)  # (1, T)
    out = model.generate(idx, max_new_tokens, temperature=temperature, top_k=top_k)
    model.train()
    return tokenizer.decode(out[0].tolist())


def train_or_load(model, train_data, val_data, amp_dtype=None):
    """Train the model from scratch, or load the weights from a cache if present."""
    if WEIGHTS_PATH.exists():
        print(f"Loading model weights from {WEIGHTS_PATH} ...")
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
        print("  loaded.")
    else:
        print(f"Training model for {STEPS:,} steps ...")
        train(model, train_data, val_data, amp_dtype)
        torch.save(model.state_dict(), WEIGHTS_PATH)
        print(f"  trained and saved to {WEIGHTS_PATH}.")


def lr_at(step):
    """Linear warmup for WARMUP_STEPS, then cosine decay from LR down to MIN_LR."""
    if step < WARMUP_STEPS:
        return LR * (step + 1) / WARMUP_STEPS
    if step >= STEPS:
        return MIN_LR
    progress = (step - WARMUP_STEPS) / max(1, STEPS - WARMUP_STEPS)
    return MIN_LR + 0.5 * (LR - MIN_LR) * (1 + math.cos(math.pi * progress))


def build_optimizer(model, lr, weight_decay):
    """AdamW with the idiomatic split: decay matmul/embedding weights, not 1D params.

    Weight decay on biases and LayerNorm gains hurts more than it helps, so only
    tensors with >= 2 dims (Linear/Embedding weights) get the L2 penalty.
    """
    decay, no_decay = [], []
    for p in model.parameters():
        (decay if p.dim() >= 2 else no_decay).append(p)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=lr)


def train(model, train_data, val_data, amp_dtype=None):
    """Standard PyTorch training loop: AdamW + GradScaler, optionally mixed precision.

    Logs the train/val curve, keeps the best (least-overfit) checkpoint, and stops
    early once val stops improving, then restores that best checkpoint.
    """
    from tqdm import tqdm

    optimizer = build_optimizer(model, LR, WEIGHT_DECAY)
    # GradScaler handles fp16's tiny-gradient underflow (dynamic scaling + skips
    # inf/nan steps). It's a no-op passthrough for bf16/fp32, hence `enabled`.
    scaler = torch.amp.GradScaler(DEVICE, enabled=(amp_dtype == torch.float16))
    model.train()

    history = []
    best_val = float("inf")
    best_state = None
    evals_since_best = 0

    with tqdm(total=STEPS, desc="training") as bar:
        for step in range(STEPS):
            lr = lr_at(step)
            for group in optimizer.param_groups:
                group["lr"] = lr
            xb, yb = get_batch(train_data, BLOCK_SIZE, BATCH_SIZE, DEVICE)
            training_step(model, optimizer, scaler, xb, yb, amp_dtype)
            if step % EVAL_INTERVAL == 0 or step == STEPS - 1:
                tr, va = estimate_loss(model, train_data, val_data, amp_dtype)
                history.append((step, tr, va))
                bar.set_postfix(train=f"{tr:.3f}", val=f"{va:.3f}", lr=f"{lr:.1e}")
                if va < best_val:
                    best_val = va
                    best_state = {k: v.detach().cpu().clone()
                                  for k, v in model.state_dict().items()}
                    evals_since_best = 0
                else:
                    evals_since_best += 1
                    if evals_since_best >= EARLY_STOP_PATIENCE:
                        print(f"\nEarly stop at step {step}: val hasn't improved for "
                              f"{EARLY_STOP_PATIENCE} evals (best val {best_val:.3f}).")
                        break
            bar.update(1)

    if best_state is not None:
        model.load_state_dict(best_state)  # roll back to the least-overfit weights
    print_history(history, best_val)


def print_history(history, best_val):
    """Print the train/val trajectory so the overfitting gap is visible over time."""
    print("\n--- loss history (step | train | val | gap) ---")
    for step, tr, va in history:
        mark = "  <- best" if va == best_val else ""
        print(f"{step:6d} | {tr:6.3f} | {va:6.3f} | {va - tr:+.3f}{mark}")


def training_step(model, optimizer, scaler, xb, yb, amp_dtype):
    """One step: zero_grad -> backward -> unscale -> clip -> step, with AMP scaling."""
    optimizer.zero_grad(set_to_none=True)
    loss = compute_loss(model, xb, yb, amp_dtype)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)  # unscale before clipping so the threshold is in real units
    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    scaler.step(optimizer)
    scaler.update()
    return loss


def main():
    if DEVICE == "cuda":
        print(f"Device: {torch.cuda.get_device_name(0)} "
              f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'all')})")
    else:
        print("Device: CPU")

    # 1. tokenizer: load or train (native Rust BPE)
    tokenizer = load_or_train_tokenizer()

    # 2. dataset: encode (cached) -> 1D token stream -> train/val split
    ids = load_or_encode_corpus(tokenizer, CORPUS_PATH, IDS_PATH)
    train_data, val_data = train_val_split(ids, VAL_FRAC)
    print(f"  train {len(train_data):,} tokens | val {len(val_data):,} tokens "
          f"| block {BLOCK_SIZE} | batch {BATCH_SIZE}")

    # 3. model
    model = GPT(vocab_size=tokenizer.vocab_size, n_embd=N_EMBD, n_head=N_HEAD,
                n_layer=N_LAYER, context_size=CONTEXT_SIZE, dropout=DROPOUT)
    assert BLOCK_SIZE <= CONTEXT_SIZE, "BLOCK_SIZE must not exceed CONTEXT_SIZE"
    model.to(DEVICE)
    amp_dtype = pick_amp_dtype()

    # 4. train (or load cached weights)
    train_or_load(model, train_data, val_data, amp_dtype)

    # 5. let the trained model continue a Rust prompt (autoregressive sampling)
    prompt = "pub fn"
    print("\n--- generation ---")
    print(f"prompt      : {prompt!r}")
    print("continuation:")
    print(generate_sample(model, tokenizer, prompt, max_new_tokens=200))


if __name__ == "__main__":
    main()
