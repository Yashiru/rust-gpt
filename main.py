from pathlib import Path
from tqdm import tqdm
import numpy as np
import tokenizer as tok
from gpt import GPT
import torch
import torch.nn.functional as F

CORPUS_PATH = Path("data/rust_corpus.txt")  # concatenated .rs files
IDS_PATH = Path("data/rust_ids.bin")        # corpus encoded to token ids (uint16 cache)
WEIGHTS_PATH = Path("weights.pt")           # trained model cache
VOCAB_PATH = Path("tokenizer.json")         # trained tokenizer cache
VOCAB_SIZE = 8192                           # 256 base bytes + 7936 learned merges
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # train on the GPU when available

STEPS = 5000
LR = 1e-3
LR_FINE = 1e-4      # after 60% of the steps, drop the learning rate
BATCH_SIZE = 64
BLOCK_SIZE = 256    # training context length (must be <= the model's context_size)
VAL_FRAC = 0.1      # fraction of the token stream held out for validation
EVAL_INTERVAL = 250 # estimate train/val loss every N steps
EVAL_ITERS = 50     # how many batches to average when estimating loss
DROPOUT = 0.2       # regularization: drop this fraction of activations during training
WEIGHT_DECAY = 0.1  # AdamW L2 penalty (applied to matmul/embedding weights only)
EARLY_STOP_PATIENCE = 5  # stop if val loss hasn't improved for this many evals


def load_or_train():
    """Load the tokenizer from the JSON cache, or train it if the cache is absent."""
    if VOCAB_PATH.exists():
        print(f"Loading tokenizer from {VOCAB_PATH} ...")
        merges, vocab = tok.load(VOCAB_PATH)
        print(f"  loaded {len(vocab)} tokens ({len(merges)} merges)")
        return merges, vocab

    if not CORPUS_PATH.exists():
        raise SystemExit(
            f"Corpus not found: {CORPUS_PATH}\n"
            f"Drop a Rust corpus there (concatenated .rs files) and re-run."
        )

    text = CORPUS_PATH.read_text(encoding="utf-8")
    n_bytes = len(text.encode("utf-8"))
    print(f"Training BPE (vocab_size={VOCAB_SIZE}) on {CORPUS_PATH} "
          f"({n_bytes:,} bytes) ...")

    with tqdm(total=VOCAB_SIZE - 256, desc="merges") as bar:
        merges, vocab = tok.train(
            text, VOCAB_SIZE,
            on_step=lambda step, pair, count, vs: bar.update(1),
        )

    tok.save(VOCAB_PATH, merges, vocab,
             meta={"corpus": str(CORPUS_PATH), "corpus_bytes": n_bytes})
    print(f"  trained {len(vocab)} tokens, saved to {VOCAB_PATH}")
    return merges, vocab


def load_or_encode_corpus(merges):
    """Encode the whole corpus to a flat array of token ids, caching it on disk.

    BPE encoding in pure Python is slow, so we do it once and cache the result as
    a uint16 binary (vocab < 65536 fits). Delete `data/rust_ids.bin` whenever the
    tokenizer is retrained, otherwise this stale cache no longer matches the vocab.
    """
    if IDS_PATH.exists():
        print(f"Loading encoded corpus from {IDS_PATH} ...")
        ids = np.fromfile(IDS_PATH, dtype=np.uint16)
        print(f"  loaded {len(ids):,} tokens")
        return ids

    print(f"Encoding the corpus {CORPUS_PATH} (one-time; pure-Python BPE is slow) ...")
    text = CORPUS_PATH.read_text(encoding="utf-8")
    ids = np.array(tok.encode(text, merges), dtype=np.uint16)
    IDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    ids.tofile(IDS_PATH)
    print(f"  encoded {len(ids):,} tokens, cached to {IDS_PATH}")
    return ids


def train_val_split(ids):
    """Move the token stream to the device as one 1D tensor, split into train/val.

    Indices into Embedding / targets for cross_entropy must be int64. The whole
    stream lives on the GPU once (a few hundred MB); batches are sliced from it.
    """
    data = torch.from_numpy(ids.astype(np.int64)).to(DEVICE)
    n = int((1 - VAL_FRAC) * len(data))
    return data[:n], data[n:]


def get_batch(data, block_size, batch_size):
    """Sample `batch_size` random contiguous windows of `block_size` tokens.

    The target `y` is `x` shifted right by one, so position t in x predicts the
    token at position t+1, every position in the block is a training example.
    """
    ix = torch.randint(len(data) - block_size, (batch_size,), device=data.device)
    x = torch.stack([data[i:i + block_size] for i in ix])          # (B, T)
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])  # (B, T), shifted by 1
    return x, y


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
            xb, yb = get_batch(data, BLOCK_SIZE, BATCH_SIZE)
            losses[k] = compute_loss(model, xb, yb, amp_dtype).item()
        means.append(losses.mean().item())
    model.train()
    return means[0], means[1]


def generate_sample(model, merges, vocab, prompt, max_new_tokens=200,
                    temperature=0.8, top_k=50):
    """Encode `prompt`, let the model continue it, and decode back to Rust text."""
    model.eval()
    ids = tok.encode(prompt, merges)
    idx = torch.tensor([ids], dtype=torch.int64, device=DEVICE)  # (1, T)
    out = model.generate(idx, max_new_tokens, temperature=temperature, top_k=top_k)
    model.train()
    return tok.decode(out[0].tolist(), vocab)


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
    decay_at = int(0.6 * STEPS)
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
            lr = LR_FINE if step >= decay_at else LR
            for group in optimizer.param_groups:
                group["lr"] = lr
            xb, yb = get_batch(train_data, BLOCK_SIZE, BATCH_SIZE)
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
    """One standard optimization step: zero_grad -> backward -> step, with AMP scaling."""
    optimizer.zero_grad(set_to_none=True)
    loss = compute_loss(model, xb, yb, amp_dtype)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    return loss

def main():
    # 1. tokenizer: load or train
    merges, vocab = load_or_train()

    # 2. dataset: encode (cached) -> 1D token stream -> train/val split
    ids = load_or_encode_corpus(merges)
    train_data, val_data = train_val_split(ids)
    print(f"  train {len(train_data):,} tokens | val {len(val_data):,} tokens "
          f"| block {BLOCK_SIZE} | batch {BATCH_SIZE}")

    # 3. model
    model = GPT(vocab_size=len(vocab), n_embd=256, n_head=4, n_layer=4,
                context_size=1024, dropout=DROPOUT)
    assert BLOCK_SIZE <= 1024, "BLOCK_SIZE must not exceed the model's context_size"
    model.to(DEVICE)
    amp_dtype = pick_amp_dtype()

    # 4. train (or load cached weights)
    train_or_load(model, train_data, val_data, amp_dtype)

    # 5. let the trained model continue a Rust prompt (autoregressive sampling)
    prompt = "pub fn "
    print("\n--- generation ---")
    print(f"prompt      : {prompt!r}")
    print("continuation:")
    print(generate_sample(model, merges, vocab, prompt, max_new_tokens=200))


if __name__ == "__main__":
    main()
