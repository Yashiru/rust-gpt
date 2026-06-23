# rust-gpt

A personal experiment: a modern GPT trained from scratch on Rust source code, in
PyTorch, with its byte-pair tokenizer also written from scratch (first in Python to
understand it, then in Rust for speed). I built this to actually understand how a small
language model works end to end, from raw bytes to generated code, rather than wiring
together a stack of libraries.

It is a companion to [nn-forge](https://github.com/Yashiru/nn-forge), my other
from-scratch experimentation repo.

## Two builds, from scratch

The repo is really two things I wanted to build myself and understand:

1. **The model**: a decoder-only transformer with a modern architecture, trained to
   roughly Chinchilla-optimal size on around 900M tokens of Rust.
2. **The tokenizer**: a BPE tokenizer I first wrote in Python to learn how modern
   tokenization works, then rewrote in Rust when the Python version turned out to be
   far too slow, for about a 199x speedup.

## The model

A 40M-parameter decoder transformer trained from scratch on permissively-licensed Rust.

| | |
|---|---|
| Parameters | ~40.5M (512 dim, 8 heads, 10 layers, vocab 16384) |
| Training data | ~900M tokens of Rust (the-stack-dedup, permissive licenses) |
| Context length | 1024 |
| Validation perplexity | **~4.6** (val loss 1.535) |
| Train / val gap | **~0** (val loss 1.546, no overfitting) |
| Hardware | single RTX 2080 Ti, fp16, ~3h |

The train/val gap being essentially zero is the result I cared about most: 40M params on
900M tokens is the right balance, so the model **generalizes instead of memorizing**.

### What it generates

Raw sampled output (temperature 0.8, top-k 50), continuing a prompt. Not cherry-picked
to be perfect, this is a representative sample:

```rust
pub fn read_data(buf: &mut [u8]) -> ReadData {
    let mut buf = [0u8; 4];
    buf[read_all(&mut buf).expect("read all data bytes");
    ReadData {
        buffer: buf,
        len: buf.len(),
    }
}

pub fn read_data(&mut self) -> &mut [u8] {
    let mut data = [0u8; 4];
    data.copy_from_slice(&self.buffer[self.len..]);
    data
}
```

It has learned real structure: struct literals, method signatures, borrowing (`&mut`),
slice syntax, doc comments, trait impls (`Iterator` with `type Item` / `fn next`). It is
not always valid Rust, but for a 40M model trained from scratch the syntax and idioms are
solid and there is no degenerate repetition or nonsense.

### Architecture

A Llama-style decoder, all written by hand in [`rustgpt/model.py`](rustgpt/model.py):

- **RoPE** rotary position embeddings instead of learned absolute positions
- **SwiGLU** gated feed-forward instead of a ReLU MLP
- **RMSNorm** pre-norm instead of LayerNorm
- **Weight tying** between the token embedding and the output head
- Fused QKV, bias-free linears, scaled residual-projection init
- Attention through the fused `scaled_dot_product_attention`, with a readable
  explicit path kept for reference

Training recipe: AdamW with separate weight-decay groups, cosine learning rate with
linear warmup, gradient clipping, dropout, and fp16 with a gradient scaler.

There is a longer write-up of how attention works, in my own words and mapped onto this
code, in [docs/attention.md](docs/attention.md).

## The tokenizer: from a Python prototype to a Rust accelerator

I wanted my own tokenizer rather than an off-the-shelf one, so I started in Python
([`rustgpt/tokenizer.py`](rustgpt/tokenizer.py)) to learn exactly how a modern subword
tokenizer works: GPT-2 style regex pre-tokenization, then byte-pair merges learned
inside each chunk, serialized to a single JSON file. That version is the readable
reference, and it is correct, but it is slow: training on a real corpus takes minutes,
and encoding hundreds of millions of tokens is impractical.

So I rewrote the same algorithm in Rust ([`tokenizer-rs/`](tokenizer-rs/)) with PyO3
bindings, `fancy-regex` for the GPT-2 lookahead pattern, and `rayon` for parallel batch
encoding. It exposes the exact same API and the same JSON format, so it is a transparent
drop-in: `from rustgpt import Tokenizer` picks the Rust module (`bpe_rs`) when it is
built and falls back to the Python one otherwise.

The two are checked for parity in [`tests/test_tokenizer.py`](tests/test_tokenizer.py):
given the same merges they produce identical token ids, so a `tokenizer.json` is
interchangeable between them.

Measured on a 46 MB corpus (vocab 8192), with `scripts/benchmark.py`:

| Operation | Python | Rust | Speedup |
|-----------|-------:|-----:|--------:|
| Train | ~693 s | **3.5 s** | **~199x** |
| Encode full corpus | (impractical) | **3.3 s** (14 MB/s) | n/a |
| `encode_batch` (parallel, rayon) | n/a | **1.0 s** | n/a |

## What's here

| Component | Location |
|-----------|----------|
| GPT model (modern decoder, RoPE / SwiGLU / RMSNorm, generation) | `rustgpt/model.py` |
| Training loop (AdamW, cosine LR, GradScaler, dropout, early-stopping) | `rustgpt/train.py` |
| Dataset plumbing (corpus to cached token stream to batches) | `rustgpt/data.py` |
| BPE tokenizer, from-scratch Python reference | `rustgpt/tokenizer.py` |
| BPE tokenizer, Rust accelerator (PyO3 drop-in, `bpe_rs`) | `tokenizer-rs/` |
| Corpus downloaders (the-stack on HF, and crates.io) | `scripts/` |
| Attention write-up | `docs/attention.md` |

## Layout

```
rust-gpt/
├── pyproject.toml                 # Python project + uv workspace root
├── rustgpt/                       # main package
│   ├── tokenizer.py               # from-scratch BPE (readable reference)
│   ├── model.py                   # the GPT (nn.Module)
│   ├── data.py                    # corpus -> token stream -> batches
│   └── train.py                   # training entrypoint
├── tokenizer-rs/                  # optional Rust accelerator (workspace member)
│   ├── Cargo.toml / src/lib.rs    #   BPE core + PyO3 bindings -> module bpe_rs
│   └── pyproject.toml             #   maturin build
├── scripts/
│   ├── download_corpus_hf.py      # build the corpus from the-stack (HF)
│   ├── download_corpus.py         # build the corpus from crates.io
│   └── benchmark.py               # Rust vs Python timing
├── docs/attention.md              # how attention works, in my own words
└── tests/test_tokenizer.py        # Rust/Python parity + round-trip
```

## Setup

Requires [uv](https://docs.astral.sh/uv/). The Rust accelerator additionally needs a
Rust toolchain ([rustup](https://rustup.rs)); `uv sync` builds it automatically.

```bash
uv sync                                     # install deps + build the Rust tokenizer
uv run python scripts/download_corpus_hf.py # build the Rust corpus
uv run python -m rustgpt.train              # train tokenizer + GPT, then generate
uv run pytest                               # Rust/Python parity tests
uv run python scripts/benchmark.py          # Rust vs Python timing
```

Using the tokenizer directly:

```python
from rustgpt import Tokenizer                # Rust if built, else pure Python

tok = Tokenizer.train(open("data/rust_corpus.txt").read(), vocab_size=8192)
ids = tok.encode("pub fn main() {}")
tok.decode(ids)                              # 'pub fn main() {}'
tok.save("tokenizer.json")                   # shared JSON schema
```

## Corpus

The corpus is high-quality, permissively-licensed Rust only. Two downloaders build it:

- [`scripts/download_corpus_hf.py`](scripts/download_corpus_hf.py) streams
  the-stack-dedup from Hugging Face (license-filtered, deduplicated), which is how the
  ~900M-token training set was built. Needs a HF account and accepting the dataset terms.
- [`scripts/download_corpus.py`](scripts/download_corpus.py) is a self-contained
  alternative that pulls the most-downloaded crates from [crates.io](https://crates.io):
  it lists them via the API (polite, identifying User-Agent), fetches each source
  tarball from the CDN, keeps idiomatic `.rs` files under a permissive license
  (MIT / Apache-2.0 / BSD / ISC and similar), deduplicates, and concatenates up to a
  configurable byte cap.

```bash
uv run python scripts/download_corpus.py --max-bytes 50_000_000 --top-n 300
```

The corpus, the download cache, and the trained weights live under `data/` (and
`weights.pt`) and are not committed. The scripts are the reproducible source of truth.
