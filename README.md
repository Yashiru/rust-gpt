# rust-gpt

A GPT trained **from scratch on Rust source code**, the Phase 1 capstone of my
ML systems journey. It builds on the from-scratch foundations in
[nn-forge](https://github.com/Yashiru/nn-forge) (micrograd, makemore, GPT, BPE).

> A GPT *trained on Rust*, implemented in PyTorch, with its from-scratch BPE
> tokenizer also reimplemented in Rust for speed.

## What's here

| Component | Location | Status |
|-----------|----------|--------|
| GPT model (`nn.Module`, multi-head attention, generation) | `rustgpt/model.py` | ✅ |
| Training loop (AdamW, GradScaler, dropout, early-stopping) | `rustgpt/train.py` | ✅ |
| BPE tokenizer (from scratch, the readable reference) | `rustgpt/tokenizer.py` | ✅ |
| Rust corpus downloader (crates.io, permissive licenses) | `scripts/download_corpus.py` | ✅ |
| **Optional** ultra-fast Rust tokenizer (PyO3 drop-in) | `tokenizer-rs/` (`bpe_rs`) | ✅ |
| Write-up: attention in my own words | n/a | 🚧 to come |

The whole pipeline, tokenizer included, is from scratch. The tokenizer is a
byte-pair encoder I wrote in Python (`rustgpt/tokenizer.py`); the `tokenizer-rs/`
crate is a **separate, optional accelerator** that reimplements it in Rust with
the same API and JSON format. `from rustgpt import Tokenizer` transparently picks
the Rust one when it's built and falls back to Python otherwise.

## Layout

```
rust-gpt/                          # the GPT project (Python)
├── pyproject.toml                 # Python project + uv workspace root
├── rustgpt/                       # main package
│   ├── tokenizer.py               # from-scratch BPE (reference implementation)
│   ├── model.py                   # the GPT (nn.Module)
│   ├── data.py                    # corpus -> token stream -> batches
│   └── train.py                   # training entrypoint
├── tokenizer-rs/                  # optional Rust accelerator (workspace member)
│   ├── Cargo.toml / src/lib.rs    #   BPE core + PyO3 bindings -> module `bpe_rs`
│   └── pyproject.toml             #   maturin build
├── scripts/
│   ├── download_corpus.py         # build the corpus from crates.io
│   └── benchmark.py               # Rust vs Python timing
└── tests/test_tokenizer.py        # Rust↔Python parity + round-trip
```

## Tokenizer performance

The Rust accelerator vs the pure-Python reference, on the 46 MB corpus (vocab 8192):

| Operation | Python | Rust | Speedup |
|-----------|-------:|-----:|--------:|
| Train     | ~693 s | **3.5 s** | **~199×** |
| Encode full corpus | (impractical) | **3.3 s** (14 MB/s) | n/a |
| `encode_batch` (rayon, parallel) | n/a | **1.0 s** | n/a |

Given the same merges, both implementations encode to **identical** token ids
(`tests/test_tokenizer.py`), so a `tokenizer.json` is interchangeable between them.

## Setup

Requires [uv](https://docs.astral.sh/uv/). The Rust accelerator additionally needs
a Rust toolchain ([rustup](https://rustup.rs)); `uv sync` builds it automatically.

```bash
uv sync                                  # install deps + build the Rust tokenizer
uv run python scripts/download_corpus.py # build data/rust_corpus.txt
uv run python -m rustgpt.train           # train tokenizer + GPT, then generate
uv run pytest                            # Rust↔Python parity tests
uv run python scripts/benchmark.py       # Rust vs Python timing
```

Using the tokenizer directly:

```python
from rustgpt import Tokenizer            # Rust if built, else pure Python

tok = Tokenizer.train(open("data/rust_corpus.txt").read(), vocab_size=8192)
ids = tok.encode("pub fn main() {}")
tok.decode(ids)                          # 'pub fn main() {}'
tok.save("tokenizer.json")               # shared JSON schema
```

## Corpus

`scripts/download_corpus.py` pulls high-quality, **permissively-licensed** Rust
from [crates.io](https://crates.io): it lists the most-downloaded crates via the
API (polite, identifying User-Agent), fetches each crate's source tarball from
the CDN, keeps idiomatic `.rs` files under a permissive license (MIT / Apache-2.0
/ BSD / ISC / …), deduplicates, and concatenates up to a configurable byte cap.

```bash
uv run python scripts/download_corpus.py --max-bytes 50_000_000 --top-n 300
```

The corpus and the download cache live under `data/` and are **not** committed;
the script is the reproducible source of truth.
