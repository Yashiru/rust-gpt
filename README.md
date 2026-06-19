# rust-gpt

A GPT trained **from scratch on Rust source code**, with an **ultra-fast BPE
tokenizer written in Rust** (exposed to Python via PyO3). The Phase 1 capstone of
my ML systems journey, it builds on the from-scratch foundations in
[nn-forge](https://github.com/Yashiru/nn-forge) (micrograd → makemore → GPT → BPE).

> A GPT *trained on Rust*, with a tokenizer *written in Rust* and a model in PyTorch.

## What's here

| Component | Location | Status |
|-----------|----------|--------|
| BPE tokenizer, Rust core + PyO3 bindings (train / encode / decode, JSON cache) | `src/lib.rs`, `rustgpt.Tokenizer` | ✅ |
| GPT model, `nn.Module`, multi-head attention, generation | `python/rustgpt/model.py` | ✅ |
| Training loop, AdamW, GradScaler, dropout, early-stopping | `python/rustgpt/train.py` | ✅ |
| Rust corpus downloader (crates.io, permissive licenses) | `scripts/download_corpus.py` | ✅ |
| Rust↔Python parity tests + benchmark | `tests/`, `scripts/benchmark.py` | ✅ |
| Write-up: attention in my own words |, | 🚧 to come |

The tokenizer is my own Byte-Pair Encoding implementation, first written in
Python (kept as `tests/reference_bpe.py`), then rewritten in Rust for speed. The
two are pinned together by a parity test suite: **given the same merges, they
encode to identical token ids**, so the same `tokenizer.json` works either way.

## Tokenizer performance

Native Rust vs the pure-Python reference, on the 46 MB corpus (vocab 8192):

| Operation | Python | Rust | Speedup |
|-----------|-------:|-----:|--------:|
| Train     | ~693 s | **3.5 s** | **~199×** |
| Encode full corpus | (impractical) | **3.3 s** (14 MB/s) |, |
| `encode_batch` (rayon, parallel) |, | **1.0 s** |, |

Training uses an incremental pair-count index (only words touched by a merge are
rescanned); batch encoding releases the GIL and parallelizes with rayon.

## Layout

```
rust-gpt/
├── Cargo.toml / src/lib.rs        # Rust BPE tokenizer + PyO3 bindings
├── pyproject.toml                 # maturin build backend
├── python/rustgpt/
│   ├── __init__.py                # exports Tokenizer
│   ├── tokenizer.py               # re-export of the native Tokenizer
│   ├── model.py                   # the GPT (nn.Module)
│   ├── data.py                    # corpus -> token stream -> batches
│   └── train.py                   # training entrypoint
├── scripts/
│   ├── download_corpus.py         # build the corpus from crates.io
│   └── benchmark.py               # Rust vs Python timing
└── tests/
    ├── reference_bpe.py           # the original pure-Python tokenizer
    └── test_tokenizer.py          # parity + round-trip tests
```

## Setup

Requires [uv](https://docs.astral.sh/uv/) and a Rust toolchain
([rustup](https://rustup.rs)). The native module is built with
[maturin](https://www.maturin.rs/).

```bash
uv run maturin develop --release        # build the Rust tokenizer into the venv
uv run python scripts/download_corpus.py # build data/rust_corpus.txt
uv run python -m rustgpt.train           # train tokenizer + GPT, then generate
uv run pytest                            # Rust↔Python parity tests
```

Using the tokenizer directly:

```python
from rustgpt import Tokenizer

tok = Tokenizer.train(open("data/rust_corpus.txt").read(), vocab_size=8192)
ids = tok.encode("pub fn main() {}")
tok.decode(ids)                          # 'pub fn main() {}'
tok.save("tokenizer.json")               # same JSON schema as the Python impl
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

The corpus and the download cache live under `data/` and are **not** committed -
the script is the reproducible source of truth.
