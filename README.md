# rust-gpt

A GPT trained **from scratch on Rust source code**, the Phase 1 capstone of my
ML systems journey. It builds on the from-scratch foundations in
[nn-forge](https://github.com/Yashiru/nn-forge) (micrograd → makemore → GPT → BPE).

> A GPT *trained on Rust*, implemented in PyTorch, not a GPT written in Rust.

## What's here

| Component | File | Status |
|-----------|------|--------|
| BPE tokenizer (train / encode / decode, JSON cache) | `tokenizer.py`, `main.py` | ✅ |
| Rust corpus downloader (crates.io, permissive licenses) | `download_corpus.py` | ✅ |
| GPT model, training & generation |, | 🚧 to come |
| Write-up: attention in my own words |, | 🚧 to come |

The tokenizer is my own Byte-Pair Encoding implementation (ported from
`nn-forge`), trained on the Rust corpus rather than relying on an off-the-shelf
tokenizer, so the whole pipeline, tokenizer included, is from scratch.

## Setup

Requires [uv](https://docs.astral.sh/uv/). Dependencies install on first run.

```bash
uv run python download_corpus.py     # build data/rust_corpus.txt (~25 MB)
uv run python main.py                # train the BPE tokenizer -> tokenizer.json
```

## Corpus

`download_corpus.py` pulls high-quality, **permissively-licensed** Rust from
[crates.io](https://crates.io): it lists the most-downloaded crates via the API
(polite, identifying User-Agent), fetches each crate's source tarball from the
CDN, keeps idiomatic `.rs` files under a permissive license (MIT / Apache-2.0 /
BSD / ISC / …), deduplicates, and concatenates up to a configurable byte cap.

```bash
uv run python download_corpus.py --max-bytes 5_000_000 --top-n 200
```

The corpus and the download cache live under `data/` and are **not** committed -
the script is the reproducible source of truth.
