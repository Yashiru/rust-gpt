# tokenizer-rs (`bpe_rs`)

An ultra-fast Rust reimplementation of rustgpt's pure-Python BPE tokenizer
(`rustgpt/tokenizer.py`). Same algorithm, same JSON format, same API, just
~200× faster training and parallel batch encoding. It's an **optional
accelerator**: the GPT project runs fine without it (falling back to Python).

Exposes a single `Tokenizer` class to Python via [PyO3](https://pyo3.rs):

```python
from bpe_rs import Tokenizer
tok = Tokenizer.train(text, vocab_size=8192)   # builds on the same JSON schema
ids = tok.encode("pub fn main() {}")
tok.encode_batch(list_of_strings)              # parallel (rayon, GIL released)
tok.save("tokenizer.json"); Tokenizer.load("tokenizer.json")
```

## Build

Built automatically as a workspace member when you `uv sync` from the repo root.
Standalone:

```bash
maturin develop --release      # from this directory
```

## Implementation notes

- **Pre-tokenization** uses `fancy-regex` because the GPT-2 pattern contains a
  lookahead (`\s+(?!\S)`) that the standard `regex` crate doesn't support.
- **Training** keeps a global adjacent-pair count plus an index of which words
  contain each pair, so a merge only rescans the affected words (not the corpus).
- **Parity** with the Python reference is enforced by `tests/test_tokenizer.py`:
  given the same merges, both encode to identical ids.
