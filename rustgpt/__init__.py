"""rustgpt | a GPT trained from scratch on Rust source code.

The project's own BPE tokenizer lives in `rustgpt.tokenizer` (pure Python, the
readable reference). The `tokenizer-rs/` crate ships an optional drop-in
accelerator (`bpe_rs`) with the same API; `Tokenizer` below prefers it when it's
been built, and silently falls back to the Python implementation otherwise.

`TOKENIZER_BACKEND` says which one you got ("rust" or "python").
"""

try:
    from bpe_rs import Tokenizer  # ultra-fast Rust drop-in (tokenizer-rs/)

    TOKENIZER_BACKEND = "rust"
except ImportError:  # accelerator not built, use the pure-Python reference
    from rustgpt.tokenizer import Tokenizer

    TOKENIZER_BACKEND = "python"

__all__ = ["Tokenizer", "TOKENIZER_BACKEND"]
