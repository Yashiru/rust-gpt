"""rustgpt, a from-scratch GPT trained on Rust, with a native Rust BPE tokenizer.

The tokenizer is implemented in Rust (PyO3) and exposed as `Tokenizer`; the GPT
model lives in `rustgpt.model`.
"""

from rustgpt._bpe import Tokenizer

__all__ = ["Tokenizer"]
