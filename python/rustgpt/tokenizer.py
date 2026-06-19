"""The BPE tokenizer, implemented natively in Rust (PyO3).

This is a thin re-export so callers can `from rustgpt.tokenizer import Tokenizer`
without caring that the implementation is a compiled extension. The pure-Python
reference it replaces lives in `tests/reference_bpe.py` and is used only to assert
parity in the test suite.

    >>> tok = Tokenizer.train(text, vocab_size=8192)
    >>> ids = tok.encode("pub fn main() {}")
    >>> tok.decode(ids)
    'pub fn main() {}'
    >>> tok.save("tokenizer.json")          # same JSON schema as the old Python impl
    >>> tok = Tokenizer.load("tokenizer.json")
"""

from rustgpt._bpe import Tokenizer

__all__ = ["Tokenizer"]
