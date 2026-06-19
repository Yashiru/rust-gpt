"""Parity tests: the Rust accelerator must match the pure-Python tokenizer.

`rustgpt.tokenizer.Tokenizer` is the canonical from-scratch implementation; the
Rust `bpe_rs.Tokenizer` (built from `tokenizer-rs/`) is an optional drop-in. The
critical guarantee is that, *given the same merges*, they encode to identical ids,
that's what makes them interchangeable and keeps an existing `tokenizer.json`
(and the caches/weights derived from it) valid regardless of which one produced it.
"""

from pathlib import Path

import pytest

from rustgpt.tokenizer import Tokenizer as PyTokenizer

bpe_rs = pytest.importorskip("bpe_rs", reason="Rust accelerator not built")
RustTokenizer = bpe_rs.Tokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent

# A small but varied Rust corpus: keywords, generics, lifetimes, numbers,
# punctuation, contractions, unicode, and assorted whitespace.
SAMPLE = """
pub fn main() {
    let xs: Vec<u32> = (0..10).collect();
    println!("{:?}", xs);
}

impl<'de> Deserialize<'de> for &'de str {
    type Error = Error;
    fn visit_str(self, v: &str) -> Result<Self::Value, E> { Ok(v) }
}

// it's a comment with don't/can't and numbers 12345 + 0xFF_u8
const GREETING: &str = "héllo, wörld, 你好";
"""

PROBES = [
    "pub fn main() {}",
    "let mut x: Vec<u32> = Vec::new();",
    "    indented\n\t\ttabs   and   spaces",
    "don't can't it's we're",
    "0xFF 123 4_096 3.14",
    "héllo wörld 你好 🦀",
    "",
    "a",
]


def test_encode_parity_given_same_merges(tmp_path):
    """Same merges -> identical ids from Rust and Python, on held-out probes."""
    py = PyTokenizer.train(SAMPLE, 512)
    path = tmp_path / "tok.json"
    py.save(str(path))
    rust = RustTokenizer.load(str(path))
    for s in PROBES:
        assert rust.encode(s) == py.encode(s), f"encode mismatch on {s!r}"


def test_rust_trained_loads_in_python(tmp_path):
    """A Rust-trained tokenizer is byte-for-byte loadable and matches in Python."""
    rust = RustTokenizer.train(SAMPLE, 512)
    path = tmp_path / "tok.json"
    rust.save(str(path))
    py = PyTokenizer.load(str(path))
    for s in PROBES:
        assert py.encode(s) == rust.encode(s), f"encode mismatch on {s!r}"


def test_decode_roundtrip():
    """decode(encode(x)) == x for valid UTF-8 text, on both implementations."""
    for cls in (PyTokenizer, RustTokenizer):
        tok = cls.train(SAMPLE, 512)
        for s in PROBES:
            assert tok.decode(tok.encode(s)) == s, f"{cls} round-trip failed on {s!r}"


def test_rust_train_compresses():
    """Rust-trained tokenizer actually compresses below raw bytes."""
    tok = RustTokenizer.train(SAMPLE, 512)
    assert 256 < tok.vocab_size <= 512  # small corpus: stops when no pair is left
    ids = tok.encode(SAMPLE)
    assert len(ids) < len(SAMPLE.encode("utf-8"))


def test_encode_batch_matches_encode():
    """encode_batch (parallel in Rust) equals calling encode one by one."""
    tok = RustTokenizer.train(SAMPLE, 512)
    assert tok.encode_batch(PROBES) == [tok.encode(s) for s in PROBES]


@pytest.mark.skipif(
    not (REPO_ROOT / "tokenizer.json").exists(),
    reason="no trained tokenizer.json present",
)
def test_parity_on_real_tokenizer():
    """If the real tokenizer.json exists, Rust and Python must encode it identically."""
    path = str(REPO_ROOT / "tokenizer.json")
    py = PyTokenizer.load(path)
    rust = RustTokenizer.load(path)
    probes = PROBES
    corpus = REPO_ROOT / "data" / "rust_corpus.txt"
    if corpus.exists():
        probes = probes + [corpus.read_text(encoding="utf-8")[:20000]]
    for s in probes:
        assert rust.encode(s) == py.encode(s)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
