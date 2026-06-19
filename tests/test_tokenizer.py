"""Parity tests: the native Rust tokenizer must match the pure-Python reference.

The reference (`reference_bpe.py`) is the original from-scratch implementation.
The critical guarantee is that, *given the same merges*, Rust and Python encode
to identical ids, that's what keeps an existing `tokenizer.json` (and the cached
`rust_ids.bin` / trained weights derived from it) valid after the rewrite.
"""

from pathlib import Path

import pytest

import reference_bpe as ref
from rustgpt import Tokenizer

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


def _train_reference(text, vocab_size=512):
    merges, vocab = ref.train(text, vocab_size)
    return merges, vocab


def test_encode_parity_given_same_merges(tmp_path):
    """Same merges -> identical ids from Rust and Python, on held-out probes."""
    merges, vocab = _train_reference(SAMPLE, vocab_size=512)
    path = tmp_path / "tok.json"
    ref.save(path, merges, vocab)

    rust = Tokenizer.load(str(path))
    for s in PROBES:
        assert rust.encode(s) == ref.encode(s, merges), f"encode mismatch on {s!r}"


def test_decode_roundtrip(tmp_path):
    """decode(encode(x)) == x for valid UTF-8 text (Rust side)."""
    merges, vocab = _train_reference(SAMPLE, vocab_size=512)
    path = tmp_path / "tok.json"
    ref.save(path, merges, vocab)
    rust = Tokenizer.load(str(path))
    for s in PROBES:
        assert rust.decode(rust.encode(s)) == s, f"round-trip failed on {s!r}"


def test_rust_train_then_roundtrip():
    """Rust-trained tokenizer encodes/decodes losslessly and actually compresses."""
    tok = Tokenizer.train(SAMPLE, 512)
    # small corpus: training stops once no mergeable pair is left, so 256 < vocab <= 512
    assert 256 < tok.vocab_size <= 512
    ids = tok.encode(SAMPLE)
    assert tok.decode(ids) == SAMPLE
    # subword merges should beat raw bytes
    assert len(ids) < len(SAMPLE.encode("utf-8"))


def test_save_load_roundtrip(tmp_path):
    """A Rust-trained tokenizer survives save/load with identical encoding."""
    tok = Tokenizer.train(SAMPLE, 512)
    path = tmp_path / "tok.json"
    tok.save(str(path))
    reloaded = Tokenizer.load(str(path))
    for s in PROBES:
        assert reloaded.encode(s) == tok.encode(s)


def test_encode_batch_matches_encode():
    """encode_batch (parallel) must equal calling encode one by one."""
    tok = Tokenizer.train(SAMPLE, 512)
    assert tok.encode_batch(PROBES) == [tok.encode(s) for s in PROBES]


@pytest.mark.skipif(
    not (REPO_ROOT / "tokenizer.json").exists(),
    reason="no trained tokenizer.json present",
)
def test_parity_on_real_tokenizer():
    """If the real (Python-trained) tokenizer.json exists, Rust must match it."""
    path = REPO_ROOT / "tokenizer.json"
    merges, _ = ref.load(path)
    rust = Tokenizer.load(str(path))
    corpus = REPO_ROOT / "data" / "rust_corpus.txt"
    probes = PROBES
    if corpus.exists():
        probes = probes + [corpus.read_text(encoding="utf-8")[:20000]]
    for s in probes:
        assert rust.encode(s) == ref.encode(s, merges)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
