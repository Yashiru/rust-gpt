"""Byte-Pair Encoding tokenizer, the from-scratch reference implementation.

This is the canonical, readable tokenizer of the project. GPT-2 style: the text
is first split with a regex, then BPE merges are learned *inside* each chunk
(merges never cross a word boundary). A trained tokenizer is fully described by
its `merges` (which pair maps to which new id, in learned order) and `vocab`
(id -> the bytes it stands for), serialized to a single JSON file.

The Rust crate in `tokenizer-rs/` (`bpe_rs`) is an optional drop-in accelerator
with the *same* API and JSON format, see `rustgpt/__init__.py`, which prefers it
when built. Parity between the two is enforced by `tests/test_tokenizer.py`.
"""

import json
from collections import Counter
from pathlib import Path

import regex as re

# GPT-2 / GPT-4 style pre-tokenization pattern.
GPT2_PAT = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


def _get_stats(ids):
    """Count adjacent pairs across all words, weighted by word frequency."""
    counts = {}
    for word, freq in ids.items():
        for pair in zip(word, word[1:]):
            counts[pair] = counts.get(pair, 0) + freq
    return counts


def _merge(ids, pair, idx):
    """Replace every non-overlapping occurrence of `pair` in `ids` with `idx`."""
    out, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(idx)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return tuple(out)


class Tokenizer:
    """A trained BPE tokenizer (pure-Python reference implementation)."""

    def __init__(self, merges, vocab, pattern=GPT2_PAT.pattern):
        self.merges = merges        # {(p0, p1): idx} in learned order
        self.vocab = vocab          # {idx: bytes}
        self.pattern = pattern

    # ----------------------------------------------------------------- train #
    @classmethod
    def train(cls, text, vocab_size):
        """Learn BPE merges on `text` up to `vocab_size`, returning a Tokenizer."""
        word_freqs = Counter(re.findall(GPT2_PAT, text))
        ids = {tuple(w.encode("utf-8")): f for w, f in word_freqs.items()}
        merges, vocab = {}, {i: bytes([i]) for i in range(256)}
        for k in range(vocab_size - 256):
            stats = _get_stats(ids)
            if not stats:
                break  # no pair left to merge
            pair = max(stats, key=stats.get)  # the most frequent pair
            idx = 256 + k
            ids = {_merge(word, pair, idx): freq for word, freq in ids.items()}
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
        return cls(merges, vocab)

    # --------------------------------------------------------- encode/decode #
    def _encode_chunk(self, chunk):
        ids = list(chunk)
        while len(ids) >= 2:
            pair = min(zip(ids, ids[1:]), key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break  # none of the remaining pairs has a learned merge
            ids = list(_merge(ids, pair, self.merges[pair]))
        return ids

    def encode(self, text):
        """Encode a string into token ids, chunk by chunk (same split as training)."""
        out = []
        for chunk in re.findall(GPT2_PAT, text):
            out.extend(self._encode_chunk(chunk.encode("utf-8")))
        return out

    def encode_batch(self, texts):
        """Encode many strings (the Rust drop-in does this in parallel)."""
        return [self.encode(t) for t in texts]

    def decode(self, ids):
        """Decode token ids back into a string (lossy on incomplete UTF-8)."""
        data = b"".join(self.vocab[i] for i in ids)
        return data.decode("utf-8", errors="replace")

    @property
    def vocab_size(self):
        return len(self.vocab)

    # -------------------------------------------------------------- persist #
    def save(self, path):
        """Serialize to a single JSON file (schema shared with `bpe_rs`)."""
        data = {
            "pattern": self.pattern,
            "vocab_size": len(self.vocab),
            "merges": [[p0, p1, idx] for (p0, p1), idx in self.merges.items()],
            "vocab": {str(idx): tok.hex() for idx, tok in self.vocab.items()},
        }
        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path):
        """Load a tokenizer saved by `save` (or by the Rust `bpe_rs`)."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        merges = {(p0, p1): idx for p0, p1, idx in data["merges"]}
        vocab = {int(idx): bytes.fromhex(h) for idx, h in data["vocab"].items()}
        return cls(merges, vocab, data.get("pattern", GPT2_PAT.pattern))

    def __repr__(self):
        return f"Tokenizer(vocab_size={len(self.vocab)})"


__all__ = ["Tokenizer"]
