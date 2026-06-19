"""Byte-Pair Encoding tokenizer (from scratch).

Ported from the Phase 1 step-08 tokenizer (nn-forge) and made self-contained
for this project. GPT-2 style: the text is first split with a regex, then BPE
merges are learned *inside* each chunk. Pairs are counted over the unique words
weighted by their frequency, and merges never cross a word boundary.

The trained tokenizer is fully described by `merges` (which pair maps to which
new id, in learned order) and `vocab` (id -> the bytes it stands for). Both are
serialized to a single JSON file via `save`/`load`.
"""

import json
from pathlib import Path

import regex as re

# GPT-2 / GPT-4 style pre-tokenization pattern.
GPT2_PAT = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train(text, vocab_size, on_step=None):
    """Train BPE merges on `text` up to `vocab_size`, returning (merges, vocab).

    Pass `on_step` to receive (step, pair, count, vocab_size) callbacks, handy
    for a progress bar.
    """
    from collections import Counter

    word_freqs = Counter(re.findall(GPT2_PAT, text))
    ids = {tuple(w.encode("utf-8")): f for w, f in word_freqs.items()}
    merges, vocab = {}, {i: bytes([i]) for i in range(256)}
    for k in range(vocab_size - 256):
        stats = get_stats(ids)
        if not stats:
            break  # no pair left to merge
        pair = max(stats, key=stats.get)  # the most frequent pair
        idx = 256 + k
        ids = merge_vocab(ids, pair, idx)
        merges[pair] = idx
        vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
        if on_step:
            on_step(k + 1, pair, stats[pair], len(vocab))
    return merges, vocab


def get_stats(ids):
    """Count adjacent pairs across all words, weighted by word frequency."""
    counts = {}
    for word, freq in ids.items():
        for pair in zip(word, word[1:]):
            counts[pair] = counts.get(pair, 0) + freq  # add freq, not 1
    return counts


def merge_vocab(ids, pair, idx):
    """Apply one merge to every word in the frequency table."""
    new_ids = {}
    for word, freq in ids.items():
        new_ids[merge(word, pair, idx)] = freq  # merge() returns a tuple -> dict key
    return new_ids


def merge(ids, pair, idx):
    """Replace every occurrence of `pair` in `ids` with the single token `idx`."""
    new_ids, i = tuple(), 0
    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
            new_ids += (idx,)
            i += 2
        else:
            new_ids += (ids[i],)
            i += 1
    return new_ids


# --------------------------------------------------------------------------- #
# Encode / decode
# --------------------------------------------------------------------------- #
def encode(text, merges):
    """Encode a string into token ids, chunk by chunk (same split as training)."""
    out = []
    for chunk in re.findall(GPT2_PAT, text):
        out.extend(encode_chunk(chunk.encode("utf-8"), merges))
    return out


def encode_chunk(chunk, merges):
    """Apply BPE to the bytes of a single chunk.

    On each pass we merge the present pair whose merge was learned earliest
    (lowest idx), until no known pair remains.
    """
    ids = list(chunk)
    while len(ids) >= 2:
        pairs = set(zip(ids, ids[1:]))
        pair = min(pairs, key=lambda p: merges.get(p, float("inf")))
        if pair not in merges:
            break  # none of the remaining pairs has a learned merge
        ids = merge(ids, pair, merges[pair])
    return ids


def decode(ids, vocab):
    """Decode a list of token ids back into a string."""
    tokens = [vocab[i] for i in ids]
    # errors="replace": generated ids can land on incomplete UTF-8, so we
    # replace instead of crashing.
    return b"".join(tokens).decode("utf-8", errors="replace")


def count_tokens(text, merges):
    """Total tokens needed to encode `text`, encoding each unique chunk once."""
    from collections import Counter

    total = 0
    for chunk, freq in Counter(re.findall(GPT2_PAT, text)).items():
        total += len(encode_chunk(chunk.encode("utf-8"), merges)) * freq
    return total


# --------------------------------------------------------------------------- #
# Persistence (JSON)
# --------------------------------------------------------------------------- #
def save(path, merges, vocab, meta=None):
    """Serialize the tokenizer to a single JSON file.

    `merges` are stored in learned order as [p0, p1, idx] triples; `vocab` maps
    each id to the hex of the bytes it stands for. `meta` is an optional dict of
    free-form metadata (corpus name, size, ...).
    """
    data = {
        "pattern": GPT2_PAT.pattern,
        "vocab_size": len(vocab),
        "merges": [[p0, p1, idx] for (p0, p1), idx in merges.items()],
        "vocab": {str(idx): tok.hex() for idx, tok in vocab.items()},
    }
    if meta:
        data["meta"] = meta
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")


def load(path):
    """Load a tokenizer saved by `save`, returning (merges, vocab)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    merges = {(p0, p1): idx for p0, p1, idx in data["merges"]}
    vocab = {int(idx): bytes.fromhex(h) for idx, h in data["vocab"].items()}
    return merges, vocab
