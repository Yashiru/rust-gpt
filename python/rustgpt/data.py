"""Dataset plumbing: corpus -> cached token stream -> random training batches."""

from pathlib import Path

import numpy as np
import torch


def load_or_encode_corpus(tokenizer, corpus_path, ids_path):
    """Encode the whole corpus to a flat array of token ids, caching it on disk.

    Encoding is cached as a uint16 binary (vocab < 65536 fits). Delete the cache
    whenever the tokenizer is retrained, otherwise it no longer matches the vocab.
    """
    ids_path = Path(ids_path)
    if ids_path.exists():
        print(f"Loading encoded corpus from {ids_path} ...")
        ids = np.fromfile(ids_path, dtype=np.uint16)
        print(f"  loaded {len(ids):,} tokens")
        return ids

    print(f"Encoding the corpus {corpus_path} ...")
    text = Path(corpus_path).read_text(encoding="utf-8")
    ids = np.array(tokenizer.encode(text), dtype=np.uint16)
    ids_path.parent.mkdir(parents=True, exist_ok=True)
    ids.tofile(ids_path)
    print(f"  encoded {len(ids):,} tokens, cached to {ids_path}")
    return ids


def train_val_split(ids, val_frac, device):
    """Move the token stream to the device as one 1D tensor, split into train/val.

    Indices into Embedding / targets for cross_entropy must be int64. The whole
    stream lives on the device once (a few hundred MB); batches are sliced from it.
    """
    data = torch.from_numpy(ids.astype(np.int64)).to(device)
    n = int((1 - val_frac) * len(data))
    return data[:n], data[n:]


def get_batch(data, block_size, batch_size):
    """Sample `batch_size` random contiguous windows of `block_size` tokens.

    The target `y` is `x` shifted right by one, so position t in x predicts the
    token at position t+1, every position in the block is a training example.
    """
    ix = torch.randint(len(data) - block_size, (batch_size,), device=data.device)
    x = torch.stack([data[i:i + block_size] for i in ix])          # (B, T)
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])  # (B, T), shifted by 1
    return x, y
