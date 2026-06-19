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
    # encode_u16_bytes returns the ids packed as bytes, so we never build a Python
    # list of hundreds of millions of ints (which would cost tens of GB of RAM).
    ids = np.frombuffer(tokenizer.encode_u16_bytes(text), dtype=np.uint16)
    ids_path.parent.mkdir(parents=True, exist_ok=True)
    ids.tofile(ids_path)
    print(f"  encoded {len(ids):,} tokens, cached to {ids_path}")
    return ids


def train_val_split(ids, val_frac):
    """Split the token stream into train/val, keeping it on the CPU as int32.

    At hundreds of millions of tokens, putting the whole stream on the GPU as
    int64 would eat several GB and leave no room for the model, so it stays
    host-side; `get_batch` copies only each minibatch to the device.
    """
    data = torch.from_numpy(ids.astype(np.int32))
    n = int((1 - val_frac) * len(data))
    return data[:n], data[n:]


def get_batch(data, block_size, batch_size, device):
    """Sample `batch_size` random contiguous windows of `block_size` tokens.

    The target `y` is `x` shifted right by one, so position t in x predicts the
    token at position t+1, every position in the block is a training example.
    Only the sampled minibatch is moved to the device (and cast to int64 there).
    """
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])          # (B, T) on CPU
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])  # (B, T), shifted by 1
    return (x.to(device, non_blocking=True).long(),
            y.to(device, non_blocking=True).long())
