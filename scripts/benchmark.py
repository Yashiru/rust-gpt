"""Benchmark the native Rust tokenizer against the pure-Python reference.

    python scripts/benchmark.py                  # 2 MB slice (Python stays tractable)
    python scripts/benchmark.py --bytes 8_000_000

Trains both implementations on the same slice and times train + encode, then
times the Rust tokenizer on the full corpus (where Python would be impractical).
"""

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tests"))

import reference_bpe as ref  # noqa: E402
from rustgpt import Tokenizer  # noqa: E402

CORPUS = REPO_ROOT / "data" / "rust_corpus.txt"
VOCAB_SIZE = 8192


def timed(fn):
    t0 = time.time()
    out = fn()
    return out, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bytes", type=int, default=2_000_000,
                    help="corpus slice for the head-to-head (default 2 MB)")
    args = ap.parse_args()

    if not CORPUS.exists():
        raise SystemExit(f"Corpus not found: {CORPUS} (run scripts/download_corpus.py)")

    full = CORPUS.read_text(encoding="utf-8")
    text = full[: args.bytes]
    nbytes = len(text.encode("utf-8"))
    print(f"head-to-head on {nbytes:,} bytes (vocab {VOCAB_SIZE})\n")

    # --- train ---
    (py_merges, _), py_train = timed(lambda: ref.train(text, VOCAB_SIZE))
    rust, rust_train = timed(lambda: Tokenizer.train(text, VOCAB_SIZE))
    print(f"train   Python {py_train:8.2f}s | Rust {rust_train:7.2f}s "
          f"| {py_train / rust_train:6.0f}x")

    # --- encode (same merges -> identical ids, asserted in the test suite) ---
    _, py_enc = timed(lambda: ref.encode(text, py_merges))
    _, rust_enc = timed(lambda: rust.encode(text))
    print(f"encode  Python {py_enc:8.2f}s | Rust {rust_enc:7.2f}s "
          f"| {py_enc / rust_enc:6.0f}x")

    # --- Rust on the full corpus ---
    full_bytes = len(full.encode("utf-8"))
    _, rust_full = timed(lambda: Tokenizer.train(full, VOCAB_SIZE))
    print(f"\nRust train on full corpus ({full_bytes / 1e6:.0f} MB): {rust_full:.2f}s")
    ids, rust_full_enc = timed(lambda: rust.encode(full))
    print(f"Rust encode full corpus: {rust_full_enc:.2f}s "
          f"({full_bytes / 1e6 / rust_full_enc:.1f} MB/s, {len(ids):,} tokens)")


if __name__ == "__main__":
    main()
