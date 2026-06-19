"""Train (or load) the BPE tokenizer on the Rust corpus and cache it to JSON.

The tokenizer is (re)trained only if the JSON cache does not exist yet;
otherwise it is loaded from disk. Run with:

    uv run python main.py
"""

from pathlib import Path

from tqdm import tqdm

import tokenizer as tok

CORPUS_PATH = Path("data/rust_corpus.txt")  # concatenated .rs files
VOCAB_PATH = Path("tokenizer.json")         # trained tokenizer cache
VOCAB_SIZE = 8192                           # 256 base bytes + 7936 learned merges


def load_or_train():
    """Load the tokenizer from the JSON cache, or train it if the cache is absent."""
    if VOCAB_PATH.exists():
        print(f"Loading tokenizer from {VOCAB_PATH} ...")
        merges, vocab = tok.load(VOCAB_PATH)
        print(f"  loaded {len(vocab)} tokens ({len(merges)} merges)")
        return merges, vocab

    if not CORPUS_PATH.exists():
        raise SystemExit(
            f"Corpus not found: {CORPUS_PATH}\n"
            f"Drop a Rust corpus there (concatenated .rs files) and re-run."
        )

    text = CORPUS_PATH.read_text(encoding="utf-8")
    n_bytes = len(text.encode("utf-8"))
    print(f"Training BPE (vocab_size={VOCAB_SIZE}) on {CORPUS_PATH} "
          f"({n_bytes:,} bytes) ...")

    with tqdm(total=VOCAB_SIZE - 256, desc="merges") as bar:
        merges, vocab = tok.train(
            text, VOCAB_SIZE,
            on_step=lambda step, pair, count, vs: bar.update(1),
        )

    tok.save(VOCAB_PATH, merges, vocab,
             meta={"corpus": str(CORPUS_PATH), "corpus_bytes": n_bytes})
    print(f"  trained {len(vocab)} tokens, saved to {VOCAB_PATH}")
    return merges, vocab


def main():
    merges, vocab = load_or_train()

    # Round-trip sanity check on a small Rust snippet.
    sample = (
        "pub fn main() {\n"
        "    let xs: Vec<u32> = (0..10).collect();\n"
        '    println!("{:?}", xs);\n'
        "}\n"
    )
    ids = tok.encode(sample, merges)
    back = tok.decode(ids, vocab)
    n_bytes = len(sample.encode("utf-8"))

    print("\n--- round-trip on a Rust snippet ---")
    print(f"bytes        : {n_bytes}")
    print(f"tokens       : {len(ids)}")
    print(f"compression  : {n_bytes / len(ids):.2f} bytes/token")
    print(f"round-trip OK: {sample == back}")
    assert sample == back, "decode(encode(x)) != x"


if __name__ == "__main__":
    main()
