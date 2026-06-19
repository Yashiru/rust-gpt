"""Build a permissively-licensed Rust corpus from a Hugging Face code dataset.

Streams a code dataset, keeps only permissively-licensed Rust files, deduplicates,
and concatenates them into `data/rust_corpus.txt` (same format as the crates.io
downloader, so it drops straight into the pipeline). The amount is configurable
with `--max-bytes`.

    # ~3 GB from the-stack-dedup (gated: needs `huggingface-cli login` first)
    uv run python scripts/download_corpus_hf.py --max-bytes 3_100_000_000

    # alternative source, lighter gating
    uv run python scripts/download_corpus_hf.py --dataset github --max-bytes 1_500_000_000

the-stack is gated: accept the terms on its HF page and authenticate
(`huggingface-cli login`, or set HF_TOKEN) before running.
"""

import argparse
import hashlib
from collections import Counter
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

# Permissive SPDX-ish identifiers we accept (lowercased), same policy as crates.io.
PERMISSIVE = {
    "mit", "mit-0", "apache-2.0", "bsd-2-clause", "bsd-3-clause", "isc",
    "unlicense", "zlib", "0bsd", "cc0-1.0", "bsl-1.0",
}

# Per-dataset adapters: where the content / path / repo / license live, and any
# extra load_dataset kwargs needed to select Rust.
DATASETS = {
    # Rust subset is selected directly via data_dir; content is included inline.
    "thestack": {
        "path": "bigcode/the-stack-dedup",
        "load_kwargs": {"data_dir": "data/rust"},
        "content_keys": ("content",),
        "path_keys": ("max_stars_repo_path", "path"),
        "repo_keys": ("max_stars_repo_name", "repo_name"),
        "license_keys": ("max_stars_repo_licenses", "licenses", "license"),
    },
    # Multi-language; filter to Rust server-side (custom loader -> trust_remote_code).
    "github": {
        "path": "codeparrot/github-code",
        "load_kwargs": {"languages": ["Rust"], "trust_remote_code": True},
        "content_keys": ("code", "content"),
        "path_keys": ("path",),
        "repo_keys": ("repo_name",),
        "license_keys": ("license",),
    },
}


def _first(rec, keys, default=None):
    """Return the first present, non-empty value among `keys`."""
    for k in keys:
        v = rec.get(k)
        if v:
            return v
    return default


def licenses_of(rec, keys):
    """Normalize the record's license field to a lowercased list of identifiers."""
    raw = _first(rec, keys, [])
    if isinstance(raw, str):
        raw = [raw]
    return [str(x).strip().lower() for x in raw if str(x).strip()]


def is_permissive(lics, allowed):
    """True only if every detected license is permissive (no mixed/unknown)."""
    return bool(lics) and all(lic in allowed for lic in lics)


def build_corpus(args):
    spec = DATASETS[args.dataset]
    allowed = (set(l.strip().lower() for l in args.licenses.split(","))
               if args.licenses else PERMISSIVE)

    print(f"Streaming {spec['path']} (dataset={args.dataset}) ...")
    try:
        ds = load_dataset(spec["path"], split="train", streaming=True,
                          **spec["load_kwargs"])
    except Exception as e:
        raise SystemExit(
            f"Could not open the dataset: {e}\n"
            f"If it is gated (the-stack), accept its terms on the HF page and run "
            f"`huggingface-cli login` (or set HF_TOKEN), then retry."
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    seen_hashes = set()
    licenses = Counter()
    total = 0
    n_files = n_dups = skipped_license = skipped_size = 0

    with out_path.open("w", encoding="utf-8") as out, \
            tqdm(total=args.max_bytes, unit="B", unit_scale=True, desc="corpus") as bar:
        for rec in ds:
            if total >= args.max_bytes:
                break

            content = _first(rec, spec["content_keys"])
            if not content:
                continue

            lics = licenses_of(rec, spec["license_keys"])
            if not is_permissive(lics, allowed):
                skipped_license += 1
                continue

            data = content.encode("utf-8")
            if not (args.min_file_bytes <= len(data) <= args.max_file_bytes):
                skipped_size += 1
                continue

            h = hashlib.blake2b(data, digest_size=16).digest()
            if h in seen_hashes:
                n_dups += 1
                continue
            seen_hashes.add(h)

            repo = _first(rec, spec["repo_keys"], "?")
            rel = _first(rec, spec["path_keys"], "?")
            header = f"\n\n// ==== {repo}/{rel} ====\n\n"
            out.write(header)
            out.write(content)
            written = len(header.encode("utf-8")) + len(data)
            total += written
            n_files += 1
            licenses["+".join(sorted(set(lics)))] += 1
            bar.update(written)

    print("\n=== summary ===")
    print(f"files kept         : {n_files}")
    print(f"duplicate files    : {n_dups}")
    print(f"skipped (license)  : {skipped_license}")
    print(f"skipped (size)     : {skipped_size}")
    print(f"corpus bytes       : {total:,}")
    print(f"output             : {out_path}")
    print(f"licenses           : {dict(licenses)}")


def parse_args():
    p = argparse.ArgumentParser(description="Build a Rust corpus from a HF dataset.")
    p.add_argument("--dataset", choices=list(DATASETS), default="thestack",
                   help="source dataset (default: thestack = bigcode/the-stack-dedup)")
    p.add_argument("--max-bytes", type=int, default=1_500_000_000,
                   help="total corpus size cap in bytes (default: 1_500_000_000)")
    p.add_argument("--out", default="data/rust_corpus.txt",
                   help="output corpus path (default: data/rust_corpus.txt)")
    p.add_argument("--max-file-bytes", type=int, default=64_000,
                   help="skip .rs files larger than this (default: 64_000)")
    p.add_argument("--min-file-bytes", type=int, default=64,
                   help="skip files smaller than this (default: 64)")
    p.add_argument("--licenses", default="",
                   help="comma-separated allowlist override (default: built-in permissive set)")
    return p.parse_args()


if __name__ == "__main__":
    build_corpus(parse_args())
