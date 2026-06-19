"""Download a high-quality, permissively-licensed Rust corpus from crates.io.

Strategy (see plan): use the crates.io API only to list the most-downloaded
crates (polite: <=1 req/s, identifying User-Agent), then pull each crate's
source tarball from the unrate-limited CDN, keep the idiomatic `.rs` files under
a permissive license, deduplicate, and concatenate up to a configurable byte cap.

Run with:

    uv run python download_corpus.py                  # ~25 MB corpus
    uv run python download_corpus.py --max-bytes 2_000_000   # quick smoke test

crates.io data access policy: https://crates.io/data-access
"""

import argparse
import gzip
import hashlib
import io
import tarfile
import time
import tomllib
from collections import Counter
from pathlib import Path

import requests

API = "https://crates.io/api/v1/crates"
CDN = "https://static.crates.io/crates"
USER_AGENT = "rust-gpt-corpus-builder/0.1 (+https://github.com/Yashiru/rust-gpt)"

# Curated, idiomatic library crates pulled in first (on-brand: candle is the
# Rust ML framework from the course). Order = priority.
SEED_CRATES = [
    "candle-core", "candle-nn", "tokio", "serde", "serde_json", "clap",
    "rayon", "anyhow", "thiserror", "regex", "itertools", "rand", "tracing",
    "bytes", "hyper", "axum", "reqwest", "indexmap", "hashbrown", "crossbeam",
]

# Permissive SPDX identifiers we accept (lowercased). "WITH" exceptions on an
# otherwise-permissive license are tolerated (handled in is_permissive).
PERMISSIVE = {
    "mit", "mit-0", "apache-2.0", "bsd-2-clause", "bsd-3-clause", "isc",
    "unlicense", "zlib", "0bsd", "cc0-1.0", "apache-2.0 with llvm-exception",
}

# FFI / binding / generated-code crates: valid Rust but low learning value and
# huge repetitive files. Skipped by exact name or the common `*-sys` suffix.
CRATE_DENYLIST = {
    "windows", "windows-sys", "windows-targets", "libc", "web-sys", "js-sys",
    "winapi", "openssl-sys",
}

KEEP_PREFIXES = ("src/", "examples/")   # within the crate root (after the version dir)
SKIP_DIRS = ("tests/", "benches/", "target/")


def fetch_top_crate_versions(top_n, session):
    """Return {crate_name: version} for the top-N crates by all-time downloads.

    Only a handful of paginated API calls (<=1 req/s), the heavy lifting (the
    tarballs) goes through the CDN, which has no rate limit.
    """
    out = {}
    page, per_page = 1, 100
    while len(out) < top_n:
        resp = session.get(
            API,
            params={"sort": "downloads", "per_page": per_page, "page": page},
            timeout=30,
        )
        resp.raise_for_status()
        crates = resp.json().get("crates", [])
        if not crates:
            break
        for c in crates:
            version = c.get("max_stable_version") or c.get("newest_version")
            if version:
                out[c["id"]] = version
        page += 1
        time.sleep(1.1)  # crates.io API: max 1 request/second
    return out


def is_denied(name):
    return name in CRATE_DENYLIST or name.endswith("-sys")


def download_crate(name, version, cache_dir, session):
    """Return the `.crate` tarball bytes, from the local cache or the CDN."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / f"{name}-{version}.crate"
    if cached.exists():
        return cached.read_bytes()
    url = f"{CDN}/{name}/{name}-{version}.crate"
    resp = session.get(url, timeout=60)
    resp.raise_for_status()
    cached.write_bytes(resp.content)
    return resp.content


def is_permissive(spdx):
    """True if the SPDX expression offers at least one fully-permissive choice.

    SPDX semantics: OR = pick one sub-expression, AND = all required. So we
    accept iff some OR-group is entirely permissive. Old `A/B` syntax = OR.
    """
    if not spdx:
        return False
    expr = spdx.replace("/", " OR ").lower()
    for group in expr.split(" or "):
        terms = [t.strip() for t in group.split(" and ") if t.strip()]
        if terms and all(t in PERMISSIVE for t in terms):
            return True
    return False


def open_crate_tar(blob):
    """Open a `.crate` (gzipped tar) from bytes."""
    return tarfile.open(fileobj=io.BytesIO(gzip.decompress(blob)))


def crate_license(tar):
    """Read the `license` field from the crate's Cargo.toml, or None."""
    for member in tar.getmembers():
        # path is `{name}-{version}/Cargo.toml`
        if member.name.count("/") == 1 and member.name.endswith("/Cargo.toml"):
            data = tomllib.loads(tar.extractfile(member).read().decode("utf-8", "replace"))
            return data.get("package", {}).get("license")
    return None


def iter_rust_files(tar, max_file_bytes):
    """Yield (rel_path, text) for kept `.rs` files inside the crate tarball."""
    for member in tar.getmembers():
        if not (member.isfile() and member.name.endswith(".rs")):
            continue
        if member.size > max_file_bytes:
            continue
        # strip the leading `{name}-{version}/` directory
        rel = member.name.split("/", 1)[1] if "/" in member.name else member.name
        if not rel.startswith(KEEP_PREFIXES):
            continue
        if any(part in rel for part in SKIP_DIRS):
            continue
        raw = tar.extractfile(member).read()
        try:
            yield rel, raw.decode("utf-8")
        except UnicodeDecodeError:
            continue


def build_corpus(args):
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT

    print(f"Listing top {args.top_n} crates by downloads ...")
    top = fetch_top_crate_versions(args.top_n, session)
    print(f"  got {len(top)} crates from the API")

    # Seed first (in order), then the rest of the top list; dedup names.
    ordered = []
    seen_names = set()
    for name in SEED_CRATES + list(top):
        if name in seen_names or is_denied(name):
            continue
        seen_names.add(name)
        ordered.append(name)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)

    seen_hashes = set()
    licenses = Counter()
    total = 0
    n_files = n_dups = n_crates = 0
    skipped_license = 0

    with out_path.open("w", encoding="utf-8") as out:
        for name in ordered:
            if total >= args.max_bytes:
                break
            version = top.get(name)
            try:
                # Seed crates may not be in the top list -> resolve their version.
                if version is None:
                    r = session.get(f"{API}/{name}", timeout=30)
                    r.raise_for_status()
                    info = r.json()["crate"]
                    version = info.get("max_stable_version") or info.get("newest_version")
                    time.sleep(1.1)
                blob = download_crate(name, version, cache_dir, session)
                tar = open_crate_tar(blob)
            except Exception as e:  # network / missing version / bad tar
                print(f"  ! skip {name}: {e}")
                continue

            lic = crate_license(tar)
            if not is_permissive(lic):
                skipped_license += 1
                continue
            licenses[lic] += 1

            used_from_crate = 0
            for rel, text in iter_rust_files(tar, args.max_file_bytes):
                h = hashlib.blake2b(text.encode("utf-8"), digest_size=16).digest()
                if h in seen_hashes:
                    n_dups += 1
                    continue
                seen_hashes.add(h)
                header = f"\n\n// ==== {name}/{rel} ====\n\n"
                out.write(header)
                out.write(text)
                total += len(header.encode("utf-8")) + len(text.encode("utf-8"))
                n_files += 1
                used_from_crate += 1
                if total >= args.max_bytes:
                    break
            if used_from_crate:
                n_crates += 1
                print(f"  + {name} ({lic}): {used_from_crate} files "
                      f"[{total:,} / {args.max_bytes:,} bytes]")

    print("\n=== summary ===")
    print(f"crates used        : {n_crates}")
    print(f"files kept         : {n_files}")
    print(f"duplicate files    : {n_dups}")
    print(f"crates skip (lic.) : {skipped_license}")
    print(f"corpus bytes       : {total:,}")
    print(f"output             : {out_path}")
    print(f"licenses           : {dict(licenses)}")


def parse_args():
    p = argparse.ArgumentParser(description="Build a Rust corpus from crates.io.")
    p.add_argument("--max-bytes", type=int, default=25_000_000,
                   help="total corpus size cap in bytes (default: 25_000_000)")
    p.add_argument("--out", default="data/rust_corpus.txt",
                   help="output corpus path (default: data/rust_corpus.txt)")
    p.add_argument("--top-n", type=int, default=1000,
                   help="how many top-downloaded crates to consider (default: 1000)")
    p.add_argument("--max-file-bytes", type=int, default=64_000,
                   help="skip .rs files larger than this (default: 64_000)")
    p.add_argument("--cache-dir", default="data/.crate_cache",
                   help="where to cache downloaded .crate tarballs")
    return p.parse_args()


if __name__ == "__main__":
    build_corpus(parse_args())
