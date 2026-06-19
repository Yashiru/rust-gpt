//! Byte-Pair Encoding tokenizer, Rust core + PyO3 bindings.
//!
//! Behaviour mirrors the from-scratch Python implementation it replaces: GPT-2
//! style regex pre-tokenization, BPE merges learned *inside* each chunk, and the
//! exact same JSON on-disk format, so an existing `tokenizer.json` keeps working
//! and encoding is identical given the same merges (see `tests/test_tokenizer.py`).
//!
//! Training uses an incremental pair-count index (only the words touched by a
//! merge are rescanned) and batch encoding is parallelized with rayon.

use std::collections::{HashMap, HashSet};
use std::sync::LazyLock;

use fancy_regex::Regex;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// GPT-2 / GPT-4 style pre-tokenization pattern (identical to the Python one).
const PATTERN: &str =
    r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";

static RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(PATTERN).expect("valid pattern"));

type Pair = (u32, u32);

/// Split text into chunks exactly as the regex `findall` would.
fn pretokenize(text: &str) -> Vec<&str> {
    RE.find_iter(text)
        .filter_map(|m| m.ok())
        .map(|m| m.as_str())
        .collect()
}

/// Replace every non-overlapping occurrence of `pair` with `new_id` (left to right).
fn merge_seq(ids: &[u32], pair: Pair, new_id: u32) -> Vec<u32> {
    let mut out = Vec::with_capacity(ids.len());
    let mut i = 0;
    while i < ids.len() {
        if i + 1 < ids.len() && ids[i] == pair.0 && ids[i + 1] == pair.1 {
            out.push(new_id);
            i += 2;
        } else {
            out.push(ids[i]);
            i += 1;
        }
    }
    out
}

/// A trained tokenizer: ordered merges (for save), rank lookup (for encode),
/// and id -> bytes (for decode).
#[derive(Clone)]
struct Core {
    merges: Vec<(Pair, u32)>,
    ranks: HashMap<Pair, u32>,
    vocab: Vec<Vec<u8>>,
    pattern: String,
}

/// Learn BPE merges on `text` up to `vocab_size` tokens.
fn train_core(text: &str, vocab_size: usize) -> Core {
    // Frequency of each pre-token chunk.
    let mut freqs: HashMap<&str, i64> = HashMap::new();
    for w in pretokenize(text) {
        *freqs.entry(w).or_insert(0) += 1;
    }

    // Each unique chunk becomes a sequence of its UTF-8 byte ids, with a weight.
    let mut words: Vec<Vec<u32>> = Vec::with_capacity(freqs.len());
    let mut wfreq: Vec<i64> = Vec::with_capacity(freqs.len());
    for (w, f) in &freqs {
        words.push(w.bytes().map(|b| b as u32).collect());
        wfreq.push(*f);
    }

    let mut vocab: Vec<Vec<u8>> = (0..256u32).map(|b| vec![b as u8]).collect();
    let mut merges: Vec<(Pair, u32)> = Vec::new();
    let mut ranks: HashMap<Pair, u32> = HashMap::new();

    // Global adjacent-pair counts + an index of which words contain each pair,
    // so a merge only has to rescan the affected words.
    let mut counts: HashMap<Pair, i64> = HashMap::new();
    let mut where_: HashMap<Pair, HashSet<usize>> = HashMap::new();
    for (i, w) in words.iter().enumerate() {
        for p in w.windows(2) {
            let pair = (p[0], p[1]);
            *counts.entry(pair).or_insert(0) += wfreq[i];
            where_.entry(pair).or_default().insert(i);
        }
    }

    let num_merges = vocab_size.saturating_sub(256);
    for k in 0..num_merges {
        if counts.is_empty() {
            break;
        }
        // Most frequent pair; ties broken by the smallest pair (deterministic).
        let mut best_pair = (0u32, 0u32);
        let mut best_count = i64::MIN;
        for (&pair, &c) in counts.iter() {
            if c > best_count || (c == best_count && pair < best_pair) {
                best_count = c;
                best_pair = pair;
            }
        }
        if best_count <= 0 {
            break;
        }

        let new_id = (256 + k) as u32;
        merges.push((best_pair, new_id));
        ranks.insert(best_pair, new_id);
        let mut nv = vocab[best_pair.0 as usize].clone();
        nv.extend_from_slice(&vocab[best_pair.1 as usize]);
        vocab.push(nv);

        // Snapshot the affected words, then re-merge each and patch the deltas.
        let affected: Vec<usize> = where_
            .get(&best_pair)
            .map(|s| s.iter().copied().collect())
            .unwrap_or_default();
        for i in affected {
            let f = wfreq[i];
            // Drop this word's old pair contributions...
            for p in words[i].windows(2) {
                let pair = (p[0], p[1]);
                if let Some(c) = counts.get_mut(&pair) {
                    *c -= f;
                    if *c <= 0 {
                        counts.remove(&pair);
                    }
                }
                if let Some(s) = where_.get_mut(&pair) {
                    s.remove(&i);
                    if s.is_empty() {
                        where_.remove(&pair);
                    }
                }
            }
            // ...apply the merge...
            words[i] = merge_seq(&words[i], best_pair, new_id);
            // ...and add the new ones.
            for p in words[i].windows(2) {
                let pair = (p[0], p[1]);
                *counts.entry(pair).or_insert(0) += f;
                where_.entry(pair).or_default().insert(i);
            }
        }
        counts.remove(&best_pair);
        where_.remove(&best_pair);
    }

    Core {
        merges,
        ranks,
        vocab,
        pattern: PATTERN.to_string(),
    }
}

/// BPE one chunk: repeatedly merge the present pair with the lowest rank.
fn encode_chunk(bytes: &[u8], ranks: &HashMap<Pair, u32>) -> Vec<u32> {
    let mut ids: Vec<u32> = bytes.iter().map(|&b| b as u32).collect();
    while ids.len() >= 2 {
        let mut best: Option<(u32, Pair)> = None; // (rank, pair)
        for w in ids.windows(2) {
            let pair = (w[0], w[1]);
            if let Some(&r) = ranks.get(&pair) {
                if best.map_or(true, |(br, _)| r < br) {
                    best = Some((r, pair));
                }
            }
        }
        match best {
            Some((r, pair)) => ids = merge_seq(&ids, pair, r), // rank == merged token id
            None => break,
        }
    }
    ids
}

fn encode_text(text: &str, ranks: &HashMap<Pair, u32>) -> Vec<u32> {
    let mut out = Vec::new();
    for chunk in pretokenize(text) {
        out.extend(encode_chunk(chunk.as_bytes(), ranks));
    }
    out
}

fn decode_ids(ids: &[u32], vocab: &[Vec<u8>]) -> String {
    let mut bytes = Vec::new();
    for &id in ids {
        if let Some(tok) = vocab.get(id as usize) {
            bytes.extend_from_slice(tok);
        }
    }
    // Lossy: generated ids can land on incomplete UTF-8 (matches Python's errors="replace").
    String::from_utf8_lossy(&bytes).into_owned()
}

// --------------------------------------------------------------------------- //
// JSON persistence (same schema as the Python `save`/`load`)
// --------------------------------------------------------------------------- //
#[derive(Serialize, Deserialize)]
struct TokJson {
    pattern: String,
    vocab_size: usize,
    merges: Vec<(u32, u32, u32)>,
    vocab: HashMap<String, String>, // str(id) -> hex(bytes)
}

fn save_core(core: &Core, path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let vocab = core
        .vocab
        .iter()
        .enumerate()
        .map(|(i, t)| (i.to_string(), hex::encode(t)))
        .collect();
    let j = TokJson {
        pattern: core.pattern.clone(),
        vocab_size: core.vocab.len(),
        merges: core.merges.iter().map(|&((a, b), i)| (a, b, i)).collect(),
        vocab,
    };
    std::fs::write(path, serde_json::to_string_pretty(&j)?)?;
    Ok(())
}

fn load_core(path: &str) -> Result<Core, Box<dyn std::error::Error>> {
    let j: TokJson = serde_json::from_str(&std::fs::read_to_string(path)?)?;
    let mut ranks = HashMap::new();
    let mut merges = Vec::with_capacity(j.merges.len());
    for (a, b, idx) in &j.merges {
        ranks.insert((*a, *b), *idx);
        merges.push(((*a, *b), *idx));
    }
    let mut vocab = vec![Vec::new(); j.vocab_size];
    for (k, v) in &j.vocab {
        let id: usize = k.parse()?;
        if id < vocab.len() {
            vocab[id] = hex::decode(v)?;
        }
    }
    Ok(Core {
        merges,
        ranks,
        vocab,
        pattern: j.pattern,
    })
}

// --------------------------------------------------------------------------- //
// PyO3 bindings
// --------------------------------------------------------------------------- //
/// A trained byte-pair-encoding tokenizer.
#[pyclass]
struct Tokenizer {
    core: Core,
}

#[pymethods]
impl Tokenizer {
    /// Train a new tokenizer on `text` up to `vocab_size` tokens.
    #[staticmethod]
    fn train(py: Python<'_>, text: String, vocab_size: usize) -> Self {
        let core = py.allow_threads(|| train_core(&text, vocab_size));
        Tokenizer { core }
    }

    /// Encode a string into a list of token ids.
    fn encode(&self, text: &str) -> Vec<u32> {
        encode_text(text, &self.core.ranks)
    }

    /// Encode many strings in parallel (releases the GIL).
    fn encode_batch(&self, py: Python<'_>, texts: Vec<String>) -> Vec<Vec<u32>> {
        py.allow_threads(|| {
            texts
                .par_iter()
                .map(|t| encode_text(t, &self.core.ranks))
                .collect()
        })
    }

    /// Decode a list of token ids back into a string.
    fn decode(&self, ids: Vec<u32>) -> String {
        decode_ids(&ids, &self.core.vocab)
    }

    /// Serialize to a JSON file (same schema as the Python implementation).
    fn save(&self, path: &str) -> PyResult<()> {
        save_core(&self.core, path).map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Load a tokenizer from a JSON file produced by `save` (or the Python one).
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let core = load_core(path).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Tokenizer { core })
    }

    #[getter]
    fn vocab_size(&self) -> usize {
        self.core.vocab.len()
    }

    #[getter]
    fn pattern(&self) -> String {
        self.core.pattern.clone()
    }

    /// The learned merges as (p0, p1, id) triples, in order, used by the parity test.
    fn merges_list(&self) -> Vec<(u32, u32, u32)> {
        self.core.merges.iter().map(|&((a, b), i)| (a, b, i)).collect()
    }

    fn __repr__(&self) -> String {
        format!("Tokenizer(vocab_size={})", self.core.vocab.len())
    }
}

#[pymodule]
fn _bpe(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Tokenizer>()?;
    Ok(())
}
