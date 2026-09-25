#!/usr/bin/env python3
"""Build the training table for the language/cipher identification model:
the same (language, cipher, window, sample) grid as run_pipeline.py, using
the exact same deterministic seeding so it reconstructs identical
ciphertext samples -- but computes statistics the OBSERVABLE-ONLY way
(pipeline/stats.py's `estimate_universe=True` path), never the oracle
`alphabet_size_language`/`letter_universe_size` that run_pipeline.py's CSVs
use. Those two columns (and the three stats derived from them:
`letter_ic_normalized`, `letter_redundancy`, `letter_chi2_uniform`) assume
knowledge of the true language/cipher -- exactly the thing a real "identify
this mystery text" caller doesn't have. See the plan/README for why this
is a separate script rather than a flag on run_pipeline.py: the two
datasets serve different purposes (run_pipeline.py's is the general-purpose
per-cipher statistics reference; this one is specifically the leakage-free
model-training table) and keeping them separate avoids any risk of the
oracle path silently leaking back in.

Output: a single output/model_dataset.csv (not one file per language/cipher
pair) since this is meant to be loaded whole for training.

IMPORTANT (fixed after the first version of this dataset): letters are
extracted via pipeline/observable.py's extract_observable_symbols(), which
looks only at the ciphertext sample itself -- NOT run_pipeline.py's
extract_cipher_symbols(), which filters against a language's precomputed
alphabet. That alphabet is itself derived with a 99.5%-frequency-coverage
cutoff (pipeline/corpus.py::derive_alphabet) that silently drops a
language's rarest real letters (e.g. English's j/z never passed the
threshold, so every "English" training sample had zero j's or z's by
construction). predict.py can't know a language's true alphabet at
inference time and correctly doesn't try to -- so training samples filtered
against that oracle alphabet taught the model a subtly sanitized version of
each language that didn't match what predict.py actually sees on real text.
Using the same sample-only extraction function in both places is what
train/serve parity requires; this was the reused-code seam where that broke
the first time.
"""
import argparse
import csv
import multiprocessing
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.manifest import build_manifest
from pipeline.corpus import group_usable_by_language, get_or_build_corpus, MIN_USABLE_CORPUS_CHARS
from pipeline.tokenize import is_word_segmented
from pipeline.ciphers import CIPHERS
from pipeline.stats import (
    unigram_stats, bigram_stats, repeat_spacing_stats,
    blank_token_level_stats, blank_positional_letter_stats,
    letter_freq_rank_stats, script_composition_stats,
)
from pipeline.langcodes import LANGUAGE_NAMES, UNSUPPORTED_LANGUAGES
from pipeline.observable import extract_observable_symbols
from run_pipeline import (
    DEFAULT_WINDOW_SIZES, DEFAULT_N_SAMPLES, DEFAULT_MAX_CORPUS_CHARS,
    compute_word_and_morpheme_stats,
)


def compute_letter_and_relation_stats_observable(letters):
    """Same shape as run_pipeline.py's compute_letter_and_relation_stats,
    but normalizes IC/redundancy/chi2 against the sample's own Chao1
    estimate instead of an oracle universe size -- see module docstring."""
    out = {}
    out.update(unigram_stats(letters, universe_size=None, prefix="letter", estimate_universe=True))
    out.update(bigram_stats(letters, prefix="letter_bigram"))
    out.update(repeat_spacing_stats(letters, n=3, prefix="letter_trigram_repeat"))
    out.update(letter_freq_rank_stats(letters, k=20, prefix="letter"))
    out.update(script_composition_stats(letters, prefix="letter"))
    return out


def build_model_row(lang, lang_name, cipher_spec, window_size, sample_id, corpus_text,
                     alphabet, word_segmented_flag):
    seed = f"{lang}|{cipher_spec.id if cipher_spec else 'plaintext'}|{window_size}|{sample_id}"
    rng = random.Random(seed)
    start = rng.randrange(0, len(corpus_text) - window_size + 1)
    plain_sample = corpus_text[start:start + window_size]

    if cipher_spec is None:
        ciphertext, category, preserves = plain_sample, "plaintext", True
    else:
        ciphertext, _key_repr = cipher_spec.fn(plain_sample, alphabet, rng, corpus_text=corpus_text)
        category, preserves = cipher_spec.category, cipher_spec.preserves_word_boundaries

    row = {
        "label_language": lang,
        "label_cipher": cipher_spec.id if cipher_spec else "plaintext",
        "label_cipher_category": category,
        "window_size": window_size,
    }
    letters = extract_observable_symbols(ciphertext)
    row.update(compute_letter_and_relation_stats_observable(letters))

    if preserves and word_segmented_flag:
        word_stats, morph_stats, pos_stats, applicable = compute_word_and_morpheme_stats(ciphertext)
    else:
        word_stats, morph_stats, pos_stats, applicable = (
            blank_token_level_stats("word"), blank_token_level_stats("morpheme"),
            blank_positional_letter_stats("letter"), False,
        )
    row["word_level_applicable"] = applicable
    row.update(word_stats)
    row["morpheme_level_applicable"] = applicable
    row.update(morph_stats)
    row.update(pos_stats)
    return row


def build_language_rows(task):
    """Every (cipher, window, sample) row for one language. Top-level so a
    multiprocessing worker can run it; each row's seed depends only on
    (lang, cipher, window, sample_id), so parallel output is identical to
    the serial loop."""
    lang, paths, window_sizes, n_samples, output_dir, max_corpus_chars = task
    lang_name = LANGUAGE_NAMES.get(lang, lang)
    t0 = time.time()
    corpus_text, alphabet, _freqs = get_or_build_corpus(
        lang, paths, output_dir, max_corpus_chars, force=False)
    alphabet_size = len(alphabet)
    if len(corpus_text) < MIN_USABLE_CORPUS_CHARS or alphabet_size < 2:
        return lang, lang_name, None, time.time() - t0

    word_segmented_flag = is_word_segmented(lang, corpus_text)
    feasible_windows = [w for w in window_sizes if len(corpus_text) > w]
    cipher_specs = [None] + [c for c in CIPHERS if c.applicable(alphabet_size)]
    rows = [
        build_model_row(lang, lang_name, cipher_spec, w, sample_id,
                        corpus_text, alphabet, word_segmented_flag)
        for cipher_spec in cipher_specs
        for w in feasible_windows
        for sample_id in range(n_samples)
    ]
    return lang, lang_name, rows, time.time() - t0



def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--languages", default="all")
    p.add_argument("--window-sizes", default=",".join(str(w) for w in DEFAULT_WINDOW_SIZES))
    p.add_argument("--n-samples", type=int, default=DEFAULT_N_SAMPLES)
    p.add_argument("--max-corpus-chars", type=int, default=DEFAULT_MAX_CORPUS_CHARS)
    p.add_argument("--dictionaries-root", default="dictionaries")
    p.add_argument("--output-dir", default="output")
    p.add_argument("--output-file", default=None, help="default: <output-dir>/model_dataset.csv")
    p.add_argument("--force", action="store_true")
    p.add_argument("--jobs", type=int, default=os.cpu_count(),
                   help="parallel worker processes, one language per task (default: all cores)")
    return p.parse_args()


def main():
    args = parse_args()
    window_sizes = sorted(int(w) for w in args.window_sizes.split(",") if w)
    output_file = args.output_file or os.path.join(args.output_dir, "model_dataset.csv")
    manifest_csv = os.path.join(args.output_dir, "corpus_manifest.csv")

    print(f"[1/2] Scanning {args.dictionaries_root} ...", file=sys.stderr)
    entries = build_manifest(args.dictionaries_root, manifest_csv)
    grouped = group_usable_by_language(entries)
    grouped.pop("unknown", None)
    for lang in UNSUPPORTED_LANGUAGES:
        grouped.pop(lang, None)

    # See the matching comment in run_pipeline.py: languages fetched via
    # training/fetch_leipzig_corpora.py have a cache but no raw files under
    # dictionaries/, so they need to be added here explicitly.
    corpora_dir = os.path.join(args.output_dir, "corpora")
    if os.path.isdir(corpora_dir):
        cached_langs = {f[:-4] for f in os.listdir(corpora_dir) if f.endswith(".txt")}
        for lang in cached_langs - set(grouped) - UNSUPPORTED_LANGUAGES:
            grouped[lang] = []

    if args.languages != "all":
        wanted = set(args.languages.split(","))
        grouped = {lang: paths for lang, paths in grouped.items() if lang in wanted}

    tasks = [(lang, paths, window_sizes, args.n_samples, args.output_dir, args.max_corpus_chars)
             for lang, paths in sorted(grouped.items())]
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    tmp_file = output_file + ".partial"
    n_rows = 0
    writer = None
    # imap (ordered) rather than imap_unordered: output row order stays
    # sorted by language, matching the serial build byte-for-byte.
    with open(tmp_file, "w", newline="", encoding="utf-8") as fh, \
            multiprocessing.Pool(max(1, args.jobs)) as pool:
        for i, (lang, lang_name, rows, elapsed) in enumerate(pool.imap(build_language_rows, tasks), 1):
            if rows is None:
                print(f"  [{i}/{len(tasks)}] {lang}: SKIPPED (insufficient corpus/alphabet)", file=sys.stderr)
                continue
            if writer is None:
                writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                writer.writeheader()
            writer.writerows(rows)
            n_rows += len(rows)
            print(f"  [{i}/{len(tasks)}] {lang} ({lang_name}): {len(rows)} rows in {elapsed:.1f}s",
                  file=sys.stderr)

    if not n_rows:
        os.remove(tmp_file)
        print("No rows generated.", file=sys.stderr)
        return
    os.replace(tmp_file, output_file)
    print(f"[2/2] Wrote {n_rows} rows -> {output_file}", file=sys.stderr)


if __name__ == "__main__":
    main()
