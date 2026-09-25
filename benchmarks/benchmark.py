#!/usr/bin/env python3
"""End-to-end benchmark of the trained cipher/language model.

Unlike models/model_eval.json (which reports accuracy on held-out rows
of the pre-built feature table output/model_dataset.csv), this generates
FRESH raw text -> cipher -> ciphertext samples and runs them through the
real feature-extraction path (client.compute_features, the same function
predict.py uses on real unknown text) rather than reading cached CSV
columns. Results are broken down per-language, which model_eval.json
does not do.

Freshness: samples use sample_id >= FRESH_SAMPLE_ID_START (1000).
Training used sample_id 0..(n_samples-1), default 30, so these seeds
were never in the training or eval split by construction.

Two prediction modes are benchmarked per sample:
  - "predicted": Stage A guesses the cipher, Stage B uses that guess.
    This is the real unknown-cipher scenario (predict.py's default mode).
  - "known_cipher": Stage B runs on a one-hot vector for the TRUE cipher,
    skipping Stage A (predict.py's --cipher mode). This isolates Stage
    B's language-ID accuracy from Stage A's cipher-ID errors.

Performance note: HistGradientBoostingClassifier.predict_proba on this
project's 101-class Stage B model costs ~0.7s for a single row but only
~0.02s/row batched (measured: 40 rows batched in ~1s vs ~30s one at a
time) -- tree traversal overhead doesn't amortize row-by-row. So this
script generates and feature-extracts every sample first, then calls
predict_proba in three total batched calls (Stage A once, Stage B twice)
instead of per-sample. Don't reintroduce a per-sample predict loop
without re-checking this.

Usage:
    python benchmarks/benchmark.py
    python benchmarks/benchmark.py --languages en,fr,de --n-samples 5
    python benchmarks/benchmark.py --languages all --window-sizes 100,1000

Run from the repo root (paths are cwd-relative, same convention as the
rest of this project).
"""
import argparse
import json
import os
import random
import sys
import time
import unicodedata

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from pipeline.corpus import get_or_build_corpus, MIN_USABLE_CORPUS_CHARS
from pipeline.ciphers import CIPHERS
from pipeline.langcodes import LANGUAGE_NAMES, UNSUPPORTED_LANGUAGES
from client import CipherLanguageClient, compute_features

FRESH_SAMPLE_ID_START = 1000

DEFAULT_LANGUAGES = [
    "en", "fr", "de", "es", "pl", "ru", "el", "ar", "he", "hi",
    "bn", "ko", "vi", "tr", "fi", "ka", "lat",
]
DEFAULT_WINDOW_SIZES = [100, 500, 1000, 5000]
DEFAULT_N_SAMPLES = 3


def generate_sample(lang, corpus_text, alphabet, cipher_spec, window_size, sample_id):
    cipher_id = cipher_spec.id if cipher_spec else "plaintext"
    seed = f"{lang}|{cipher_id}|{window_size}|{sample_id}"
    rng = random.Random(seed)
    start = rng.randrange(0, len(corpus_text) - window_size + 1)
    plain_sample = corpus_text[start:start + window_size]
    if cipher_spec is None:
        return plain_sample
    ciphertext, _key_repr = cipher_spec.fn(plain_sample, alphabet, rng, corpus_text=corpus_text)
    return ciphertext


def collect_samples(languages, window_sizes, n_samples, output_dir, max_corpus_chars):
    """Generates every (metadata, text) pair up front. Returns a list of
    metadata dicts and a parallel list of raw texts."""
    cipher_specs = [None] + list(CIPHERS)
    metas, texts = [], []

    for li, lang in enumerate(languages, 1):
        lang_name = LANGUAGE_NAMES.get(lang, lang)
        corpus_text, alphabet, _freqs = get_or_build_corpus(lang, [], output_dir, max_corpus_chars)
        alphabet_size = len(alphabet)
        if len(corpus_text) < MIN_USABLE_CORPUS_CHARS or alphabet_size < 2:
            print(f"  [{li}/{len(languages)}] {lang} ({lang_name}): SKIPPED, no cached corpus", file=sys.stderr)
            continue

        feasible_windows = [w for w in window_sizes if len(corpus_text) > w]
        applicable_ciphers = [c for c in cipher_specs if c is None or c.applicable(alphabet_size)]
        before = len(metas)

        for cipher_spec in applicable_ciphers:
            true_cipher = cipher_spec.id if cipher_spec else "plaintext"
            category = cipher_spec.category if cipher_spec else "plaintext"
            for w in feasible_windows:
                for i in range(n_samples):
                    sample_id = FRESH_SAMPLE_ID_START + i
                    text = generate_sample(lang, corpus_text, alphabet, cipher_spec, w, sample_id)
                    if not text.strip():
                        continue
                    metas.append({
                        "language": lang, "language_name": lang_name,
                        "cipher": true_cipher, "cipher_category": category,
                        "window_size": w, "sample_id": sample_id,
                    })
                    texts.append(text)

        print(f"  [{li}/{len(languages)}] {lang} ({lang_name}): {len(applicable_ciphers)} ciphers x "
              f"{len(feasible_windows)} windows x {n_samples} samples -> {len(metas) - before} samples",
              file=sys.stderr)

    return metas, texts


def top_k_from_proba(proba, classes, k):
    """Vectorized top-k over a (n_rows, n_classes) probability matrix.
    Returns (labels, probs), each shape (n_rows, k)."""
    order = np.argsort(-proba, axis=1)[:, :k]
    labels = np.asarray(classes)[order]
    probs = np.take_along_axis(proba, order, axis=1)
    return labels, probs


def run_benchmark(languages, window_sizes, n_samples, models_dir, output_dir, max_corpus_chars):
    client = CipherLanguageClient(models_dir=models_dir)
    feature_cols = client.manifest["feature_cols"]
    cipher_classes = list(client.stage_a.classes_)
    stack_cols = [f"stageA_proba_{c}" for c in cipher_classes]

    print(f"Generating samples for {len(languages)} languages ...", file=sys.stderr)
    t0 = time.time()
    metas, texts = collect_samples(languages, window_sizes, n_samples, output_dir, max_corpus_chars)
    print(f"Generated {len(metas)} samples in {time.time() - t0:.1f}s", file=sys.stderr)
    if not metas:
        return pd.DataFrame()

    print("Extracting features ...", file=sys.stderr)
    t0 = time.time()
    feature_rows = []
    for text in texts:
        text_norm = unicodedata.normalize("NFC", text).lower()
        row = compute_features(text_norm)
        feature_rows.append([row.get(c, np.nan) for c in feature_cols])
    x = pd.DataFrame(feature_rows, columns=feature_cols).astype(float)
    print(f"Extracted features for {len(x)} samples in {time.time() - t0:.1f}s", file=sys.stderr)

    print("Running batched inference (Stage A once, Stage B twice) ...", file=sys.stderr)
    t0 = time.time()
    cipher_proba = client.stage_a.predict_proba(x)
    cipher_top, cipher_top_probs = top_k_from_proba(cipher_proba, cipher_classes, 3)

    true_ciphers = [m["cipher"] for m in metas]
    onehot = np.zeros((len(metas), len(cipher_classes)), dtype=float)
    cipher_index = {c: i for i, c in enumerate(cipher_classes)}
    for row_i, tc in enumerate(true_ciphers):
        onehot[row_i, cipher_index[tc]] = 1.0

    x_b_predicted = pd.concat([x, pd.DataFrame(cipher_proba, columns=stack_cols)], axis=1)
    x_b_known = pd.concat([x, pd.DataFrame(onehot, columns=stack_cols)], axis=1)

    lang_classes = list(client.stage_b.classes_)
    lang_proba_pred = client.stage_b.predict_proba(x_b_predicted)
    lang_top_pred, lang_top_pred_probs = top_k_from_proba(lang_proba_pred, lang_classes, 3)
    lang_proba_known = client.stage_b.predict_proba(x_b_known)
    lang_top_known, lang_top_known_probs = top_k_from_proba(lang_proba_known, lang_classes, 3)
    print(f"Inference done in {time.time() - t0:.1f}s", file=sys.stderr)

    rows = []
    for i, meta in enumerate(metas):
        rows.append({
            **meta,
            "cipher_pred": cipher_top[i, 0], "cipher_correct": cipher_top[i, 0] == meta["cipher"],
            "cipher_confidence": float(cipher_top_probs[i, 0]),
            "cipher_top3": list(cipher_top[i]),
            "lang_pred": lang_top_pred[i, 0], "lang_correct": lang_top_pred[i, 0] == meta["language"],
            "lang_confidence": float(lang_top_pred_probs[i, 0]),
            "lang_top3": list(lang_top_pred[i]),
            "lang_pred_known_cipher": lang_top_known[i, 0],
            "lang_correct_known_cipher": lang_top_known[i, 0] == meta["language"],
            "lang_confidence_known_cipher": float(lang_top_known_probs[i, 0]),
        })
    return pd.DataFrame(rows)


def summarize(df):
    summary = {
        "n_samples": len(df),
        "cipher_id_accuracy": round(df["cipher_correct"].mean(), 4),
        "language_id_accuracy_predicted_cipher": round(df["lang_correct"].mean(), 4),
        "language_id_accuracy_known_cipher": round(df["lang_correct_known_cipher"].mean(), 4),
        "by_language": {},
        "by_cipher": {},
        "by_window_size": {},
        "top_cipher_confusions": [],
        "top_language_confusions": [],
    }

    for lang, g in df.groupby("language"):
        summary["by_language"][lang] = {
            "n": len(g),
            "cipher_acc": round(g["cipher_correct"].mean(), 4),
            "lang_acc_predicted_cipher": round(g["lang_correct"].mean(), 4),
            "lang_acc_known_cipher": round(g["lang_correct_known_cipher"].mean(), 4),
        }
    for cipher, g in df.groupby("cipher"):
        summary["by_cipher"][cipher] = {
            "n": len(g),
            "cipher_acc": round(g["cipher_correct"].mean(), 4),
            "lang_acc_predicted_cipher": round(g["lang_correct"].mean(), 4),
            "lang_acc_known_cipher": round(g["lang_correct_known_cipher"].mean(), 4),
        }
    for w, g in df.groupby("window_size"):
        summary["by_window_size"][int(w)] = {
            "n": len(g),
            "cipher_acc": round(g["cipher_correct"].mean(), 4),
            "lang_acc_predicted_cipher": round(g["lang_correct"].mean(), 4),
            "lang_acc_known_cipher": round(g["lang_correct_known_cipher"].mean(), 4),
        }

    cipher_confusions = (
        df[~df["cipher_correct"]].groupby(["cipher", "cipher_pred"]).size()
        .sort_values(ascending=False).head(15)
    )
    summary["top_cipher_confusions"] = [
        {"true": t, "predicted": p, "count": int(n)} for (t, p), n in cipher_confusions.items()
    ]
    lang_confusions = (
        df[~df["lang_correct"]].groupby(["language", "lang_pred"]).size()
        .sort_values(ascending=False).head(15)
    )
    summary["top_language_confusions"] = [
        {"true": t, "predicted": p, "count": int(n)} for (t, p), n in lang_confusions.items()
    ]
    return summary


def worst_failures(df, n=20):
    """High-confidence wrong predictions -- the model was sure and wrong,
    which is more actionable than a low-confidence miss."""
    wrong = df[~df["lang_correct"]].copy()
    wrong = wrong.sort_values("lang_confidence", ascending=False).head(n)
    return wrong[[
        "language", "cipher", "window_size", "sample_id",
        "lang_pred", "lang_confidence", "lang_top3",
    ]].to_dict(orient="records")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--languages", default=",".join(DEFAULT_LANGUAGES),
                    help="Comma-separated language codes, or 'all' for every cached language")
    p.add_argument("--window-sizes", default=",".join(str(w) for w in DEFAULT_WINDOW_SIZES))
    p.add_argument("--n-samples", type=int, default=DEFAULT_N_SAMPLES)
    p.add_argument("--models-dir", default="models")
    p.add_argument("--output-dir", default="output")
    p.add_argument("--max-corpus-chars", type=int, default=3_000_000)
    p.add_argument("--results-csv", default="output/benchmark_results.csv")
    p.add_argument("--summary-json", default="output/benchmark_summary.json")
    return p.parse_args()


def main():
    args = parse_args()
    if args.languages == "all":
        languages = sorted(
            f[:-4] for f in os.listdir(os.path.join(args.output_dir, "corpora")) if f.endswith(".txt")
            and f[:-4] not in UNSUPPORTED_LANGUAGES
        )
    else:
        languages = args.languages.split(",")
    window_sizes = [int(w) for w in args.window_sizes.split(",") if w]

    print(f"Benchmarking {len(languages)} languages x up to {len(CIPHERS) + 1} ciphers x "
          f"{len(window_sizes)} window sizes x {args.n_samples} samples ...", file=sys.stderr)

    t_total = time.time()
    df = run_benchmark(languages, window_sizes, args.n_samples, args.models_dir, args.output_dir, args.max_corpus_chars)
    df.to_csv(args.results_csv, index=False)
    print(f"\nWrote {len(df)} rows -> {args.results_csv} (total {time.time() - t_total:.1f}s)", file=sys.stderr)

    summary = summarize(df)
    summary["failures"] = worst_failures(df)
    with open(args.summary_json, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"Wrote summary -> {args.summary_json}", file=sys.stderr)

    print(f"\n=== Summary ===")
    print(f"Cipher ID accuracy (Stage A, unknown cipher): {summary['cipher_id_accuracy']*100:.1f}%")
    print(f"Language ID accuracy (cipher unknown, real scenario): {summary['language_id_accuracy_predicted_cipher']*100:.1f}%")
    print(f"Language ID accuracy (cipher known): {summary['language_id_accuracy_known_cipher']*100:.1f}%")


if __name__ == "__main__":
    main()
