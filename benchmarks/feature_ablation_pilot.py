#!/usr/bin/env python3
"""Controlled ablation: does adding the three new feature families (letter
frequency-rank vector, Unicode script composition, word/morpheme-length
sequence stats -- see pipeline/stats.py) improve cipher identification?

Scope: ONE language, ALL applicable ciphers. This is deliberately smaller
than a full retrain -- it's a fast way to gauge whether the new features
help BEFORE paying for the multi-hour, all-101-languages rebuild
(training/build_model_dataset.py + training/train_model.py).

Why cipher ID (Stage A) only, not language ID (Stage B): with a single
language there is only one class to predict, so language accuracy isn't
a meaningful thing to test here. Stage A's 21-way cipher classification
is language-agnostic by design, so it's the part a single-language pilot
CAN validate honestly.

Design: same train/test split, same fixed hyperparameters (reused from
models/feature_manifest.json's stage_a_params, so this isolates the
feature set as the only variable) -- trains two HistGradientBoosting
classifiers, one restricted to the current 46 production feature columns,
one given all columns (46 + new). Reports both, plus (for context only,
NOT an apples-to-apples number -- see the report) the current production
model's real accuracy on this language, pulled from
output/benchmark_results.csv if present.

Usage:
    python benchmarks/feature_ablation_pilot.py --language en
"""
import argparse
import json
import os
import sys
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
# build_model_dataset.py does `from run_pipeline import ...` as a bare
# sibling import (it's normally run directly, so its own directory ends up
# on sys.path[0] automatically) -- importing it as training.build_model_dataset
# skips that, so put training/ on sys.path too.
sys.path.insert(0, os.path.join(_REPO_ROOT, "training"))

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score
from sklearn.model_selection import train_test_split

from pipeline.corpus import get_or_build_corpus, MIN_USABLE_CORPUS_CHARS
from pipeline.ciphers import CIPHERS
from pipeline.langcodes import LANGUAGE_NAMES
from build_model_dataset import build_model_row

DEFAULT_WINDOW_SIZES = [100, 500, 1000, 2000, 5000, 10000]
DEFAULT_N_SAMPLES = 50
# Where the previous model generation's feature_manifest.json /
# benchmark_results.csv live after being archived (see HANDOFF.md,
# "data source replaced: OPUS-100 -> Leipzig Corpora Collection") --
# used as the "previous performance" reference when models/ is empty
# mid-rebuild. Overridable via --reference-manifest / --reference-benchmark-csv
# for re-runs against a different prior generation.
DEFAULT_REFERENCE_MANIFEST = "archive/opus100_pipeline/models/feature_manifest.json"
DEFAULT_REFERENCE_BENCHMARK_CSV = "archive/opus100_pipeline/output/benchmark_results.csv"


def generate_rows(lang, window_sizes, n_samples, output_dir, max_corpus_chars):
    lang_name = LANGUAGE_NAMES.get(lang, lang)
    corpus_text, alphabet, _freqs = get_or_build_corpus(lang, [], output_dir, max_corpus_chars)
    if len(corpus_text) < MIN_USABLE_CORPUS_CHARS or len(alphabet) < 2:
        raise SystemExit(f"{lang}: no usable cached corpus")
    from pipeline.tokenize import is_word_segmented
    word_segmented_flag = is_word_segmented(lang, corpus_text)
    feasible_windows = [w for w in window_sizes if len(corpus_text) > w]
    cipher_specs = [None] + [c for c in CIPHERS if c.applicable(len(alphabet))]

    rows = []
    for cipher_spec in cipher_specs:
        for w in feasible_windows:
            for sample_id in range(n_samples):
                rows.append(build_model_row(lang, lang_name, cipher_spec, w, sample_id,
                                             corpus_text, alphabet, word_segmented_flag))
    return pd.DataFrame(rows), lang_name, len(cipher_specs), feasible_windows


def fit_eval(X_train, X_test, y_train, y_test, params):
    clf = HistGradientBoostingClassifier(**params, random_state=0)
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test)
    pred = clf.classes_[np.argmax(proba, axis=1)]
    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, average="macro")
    top5 = top_k_accuracy_score(y_test, proba, k=5, labels=clf.classes_)
    return clf, {"accuracy": acc, "macro_f1": f1, "top5_accuracy": top5}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--language", default="en")
    p.add_argument("--window-sizes", default=",".join(str(w) for w in DEFAULT_WINDOW_SIZES))
    p.add_argument("--n-samples", type=int, default=DEFAULT_N_SAMPLES)
    p.add_argument("--output-dir", default="output")
    p.add_argument("--max-corpus-chars", type=int, default=3_000_000)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--report-json", default=None)
    p.add_argument("--reference-manifest", default=DEFAULT_REFERENCE_MANIFEST,
                    help="feature_manifest.json to source the old 46 feature columns + "
                         "stage_a hyperparameters from")
    p.add_argument("--reference-benchmark-csv", default=DEFAULT_REFERENCE_BENCHMARK_CSV,
                    help="benchmark_results.csv to pull the previous production model's "
                         "real accuracy on this language from, for context")
    args = p.parse_args()
    window_sizes = [int(w) for w in args.window_sizes.split(",") if w]

    reference = json.load(open(args.reference_manifest))
    old_feature_cols = reference["feature_cols"]

    print(f"Generating {args.language} samples across all applicable ciphers ...", file=sys.stderr)
    t0 = time.time()
    df, lang_name, n_ciphers, feasible_windows = generate_rows(
        args.language, window_sizes, args.n_samples, args.output_dir, args.max_corpus_chars)
    print(f"  {len(df)} rows, {n_ciphers} ciphers x {len(feasible_windows)} windows x "
          f"{args.n_samples} samples, {time.time()-t0:.1f}s", file=sys.stderr)

    new_cols = [c for c in df.columns if c not in old_feature_cols
                and c not in ("label_language", "label_cipher", "label_cipher_category")]
    all_feature_cols = old_feature_cols + [c for c in new_cols if c not in old_feature_cols]
    print(f"  {len(old_feature_cols)} old columns, {len(new_cols)} new columns, "
          f"{len(all_feature_cols)} total", file=sys.stderr)

    y = df["label_cipher"]
    X_old = df[old_feature_cols].astype(float)
    X_new = df[all_feature_cols].astype(float)

    idx_train, idx_test = train_test_split(
        df.index, test_size=args.test_size, random_state=0, stratify=y)

    params = reference["stage_a_params"]
    print(f"Training (fixed hyperparameters, same train/test split for both) ...", file=sys.stderr)

    t0 = time.time()
    _, old_metrics = fit_eval(X_old.loc[idx_train], X_old.loc[idx_test],
                               y.loc[idx_train], y.loc[idx_test], params)
    print(f"  old features ({len(old_feature_cols)} cols): acc={old_metrics['accuracy']:.4f} "
          f"macro_f1={old_metrics['macro_f1']:.4f} top5={old_metrics['top5_accuracy']:.4f} "
          f"({time.time()-t0:.1f}s)", file=sys.stderr)

    t0 = time.time()
    _, new_metrics = fit_eval(X_new.loc[idx_train], X_new.loc[idx_test],
                               y.loc[idx_train], y.loc[idx_test], params)
    print(f"  new features ({len(all_feature_cols)} cols): acc={new_metrics['accuracy']:.4f} "
          f"macro_f1={new_metrics['macro_f1']:.4f} top5={new_metrics['top5_accuracy']:.4f} "
          f"({time.time()-t0:.1f}s)", file=sys.stderr)

    # Context only: previous production model's real accuracy on this language,
    # from the archived out-of-sample benchmark (NOT the same train/test split
    # or training scope -- trained on the old contaminated OPUS-100 data
    # across all 101 languages, not just this one).
    production_context = None
    bench_path = args.reference_benchmark_csv
    if os.path.exists(bench_path):
        bench = pd.read_csv(bench_path)
        sub = bench[bench.language == args.language]
        if len(sub):
            production_context = {
                "n": int(len(sub)),
                "cipher_accuracy": round(float(sub.cipher_correct.mean()), 4),
            }

    report = {
        "language": args.language, "language_name": lang_name,
        "n_rows": len(df), "n_ciphers": n_ciphers, "window_sizes": feasible_windows,
        "n_samples_per_cell": args.n_samples,
        "n_old_features": len(old_feature_cols), "n_new_features": len(new_cols),
        "n_total_features": len(all_feature_cols),
        "old_features_result": old_metrics,
        "new_features_result": new_metrics,
        "delta_accuracy": round(new_metrics["accuracy"] - old_metrics["accuracy"], 4),
        "delta_macro_f1": round(new_metrics["macro_f1"] - old_metrics["macro_f1"], 4),
        "production_model_context": production_context,
    }

    print("\n=== Summary ===")
    print(f"Language: {lang_name} ({args.language}), {len(df)} fresh samples, {n_ciphers} ciphers")
    print(f"Old features ({len(old_feature_cols)} cols):        acc={old_metrics['accuracy']*100:.1f}%  macro-F1={old_metrics['macro_f1']*100:.1f}%  top5={old_metrics['top5_accuracy']*100:.1f}%")
    print(f"New features ({len(all_feature_cols)} cols):       acc={new_metrics['accuracy']*100:.1f}%  macro-F1={new_metrics['macro_f1']*100:.1f}%  top5={new_metrics['top5_accuracy']*100:.1f}%")
    print(f"Delta:                          acc={report['delta_accuracy']*100:+.1f}pp  macro-F1={report['delta_macro_f1']*100:+.1f}pp")
    if production_context:
        print(f"\n(context, not apples-to-apples -- different data source, feature set, and language scope) "
              f"PREVIOUS production model (OPUS-100, 101 languages, 46 features) on real {args.language} traffic: "
              f"cipher acc={production_context['cipher_accuracy']*100:.1f}% (n={production_context['n']})")

    if args.report_json:
        with open(args.report_json, "w") as fh:
            json.dump(report, fh, indent=2)
        print(f"\nWrote {args.report_json}", file=sys.stderr)


if __name__ == "__main__":
    main()
