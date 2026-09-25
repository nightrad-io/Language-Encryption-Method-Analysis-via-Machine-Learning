#!/usr/bin/env python3
"""Quick baseline: can a plain classifier recover the underlying language
from nothing but the letter/word/morpheme statistics computed on a cipher's
OUTPUT (no access to plaintext)? This validates that the dataset generated
by run_pipeline.py actually carries a learnable signal, before anyone
invests in a serious model.

Trains one RandomForest per cipher (features = all numeric stat columns,
target = language), and separately reports accuracy broken down by
window_size for one cipher, to show how sample length affects identifiability
-- the same "variance across window sizes" axis the dataset is built around.
"""
import glob
import json
import os

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, top_k_accuracy_score
from sklearn.model_selection import train_test_split

NON_FEATURE_COLS = {
    "language", "language_name", "cipher", "cipher_category", "sample_id",
    "seed", "corpus_offset", "label_key_repr", "preserves_word_boundaries",
    "word_level_applicable", "morpheme_level_applicable",
}

CIPHERS_TO_TEST = [
    "plaintext", "mono_substitution", "homophonic_substitution", "homophonic_flat_substitution",
    "vigenere", "beaufort", "columnar_transposition", "rail_fence", "playfair", "adfgvx_style",
]


def load_cipher_df(csv_dir, cipher):
    frames = []
    for path in glob.glob(os.path.join(csv_dir, f"*__{cipher}__samples.csv")):
        frames.append(pd.read_csv(path))
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def prep_features(df):
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    X = df[feature_cols].copy()
    # bool-typed applicability/preservation columns can slip in per-cipher; coerce to float
    for c in X.columns:
        if X[c].dtype == bool:
            X[c] = X[c].astype(float)
    all_nan_cols = [c for c in X.columns if X[c].isna().all()]
    X = X.drop(columns=all_nan_cols)
    y = df["language"]
    return X, y, X.columns.tolist()


def run_cipher_experiment(csv_dir, cipher, min_samples_per_class=20):
    df = load_cipher_df(csv_dir, cipher)
    if df is None:
        return {"cipher": cipher, "status": "no data"}

    counts = df["language"].value_counts()
    keep_langs = counts[counts >= min_samples_per_class].index
    df = df[df["language"].isin(keep_langs)]

    X, y, feature_cols = prep_features(df)
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)

    X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
        X_imputed, y, df, test_size=0.3, random_state=42, stratify=y)

    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    top5 = top_k_accuracy_score(y_test, y_proba, k=5, labels=clf.classes_)
    n_classes = y.nunique()

    importances = sorted(zip(feature_cols, clf.feature_importances_),
                          key=lambda t: -t[1])[:10]

    by_window = None
    if "window_size" in df_test.columns:
        by_window = {}
        for w in sorted(df_test["window_size"].unique()):
            mask = (df_test["window_size"] == w).values
            if mask.sum() == 0:
                continue
            by_window[int(w)] = round(accuracy_score(y_test[mask], y_pred[mask]), 4)

    return {
        "cipher": cipher,
        "n_rows": len(df),
        "n_languages": int(n_classes),
        "n_features": len(feature_cols),
        "test_accuracy": round(acc, 4),
        "test_top5_accuracy": round(top5, 4),
        "chance_accuracy": round(1 / n_classes, 4),
        "accuracy_by_window_size": by_window,
        "top_features": [(f, round(v, 4)) for f, v in importances],
    }


def main():
    csv_dir = "output/csv"
    results = []
    for cipher in CIPHERS_TO_TEST:
        print(f"Running {cipher} ...", flush=True)
        res = run_cipher_experiment(csv_dir, cipher)
        results.append(res)
        print(json.dumps(res, indent=2), flush=True)

    with open("output/baseline_results.json", "w") as fh:
        json.dump(results, fh, indent=2)
    print("\nWrote output/baseline_results.json")


if __name__ == "__main__":
    main()
