#!/usr/bin/env python3
"""Train the two-stage language/cipher identification model on
output/model_dataset.csv (built by build_model_dataset.py -- the
leakage-free feature table; see that file's docstring for why it's
separate from run_pipeline.py's CSVs).

Stage A: HistGradientBoostingClassifier predicts the cipher (21-way:
20 ciphers + plaintext) from the observable feature columns. Chosen over
plain RandomForest because it handles missing values (NaN) natively --
word/morpheme columns are legitimately absent for letters-only ciphers,
and that absence is itself informative, not something to impute away.

Stage B: a second HistGradientBoostingClassifier predicts the language
(101-way) from the same feature columns PLUS Stage A's predicted cipher
probabilities, stacked in as extra features. The stacking features for
the training rows are out-of-fold predictions (cross_val_predict on the
already-fit-on-train Stage A model) -- never Stage A's in-sample
predictions, which would leak.

Hyperparameter search: BayesSearchCV (scikit-optimize) over both stages.
To keep a "thorough" search (~100+ configurations x 3-fold CV = 300+ fits
per stage) tractable in one run, the SEARCH runs on a bounded stratified
subsample of the training set (fast per-fit, and relative ranking of
hyperparameter configs is stable across subsample size); the FINAL model
for each stage is then refit once on the FULL training set using the
best hyperparameters found. This is a standard, transparent trick to
make thorough search affordable, not a shortcut that changes what gets
reported: final accuracy numbers are always from the full-data fit,
evaluated on a held-out test set the search never touched.
"""
import argparse
import json
import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_predict, StratifiedKFold
from skopt import BayesSearchCV
from skopt.space import Integer, Real

SEARCH_SPACE = {
    # max_iter capped at 350 (not e.g. 600): a high-max_iter config found during search
    # on the subsample makes the FINAL full-training-set refit expensive in direct
    # proportion (observed: one max_iter=577 config's full-data refit alone ran 15-20+
    # minutes). early_stopping=True already lets a config use fewer boosting rounds
    # when the data doesn't need more, so this bounds worst-case refit cost without
    # meaningfully constraining model capacity.
    "max_iter": Integer(50, 350),
    "max_leaf_nodes": Integer(15, 255),
    "learning_rate": Real(0.01, 0.3, prior="log-uniform"),
    "min_samples_leaf": Integer(5, 100),
    "l2_regularization": Real(1e-4, 10.0, prior="log-uniform"),
    "max_bins": Integer(64, 255),
}

NON_FEATURE_COLS = {"label_language", "label_cipher", "label_cipher_category"}

# Evaluate this many candidate configs per Bayesian-optimization step (x 3 folds = fits per
# step), not skopt's default of 1 -- the default leaves most of a multi-core machine idle
# (only cv folds parallelize, not across configs).
N_POINTS = 5


def load_dataset(path):
    df = pd.read_csv(path)
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    X = df[feature_cols].astype(float)
    return df, X, feature_cols


def search_best_params(X_train, y_train, n_iter, search_sample_size, seed):
    if len(X_train) > search_sample_size:
        X_search, _, y_search, _ = train_test_split(
            X_train, y_train, train_size=search_sample_size, stratify=y_train, random_state=seed)
    else:
        X_search, y_search = X_train, y_train

    base = HistGradientBoostingClassifier(random_state=seed, early_stopping=True,
                                           validation_fraction=0.1, n_iter_no_change=15)
    search = BayesSearchCV(
        base, SEARCH_SPACE, n_iter=n_iter, cv=StratifiedKFold(3, shuffle=True, random_state=seed),
        scoring="accuracy", random_state=seed, n_jobs=-1, n_points=N_POINTS, refit=False,
    )
    t0 = time.time()
    n_steps = -(-n_iter // N_POINTS)

    def log_progress(res):
        # One call per N_POINTS batch; res.func_vals are negated scores.
        step = -(-len(res.func_vals) // N_POINTS)
        print(f"    search step {step}/{n_steps}: {len(res.func_vals)} configs, "
              f"best cv accuracy so far {-min(res.func_vals):.4f}, {time.time()-t0:.0f}s elapsed",
              flush=True)

    search.fit(X_search, y_search, callback=log_progress)
    print(f"    search: {n_iter} configs on {len(X_search)} rows in {time.time()-t0:.0f}s, "
          f"best cv accuracy {search.best_score_:.4f}")
    return dict(search.best_params_)


def train_stage(name, X_train, y_train, X_test, y_test, n_iter, search_sample_size, seed):
    print(f"[{name}] searching hyperparameters ...")
    best_params = search_best_params(X_train, y_train, n_iter, search_sample_size, seed)
    print(f"[{name}] best params: {best_params}")

    print(f"[{name}] refitting on full training set ({len(X_train)} rows) ...")
    model = HistGradientBoostingClassifier(random_state=seed, early_stopping=True,
                                            validation_fraction=0.1, n_iter_no_change=15,
                                            **best_params)
    t0 = time.time()
    model.fit(X_train, y_train)
    print(f"[{name}] fit in {time.time()-t0:.0f}s")

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    n_classes = len(model.classes_)
    # k must stay strictly below n_classes -- top-k of ALL classes is always
    # 100% by definition (sklearn warns "meaningless" at k >= n_classes, and
    # errors outright at n_classes == 2 -- see below). Only matters for
    # small-class-count runs (language-scoped smoke tests); the real run's
    # 38 languages keeps k=5 exactly as before.
    k = min(5, max(1, n_classes - 1))
    if n_classes == 2:
        # sklearn's top_k_accuracy_score requires a 1D score array (positive-
        # class probability) for binary classification and rejects a 2D
        # array even with labels passed.
        top_k = top_k_accuracy_score(y_test, y_proba[:, 1], k=k, labels=model.classes_)
    else:
        top_k = top_k_accuracy_score(y_test, y_proba, k=k, labels=model.classes_)
    print(f"[{name}] test accuracy={acc:.4f} macro-F1={f1:.4f} top-{k}={top_k:.4f}")

    return model, best_params, {"test_accuracy": acc, "test_macro_f1": f1, f"test_top{k}_accuracy": top_k}


def oof_proba_frame(model_params, X_train, y_train, seed, prefix):
    """Out-of-fold predicted probabilities for the training rows, using
    the SAME hyperparameters as the final Stage A model but refit inside
    each CV fold -- so Stage B's stacking feature for a training row never
    comes from a model that saw that row's label during Stage A training."""
    base = HistGradientBoostingClassifier(random_state=seed, early_stopping=True,
                                           validation_fraction=0.1, n_iter_no_change=15,
                                           **model_params)
    proba = cross_val_predict(base, X_train, y_train, cv=StratifiedKFold(5, shuffle=True, random_state=seed),
                               method="predict_proba", n_jobs=-1)
    classes = sorted(y_train.unique())
    cols = [f"{prefix}_{c}" for c in classes]
    return pd.DataFrame(proba, columns=cols, index=X_train.index)


def top_confusions(y_true, y_pred, n=15):
    labels = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    pairs = []
    for i, t in enumerate(labels):
        for j, p in enumerate(labels):
            if i != j and cm[i, j] > 0:
                pairs.append((int(cm[i, j]), t, p))
    pairs.sort(reverse=True)
    return [{"true": t, "predicted": p, "count": c} for c, t, p in pairs[:n]]


def breakdown_accuracy(df_test, y_true, y_pred, group_col):
    out = {}
    for g, idx in df_test.groupby(group_col).groups.items():
        mask = df_test.index.isin(idx)
        out[str(g)] = round(accuracy_score(np.array(y_true)[mask], np.array(y_pred)[mask]), 4)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="output/model_dataset.csv")
    p.add_argument("--n-iter", type=int, default=100, help="BayesSearchCV iterations per stage")
    p.add_argument("--search-sample-size", type=int, default=60000)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--models-dir", default="models")
    p.add_argument("--output-dir", default="output")
    args = p.parse_args()

    os.makedirs(args.models_dir, exist_ok=True)

    print("Loading dataset ...")
    df, X, feature_cols = load_dataset(args.dataset)
    print(f"  {len(df)} rows, {len(feature_cols)} features, "
          f"{df['label_cipher'].nunique()} ciphers, {df['label_language'].nunique()} languages")

    strat_key = df["label_language"] + "|" + df["label_cipher"]
    train_idx, test_idx = train_test_split(
        df.index, test_size=args.test_size, stratify=strat_key, random_state=args.seed)
    X_train, X_test = X.loc[train_idx], X.loc[test_idx]
    df_train, df_test = df.loc[train_idx], df.loc[test_idx]

    # ---------------------------------------------------------------- Stage A
    y_cipher_train, y_cipher_test = df_train["label_cipher"], df_test["label_cipher"]
    stage_a_model, stage_a_params, stage_a_metrics = train_stage(
        "Stage A: cipher", X_train, y_cipher_train, X_test, y_cipher_test,
        args.n_iter, args.search_sample_size, args.seed)
    joblib.dump(stage_a_model, os.path.join(args.models_dir, "cipher_classifier.joblib"))

    cipher_confusions = top_confusions(y_cipher_test.tolist(), stage_a_model.predict(X_test).tolist())

    # ------------------------------------------------------- Stacking feature
    print("[Stage B] computing out-of-fold cipher probabilities for stacking ...")
    oof_train = oof_proba_frame(stage_a_params, X_train, y_cipher_train, args.seed, "stageA_proba")
    test_proba = stage_a_model.predict_proba(X_test)
    oof_test = pd.DataFrame(test_proba, columns=[f"stageA_proba_{c}" for c in stage_a_model.classes_],
                             index=X_test.index)
    # oof_train columns come from per-fold class sets, sorted; align to stage_a_model.classes_ order/columns
    oof_train = oof_train.reindex(columns=oof_test.columns, fill_value=0.0)

    X_train_b = pd.concat([X_train, oof_train], axis=1)
    X_test_b = pd.concat([X_test, oof_test], axis=1)

    # ---------------------------------------------------------------- Stage B
    y_lang_train, y_lang_test = df_train["label_language"], df_test["label_language"]
    stage_b_model, stage_b_params, stage_b_metrics = train_stage(
        "Stage B: language", X_train_b, y_lang_train, X_test_b, y_lang_test,
        args.n_iter, args.search_sample_size, args.seed)
    joblib.dump(stage_b_model, os.path.join(args.models_dir, "language_classifier.joblib"))

    lang_pred = stage_b_model.predict(X_test_b)
    cipher_pred = stage_a_model.predict(X_test)
    joint_correct = (lang_pred == y_lang_test.values) & (cipher_pred == y_cipher_test.values)
    joint_accuracy = float(joint_correct.mean())

    lang_by_window = breakdown_accuracy(df_test, y_lang_test.values, lang_pred, "window_size")
    lang_by_cipher_category = breakdown_accuracy(df_test, y_lang_test.values, lang_pred, "label_cipher_category")
    cipher_by_window = breakdown_accuracy(df_test, y_cipher_test.values, stage_a_model.predict(X_test), "window_size")

    manifest = {
        "feature_cols": feature_cols,
        "stage_a_classes": list(stage_a_model.classes_),
        "stage_a_params": stage_a_params,
        "stage_b_classes": list(stage_b_model.classes_),
        "stage_b_params": stage_b_params,
        "n_iter_search": args.n_iter,
        "search_sample_size": args.search_sample_size,
        "test_size": args.test_size,
        "seed": args.seed,
    }
    with open(os.path.join(args.models_dir, "feature_manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)

    results = {
        "n_rows": len(df),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_features_stage_a": X_train.shape[1],
        "n_features_stage_b": X_train_b.shape[1],
        "stage_a_cipher": {**stage_a_metrics, "top_confusions": cipher_confusions,
                            "accuracy_by_window_size": cipher_by_window},
        "stage_b_language": {**stage_b_metrics, "accuracy_by_window_size": lang_by_window,
                              "accuracy_by_cipher_category": lang_by_cipher_category},
        "joint_accuracy": joint_accuracy,
    }
    with open(os.path.join(args.output_dir, "model_eval.json"), "w") as fh:
        json.dump(results, fh, indent=2)

    print("\n=== Summary ===")
    print(f"Stage A (cipher, {len(stage_a_model.classes_)}-way): "
          f"acc={stage_a_metrics['test_accuracy']:.4f}")
    print(f"Stage B (language, {len(stage_b_model.classes_)}-way): "
          f"acc={stage_b_metrics['test_accuracy']:.4f}")
    print(f"Joint (both correct): {joint_accuracy:.4f}")
    print(f"Wrote models to {args.models_dir}/, eval report to {args.output_dir}/model_eval.json")


if __name__ == "__main__":
    main()
