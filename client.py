"""Reusable Python client for querying the trained two-stage cipher/language
identification models (models/cipher_classifier.joblib,
models/language_classifier.joblib).

    from client import CipherLanguageClient
    client = CipherLanguageClient()
    result = client.predict("gur dhvpx oebja sbk...")

For a command-line interface, use predict.py, which is a thin wrapper
around this class.

Feature extraction here imports extract_observable_symbols() from
pipeline/observable.py -- the SAME function training/build_model_dataset.py
uses to build the training table. That sharing is load-bearing, not tidiness:
an earlier version of this file had its own copy of this logic, and
build_model_dataset.py used a different (oracle-alphabet-based) one, which
caused real train/serve skew -- see pipeline/observable.py's docstring for
what that bug was.
"""
import json
import os
import unicodedata

import joblib
import numpy as np
import pandas as pd

from pipeline.observable import extract_observable_symbols
from pipeline.stats import (
    unigram_stats, bigram_stats, repeat_spacing_stats,
    letter_freq_rank_stats, script_composition_stats,
    token_level_stats, blank_token_level_stats,
    positional_letter_stats, blank_positional_letter_stats,
)
from pipeline.tokenize import tokenize_words
from pipeline.morphemes import train_bpe, segment_text
from pipeline.langcodes import LANGUAGE_NAMES, LANGUAGE_FAMILIES, LANGUAGE_FAMILY_LABELS

MIN_WORDS_FOR_TOKEN_STATS = 10

# Below this, a family rival's probability is noise, not real ambiguity --
# no note gets attached at all.
FAMILY_RIVAL_MIN_PROB = 0.05
# The top pick must lead its nearest family rival by at least this ratio to
# report a "pronounced lean" rather than "no clear favorite."
FAMILY_LEAN_RATIO = 1.4


def family_ambiguity_note(lang_classes, lang_proba, top_code, top_name):
    """If the top-predicted language belongs to one of pipeline.langcodes'
    close-knit LANGUAGE_FAMILIES and another member of that same family
    also carries real probability mass, the two are genuinely hard for
    this model to tell apart -- see LANGUAGE_FAMILIES' docstring, this is
    measured confusion from the 38-language rebuild, not guesswork.
    Returns a human-readable note, or None if there's no real family
    rival in this particular prediction."""
    family = next((f for f in LANGUAGE_FAMILIES if top_code in f), None)
    if family is None:
        return None

    proba_by_code = dict(zip(lang_classes, lang_proba))
    rivals = sorted(
        ((code, p) for code, p in proba_by_code.items() if code in family and code != top_code),
        key=lambda cp: -cp[1],
    )
    if not rivals or rivals[0][1] < FAMILY_RIVAL_MIN_PROB:
        return None

    runner_code, runner_prob = rivals[0]
    runner_name = LANGUAGE_NAMES.get(runner_code, runner_code)
    label = LANGUAGE_FAMILY_LABELS.get(family, "/".join(sorted(family)))
    top_prob = proba_by_code[top_code]

    if top_prob < FAMILY_LEAN_RATIO * runner_prob:
        return (f"{top_name} and {runner_name} are both in the {label} family and close here "
                f"({top_prob*100:.0f}% vs {runner_prob*100:.0f}%) -- no clear favorite within the family.")
    return (f"{top_name} is in the {label} family; {runner_name} is the closest rival within it "
            f"({runner_prob*100:.0f}%) -- leaning {top_name}.")


def compute_features(text):
    letters = extract_observable_symbols(text)
    row = {}
    row.update(unigram_stats(letters, universe_size=None, prefix="letter", estimate_universe=True))
    row.update(bigram_stats(letters, prefix="letter_bigram"))
    row.update(repeat_spacing_stats(letters, n=3, prefix="letter_trigram_repeat"))
    row.update(letter_freq_rank_stats(letters, k=20, prefix="letter"))
    row.update(script_composition_stats(letters, prefix="letter"))

    words = tokenize_words(text)
    if len(words) >= MIN_WORDS_FOR_TOKEN_STATS:
        row.update(token_level_stats(words, "word"))
        merges = train_bpe(text)
        morphemes = segment_text(text, merges)
        row.update(token_level_stats(morphemes, "morpheme"))
        row["word_level_applicable"] = True
        row["morpheme_level_applicable"] = True
        row.update(positional_letter_stats(words, "letter"))
    else:
        row.update(blank_token_level_stats("word"))
        row.update(blank_token_level_stats("morpheme"))
        row["word_level_applicable"] = False
        row["morpheme_level_applicable"] = False
        row.update(blank_positional_letter_stats("letter"))

    row["window_size"] = len(text)
    return row


class CipherLanguageClient:
    """Loads the two-stage cipher/language classifier once and exposes
    .predict(text) for repeated queries -- avoids reloading the ~500MB of
    joblib models on every call."""

    def __init__(self, models_dir="models", eval_report="output/model_eval.json"):
        self.models_dir = models_dir
        self.manifest = json.load(open(os.path.join(models_dir, "feature_manifest.json")))
        self.stage_a = joblib.load(os.path.join(models_dir, "cipher_classifier.joblib"))
        self.stage_b = joblib.load(os.path.join(models_dir, "language_classifier.joblib"))
        self.model_eval = json.load(open(eval_report)) if os.path.exists(eval_report) else None

    @property
    def cipher_ids(self):
        """Valid --cipher / known_cipher values, i.e. Stage A's class labels."""
        return list(self.stage_a.classes_)

    def predict(self, text, top_k=3, known_cipher=None):
        """Returns a dict: n_chars, word_level_applicable, cipher_source
        ("predicted" or "user_specified"), top_ciphers (list of
        (cipher_id, probability)), top_languages (list of (lang_code,
        lang_name, probability)), and length_caveat (str, present only if
        an eval report was loaded).

        If known_cipher is given (one of self.cipher_ids), Stage A is
        skipped entirely: Stage B runs on a one-hot cipher vector instead
        of Stage A's predicted probabilities, and top_ciphers reports the
        given cipher at 100% confidence. This is for the case where the
        cipher mechanism is already known and only the language is
        unknown -- it feeds Stage B better information than a predicted
        probability distribution would."""
        text = unicodedata.normalize("NFC", text).lower()
        row = compute_features(text)
        feature_cols = self.manifest["feature_cols"]
        x = pd.DataFrame([[row.get(c, np.nan) for c in feature_cols]], columns=feature_cols).astype(float)

        cipher_classes = list(self.stage_a.classes_)
        if known_cipher is not None:
            if known_cipher not in cipher_classes:
                raise ValueError(f"Unknown cipher {known_cipher!r}; must be one of {cipher_classes}")
            cipher_proba = np.array([1.0 if c == known_cipher else 0.0 for c in cipher_classes])
            top_ciphers = [(known_cipher, 1.0)]
            cipher_source = "user_specified"
        else:
            cipher_proba = self.stage_a.predict_proba(x)[0]
            cipher_order = np.argsort(-cipher_proba)
            top_ciphers = [(cipher_classes[i], float(cipher_proba[i])) for i in cipher_order[:top_k]]
            cipher_source = "predicted"

        stack_cols = [f"stageA_proba_{c}" for c in cipher_classes]
        x_b = pd.concat([x, pd.DataFrame([cipher_proba], columns=stack_cols)], axis=1)

        lang_proba = self.stage_b.predict_proba(x_b)[0]
        lang_order = np.argsort(-lang_proba)
        top_languages = [
            (
                self.stage_b.classes_[i],
                LANGUAGE_NAMES.get(self.stage_b.classes_[i], self.stage_b.classes_[i]),
                float(lang_proba[i]),
            )
            for i in lang_order[:top_k]
        ]

        n_chars = row["window_size"]
        result = {
            "n_chars": n_chars,
            "word_level_applicable": row.get("word_level_applicable", False),
            "cipher_source": cipher_source,
            "top_ciphers": top_ciphers,
            "top_languages": top_languages,
        }
        top_code, top_name, _top_prob = top_languages[0]
        family_note = family_ambiguity_note(self.stage_b.classes_, lang_proba, top_code, top_name)
        if family_note:
            result["family_note"] = family_note
        if self.model_eval is not None:
            result["length_caveat"] = self._length_caveat(n_chars)
        return result

    def _length_caveat(self, n_chars):
        windows = sorted(int(w) for w in self.model_eval["stage_b_language"]["accuracy_by_window_size"])
        closest = min(windows, key=lambda w: abs(w - n_chars))
        acc = self.model_eval["stage_b_language"]["accuracy_by_window_size"][str(closest)]
        return (f"Input is {n_chars} characters. On held-out test samples near that length "
                f"(~{closest} chars), the language model was right {acc*100:.0f}% of the time -- "
                f"treat this prediction with that in mind, not the dataset-wide averages.")
