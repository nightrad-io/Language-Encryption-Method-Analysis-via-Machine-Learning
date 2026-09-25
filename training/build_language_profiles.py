#!/usr/bin/env python3
"""Build a per-language reference profile from each language's cached
plaintext corpus (output/corpora/<lang>.txt, written by run_pipeline.py):
word-initial and word-final letter frequency tables, a word-length
distribution, and a frequency-ranked common-word list.

These are corpus-level ground truth, not per-sample statistics, so they
live alongside (not inside) the per-sample CSVs. They exist for two
reasons: (1) positional letter frequency is one of the few statistics that
survives homophonic substitution (see output/README.md), and comparing a
sample's own positional stats against these per-language reference tables
is what makes that comparison useful rather than just "vs. English
defaults"; (2) a frequency-ranked word list is the key input a
plausible-plaintext/crossword-style recovery stage would need -- a raw
dictionary is full of archaic/rare entries that real text rarely uses.
"""
import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.tokenize import tokenize_words, is_word_segmented
from pipeline.langcodes import UNSUPPORTED_LANGUAGES

TOP_N_WORDS = 5000


def build_profile(lang, text):
    if not is_word_segmented(lang, text):
        return None
    words = tokenize_words(text)
    if len(words) < 200:
        return None

    initial_counts = Counter(w[0] for w in words if w)
    final_counts = Counter(w[-1] for w in words if w)
    length_counts = Counter(len(w) for w in words)
    word_counts = Counter(words)

    n_initial = sum(initial_counts.values())
    n_final = sum(final_counts.values())
    n_len = sum(length_counts.values())
    lengths = [len(w) for w in words]
    mean_len = sum(lengths) / len(lengths)
    var_len = sum((l - mean_len) ** 2 for l in lengths) / len(lengths)

    return {
        "language": lang,
        "n_words": len(words),
        "n_distinct_words": len(word_counts),
        "word_initial_letter_freq": {c: n / n_initial for c, n in initial_counts.most_common()},
        "word_final_letter_freq": {c: n / n_final for c, n in final_counts.most_common()},
        "word_length_distribution": {str(l): n / n_len for l, n in sorted(length_counts.items())},
        "word_length_mean": mean_len,
        "word_length_std": var_len ** 0.5,
        "top_words": [{"word": w, "count": c} for w, c in word_counts.most_common(TOP_N_WORDS)],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", default="output")
    args = p.parse_args()

    corpora_dir = os.path.join(args.output_dir, "corpora")
    profiles_dir = os.path.join(args.output_dir, "language_profiles")
    os.makedirs(profiles_dir, exist_ok=True)

    langs = sorted(
        f[:-4] for f in os.listdir(corpora_dir)
        if f.endswith(".txt") and f[:-4] not in UNSUPPORTED_LANGUAGES
    )
    n_written, n_skipped = 0, 0
    for lang in langs:
        with open(os.path.join(corpora_dir, f"{lang}.txt"), encoding="utf-8") as fh:
            text = fh.read()
        profile = build_profile(lang, text)
        if profile is None:
            n_skipped += 1
            continue
        with open(os.path.join(profiles_dir, f"{lang}.json"), "w", encoding="utf-8") as fh:
            json.dump(profile, fh, ensure_ascii=False)
        n_written += 1

    print(f"Wrote {n_written} language profiles to {profiles_dir} ({n_skipped} skipped: "
          f"not word-segmented or too little text)")


if __name__ == "__main__":
    main()
