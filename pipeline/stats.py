"""Statistical measures at letter, letter-relation (bigram), word, word-relation,
morpheme and morpheme-relation levels. All are plain functions over counters
so they apply identically to plaintext and any cipher's output.
"""
import math
import unicodedata
from collections import Counter
from functools import reduce


def _ic(counts: Counter):
    n = sum(counts.values())
    if n < 2:
        return None
    numer = sum(c * (c - 1) for c in counts.values())
    return numer / (n * (n - 1))


def _entropy(counts: Counter):
    n = sum(counts.values())
    if n == 0:
        return None
    ent = 0.0
    for c in counts.values():
        if c == 0:
            continue
        p = c / n
        ent -= p * math.log2(p)
    return ent


def _chao1(counts: Counter):
    """Species-richness estimator: predicts the TRUE number of distinct
    symbol types (e.g. a cipher's real codespace size) from only the
    observed sample, using how many types were seen exactly once (f1) or
    twice (f2). Most useful where the observed distinct-count is known to
    undercount the true alphabet -- e.g. homophonic substitution, whose
    ciphertext symbol space is often only partially sampled at small
    window sizes. Bias-corrected form (avoids the f2=0 singularity)."""
    if not counts:
        return None
    d_obs = len(counts)
    f1 = sum(1 for c in counts.values() if c == 1)
    f2 = sum(1 for c in counts.values() if c == 2)
    return d_obs + (f1 * (f1 - 1)) / (2 * (f2 + 1))


def unigram_stats(symbols, universe_size=None, prefix="letter", estimate_universe=False):
    """symbols: iterable of hashable tokens (chars, words, or morphemes).

    `universe_size`, when given, is treated as authoritative (used by
    run_pipeline.py, which knows the true language alphabet / cipher
    codespace size -- fine for building a labeled statistics dataset, but
    NOT something a real "identify this mystery text" caller could ever
    know). `estimate_universe=True` is for that real scenario: when no
    oracle `universe_size` is passed, normalize against the sample's own
    Chao1 estimate instead, so `*_ic_normalized`/`*_redundancy`/
    `*_chi2_uniform` stay honestly computable from the sample alone.
    """
    counts = Counter(symbols)
    n = sum(counts.values())
    distinct = len(counts)
    ic = _ic(counts)
    ent = _entropy(counts)
    chao1 = _chao1(counts)
    out = {
        f"{prefix}_count": n,
        f"{prefix}_distinct_count": distinct,
        f"{prefix}_ic_raw": ic,
        f"{prefix}_entropy": ent,
        f"{prefix}_chao1_estimate": chao1,
    }
    if universe_size is None and estimate_universe:
        universe_size = chao1 if chao1 else (distinct or None)
    if universe_size and universe_size > 0:
        out[f"{prefix}_ic_normalized"] = ic * universe_size if ic is not None else None
        max_ent = math.log2(universe_size)
        out[f"{prefix}_redundancy"] = (max_ent - ent) / max_ent if ent is not None and max_ent > 0 else None
        expected = n / universe_size if universe_size else None
        if expected and expected > 0:
            observed_over_universe = [counts.get(s, 0) for s in counts]  # observed-only chi2 approx
            chi2 = sum((o - expected) ** 2 / expected for o in observed_over_universe)
            # account for the universe symbols that were never observed (o=0 each)
            missing = max(universe_size - distinct, 0)
            chi2 += missing * expected
            out[f"{prefix}_chi2_uniform"] = chi2
        else:
            out[f"{prefix}_chi2_uniform"] = None
    return out


def letter_freq_rank_stats(symbols, k=20, prefix="letter"):
    """Sorted-descending relative-frequency vector of the k most common
    symbols. Unlike entropy/IC (single scalars summarizing how "peaked"
    the distribution is), this keeps the distribution's actual SHAPE --
    e.g. English's [12.5%, 9.0%, 8.1%, ...] vs German's [15.6%, 9.9%,
    8.3%, ...] are visibly different curves even though their entropy
    values are nearly identical. Invariant to any bijective substitution
    cipher (relabels symbols, never their frequency values) and exactly
    preserved under transposition -- only the symbol identities move.
    Zero-padded when fewer than k distinct symbols occur (a real fact
    about the sample, not a missing value)."""
    counts = Counter(symbols)
    n = sum(counts.values())
    out = {}
    if n == 0:
        for i in range(1, k + 1):
            out[f"{prefix}_freq_rank_{i}"] = None
        return out
    freqs = sorted((c / n for c in counts.values()), reverse=True)[:k]
    freqs += [0.0] * (k - len(freqs))
    for i, f in enumerate(freqs, 1):
        out[f"{prefix}_freq_rank_{i}"] = f
    return out


# Unicode character names start with their script, e.g. "LATIN SMALL LETTER
# A", "CYRILLIC SMALL LETTER A", "CJK UNIFIED IDEOGRAPH-4E00" -- a free
# script tag with no extra dependency. Covers every script among this
# project's 101 supported languages; anything else falls into OTHER.
_SCRIPT_BUCKETS = [
    "LATIN", "CYRILLIC", "GREEK", "ARABIC", "HEBREW", "DEVANAGARI",
    "BENGALI", "GURMUKHI", "GUJARATI", "ORIYA", "TAMIL", "TELUGU",
    "KANNADA", "MALAYALAM", "SINHALA", "THAI", "MYANMAR", "KHMER",
    "TIBETAN", "GEORGIAN", "ARMENIAN", "ETHIOPIC", "CJK", "HANGUL",
    "HIRAGANA", "KATAKANA",
]


def _script_of(ch):
    try:
        name = unicodedata.name(ch)
    except ValueError:
        return "OTHER"
    first = name.split(" ", 1)[0]
    return first if first in _SCRIPT_BUCKETS else "OTHER"


def script_composition_stats(symbols, prefix="letter"):
    """Fraction of observed symbols in each Unicode script bucket. Script
    identity is preserved by plaintext, all substitution/polyalphabetic
    ciphers (they index within the source language's own alphabet -- see
    pipeline/ciphers.py) and transposition (pure reordering); it's the
    single cheapest way to separate e.g. Arabic/Greek/Devanagari/Cyrillic
    samples that land on nearly identical entropy/IC values despite being
    completely different scripts."""
    buckets = _SCRIPT_BUCKETS + ["OTHER"]
    n = len(symbols)
    if n == 0:
        return {f"{prefix}_script_{b.lower()}_frac": None for b in buckets}
    counts = Counter(_script_of(c) for c in symbols)
    return {f"{prefix}_script_{b.lower()}_frac": counts.get(b, 0) / n for b in buckets}


def bigram_stats(symbols_seq, prefix="letter_bigram"):
    seq = list(symbols_seq)
    bigrams = list(zip(seq, seq[1:]))
    counts = Counter(bigrams)
    n = sum(counts.values())
    ic = _ic(counts)
    ent = _entropy(counts)
    return {
        f"{prefix}_count": n,
        f"{prefix}_distinct_count": len(counts),
        f"{prefix}_ic_raw": ic,
        f"{prefix}_entropy": ent,
    }


def repeat_spacing_stats(seq, n=3, prefix="repeat"):
    """Kasiski-style: find repeated n-grams over `seq`, report spacing
    between successive occurrences of the same n-gram (classical
    polyalphabetic key-length clue)."""
    seq = list(seq)
    positions = {}
    distances = []
    for i in range(len(seq) - n + 1):
        gram = tuple(seq[i:i + n])
        if gram in positions:
            distances.append(i - positions[gram][-1])
            positions[gram].append(i)
        else:
            positions[gram] = [i]
    repeated = {g: p for g, p in positions.items() if len(p) > 1}
    if not distances:
        return {
            f"{prefix}_repeated_ngram_count": 0,
            f"{prefix}_mean_spacing": None,
            f"{prefix}_gcd_spacing": None,
        }
    gcd_spacing = reduce(math.gcd, distances)
    return {
        f"{prefix}_repeated_ngram_count": len(repeated),
        f"{prefix}_mean_spacing": sum(distances) / len(distances),
        f"{prefix}_gcd_spacing": gcd_spacing,
    }


def type_token_ratio(tokens):
    n = len(tokens)
    if n == 0:
        return None
    return len(set(tokens)) / n


def length_stats(tokens, prefix):
    """Mean/std of token length, plus IC/entropy of the length distribution
    itself (its shape, not just its first two moments) and IC/entropy of
    the length BIGRAM distribution -- i.e. how predictable a token's length
    is given the previous token's length ("given the current word's size,
    how likely is the next word's size" -- a word-length analog of the
    letter-level Index of Coincidence). Survives every word-boundary-
    preserving cipher, including homophonic substitution: token length
    (character count) doesn't change no matter what each character maps
    to."""
    lens = [len(t) for t in tokens]
    if not lens:
        out = {f"{prefix}_mean_length": None, f"{prefix}_std_length": None,
                f"{prefix}_length_ic": None, f"{prefix}_length_entropy": None}
        out.update({k: None for k in bigram_stats([], prefix=f"{prefix}_length_bigram")})
        return out
    mean = sum(lens) / len(lens)
    var = sum((l - mean) ** 2 for l in lens) / len(lens)
    counts = Counter(lens)
    out = {
        f"{prefix}_mean_length": mean, f"{prefix}_std_length": math.sqrt(var),
        f"{prefix}_length_ic": _ic(counts), f"{prefix}_length_entropy": _entropy(counts),
    }
    out.update(bigram_stats(lens, prefix=f"{prefix}_length_bigram"))
    return out


def positional_letter_stats(words, prefix="letter"):
    """IC/entropy of the word-initial and word-final letter distributions.
    Survives ciphers that redraw a fresh symbol per occurrence (homophonic
    substitution) as long as word boundaries survive, because it's
    positional structure, not raw per-symbol frequency -- see the
    homophonic-substitution notes in output/README.md."""
    initials = Counter(w[0] for w in words if w)
    finals = Counter(w[-1] for w in words if w)
    return {
        f"{prefix}_word_initial_ic_raw": _ic(initials),
        f"{prefix}_word_initial_entropy": _entropy(initials),
        f"{prefix}_word_final_ic_raw": _ic(finals),
        f"{prefix}_word_final_entropy": _entropy(finals),
    }


def blank_positional_letter_stats(prefix="letter"):
    keys = list(positional_letter_stats(["ab", "ba"], prefix).keys())
    return {k: None for k in keys}


def token_level_stats(tokens, prefix):
    """Full stat bundle for a token stream (words or morphemes): unigram
    IC/entropy/distinct-count (the vocabulary-size / "character-set size"
    analog at this level), adjacent-pair (bigram) IC/entropy as the
    token-to-token relational measure, type/token ratio, and length stats."""
    out = {}
    out.update(unigram_stats(tokens, universe_size=None, prefix=prefix))
    out.update(bigram_stats(tokens, prefix=f"{prefix}_bigram"))
    out[f"{prefix}_type_token_ratio"] = type_token_ratio(tokens)
    out.update(length_stats(tokens, prefix))
    return out


def blank_token_level_stats(prefix):
    """Same keys as token_level_stats but all-blank, for samples where a
    level isn't applicable (e.g. word/morpheme stats on a cipher that
    destroys word boundaries)."""
    keys = list(token_level_stats(["a", "b", "a"], prefix).keys())
    return {k: None for k in keys}
