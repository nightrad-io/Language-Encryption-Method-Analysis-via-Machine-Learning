"""Symbol-stream extraction that works from a sample of text ALONE, with no
knowledge of which language or cipher produced it.

This is the ONE place this logic lives, imported by both
build_model_dataset.py (to build the training table) and predict.py (to
score new text) -- if they diverged, that would silently reintroduce
train/serve skew of exactly the kind this module exists to prevent. See
build_model_dataset.py's docstring for the concrete bug this fixed: filtering
letters against a language's precomputed alphabet (which itself drops rare
letters below a frequency-coverage threshold, e.g. English's j/z) leaks
oracle language knowledge into "observable" features and doesn't match what
predict.py can do on genuinely unknown text.
"""
import unicodedata

from .ciphers import PUA_START, PUA_CAPACITY


def extract_observable_symbols(text):
    """Pick the "cipher symbol" stream from raw text using only what's
    actually in the text:

      * mostly Private-Use-Area characters -> homophonic-style output;
        symbols ARE the PUA characters.
      * otherwise, any Unicode letters present -> plaintext or any
        letter-preserving cipher (substitution family, or a letters-only
        digraphic/transposition cipher, which reuses the same alphabet).
      * no letters and no PUA at all (e.g. an ADFGVX-style digit stream)
        -> fall back to every non-whitespace character.
    """
    unicode_letters = [c for c in text if unicodedata.category(c).startswith("L")]
    pua = [c for c in text if PUA_START <= ord(c) < PUA_START + PUA_CAPACITY]
    if pua and len(pua) >= len(unicode_letters):
        return pua
    if unicode_letters:
        return unicode_letters
    return [c for c in text if not c.isspace()]
