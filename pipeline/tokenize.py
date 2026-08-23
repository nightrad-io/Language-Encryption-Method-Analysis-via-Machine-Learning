"""Word tokenization and a check for whether a language's writing system
is even whitespace-word-segmented (Chinese/Japanese/Thai/Khmer/Burmese are
not, and without a language-specific segmenter we can't fake real word
boundaries for them -- see README caveat)."""
import re

# A "word character" is a Unicode letter, OR a Private-Use-Area codepoint --
# the latter because homophonic_substitution's ciphertext symbols live in
# the PUA range (see pipeline/ciphers.py) and, unlike real letters, aren't
# matched by \w/isalnum(), which would otherwise make every homophonic
# sample look like it has zero words.
_WORD_CHAR = r"[^\W\d_]|[-]"
WORD_RE = re.compile(r"(?:" + _WORD_CHAR + r")+(?:['\-](?:" + _WORD_CHAR + r")+)*", re.UNICODE)

# Scripts conventionally written without inter-word whitespace. Word/morpheme
# level statistics are not computed for these languages (see output README).
NO_WHITESPACE_SEGMENTATION_LANGS = {"zh", "ja", "th", "km", "my", "dz"}


def tokenize_words(text: str):
    return WORD_RE.findall(text)


def is_word_segmented(lang: str, sample_text: str) -> bool:
    if lang in NO_WHITESPACE_SEGMENTATION_LANGS:
        return False
    tokens = tokenize_words(sample_text[:20000])
    if not tokens:
        return False
    avg_len = sum(len(t) for t in tokens) / len(tokens)
    # Whitespace-segmented natural language words average well under this;
    # unsegmented scripts collapse whole clauses into one "token".
    return avg_len < 20
