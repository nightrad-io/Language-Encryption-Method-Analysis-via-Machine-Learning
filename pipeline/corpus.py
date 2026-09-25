"""Per-language corpus assembly: stream-concatenate usable files for a
language up to a character cap, clean the text, and derive its alphabet.
"""
import json
import os
import random
import re
import unicodedata
from collections import Counter, defaultdict

WHITESPACE_RE = re.compile(r"\s+")
BOILERPLATE_RE = re.compile(
    r"(the latin library|^\s*ipf\s*$|^\s*perseus\b|^\s*intratext\b)",
    re.IGNORECASE,
)
# Web-scrape/metadata leakage found spread throughout several OPUS-100
# corpora, not just at the start (raw URLs, e-commerce/copyright text,
# archive.org crawl artifacts) -- e.g. German: "...copyright © 2009 auto
# chiptuning...", Estonian: crawl-log dumps with literal URLs spliced into
# what's supposed to be sentence-level parallel text. Dropped per-line
# rather than character-scrubbed, since a line built around a URL or a
# copyright notice isn't natural-language prose regardless of what's cut
# out of it.
METADATA_RE = re.compile(
    r"https?\s*:\s*/\s*/|www\s*\.\s*[a-z0-9-]+\s*\.|copyright\b|isbn\b|©|\ball rights reserved\b|"
    r"@[a-z0-9_.+-]+\.[a-z]{2,}",
    re.IGNORECASE,
)
MIN_USABLE_CORPUS_CHARS = 5000
MIN_LINE_LETTER_DENSITY = 0.4


def group_usable_by_language(entries):
    grouped = defaultdict(list)
    for e in entries:
        if e.usable:
            grouped[e.resolved_language].append(e.path)
    for lang in grouped:
        grouped[lang].sort()
    return grouped


def _looks_like_garbage(line: str) -> bool:
    """Catches crawl-log/encoding-corruption fragments that aren't natural
    language at all: a line this sparse in actual letters isn't useful
    prose no matter what else is wrong with it."""
    if not line:
        return False
    letters = sum(1 for c in line if unicodedata.category(c).startswith("L"))
    return letters / len(line) < MIN_LINE_LETTER_DENSITY


def _clean_line(line: str) -> str:
    if BOILERPLATE_RE.search(line):
        return ""
    if METADATA_RE.search(line):
        return ""
    line = unicodedata.normalize("NFC", line)
    line = line.lower()
    line = WHITESPACE_RE.sub(" ", line).strip()
    if _looks_like_garbage(line):
        return ""
    return line


def build_language_corpus(paths, max_chars: int, seed: str) -> str:
    """Stream-read files for one language, cleaning line by line, stopping
    once max_chars characters have been collected. Files are visited in a
    seeded shuffle of their sorted paths: raw os.walk order is filesystem-
    dependent (the same tree built different corpora on APFS vs. Linux) and
    front-loads whatever sorts first -- the Latin Library's 3M-char cap was
    filled by 80 of 2,142 files. Each file's text stays contiguous."""
    paths = sorted(paths)
    random.Random(seed).shuffle(paths)
    parts = []
    total = 0
    for path in paths:
        if total >= max_chars:
            break
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    cleaned = _clean_line(line)
                    if not cleaned:
                        continue
                    parts.append(cleaned)
                    total += len(cleaned) + 1
                    if total >= max_chars:
                        break
        except OSError:
            continue
    text = " ".join(parts)
    return text[:max_chars]


def derive_alphabet(text: str, coverage: float = 0.995):
    counts = Counter(ch for ch in text if unicodedata.category(ch).startswith("L"))
    if not counts:
        return [], {}
    total = sum(counts.values())
    ranked = counts.most_common()
    alphabet = []
    cumulative = 0
    for ch, n in ranked:
        alphabet.append(ch)
        cumulative += n
        if cumulative / total >= coverage:
            break
    freqs = {ch: counts[ch] for ch in alphabet}
    return alphabet, freqs


def get_or_build_corpus(lang: str, paths, cache_dir: str, max_chars: int, force: bool = False):
    corpus_path = os.path.join(cache_dir, "corpora", f"{lang}.txt")
    alphabet_path = os.path.join(cache_dir, "alphabets", f"{lang}.json")
    os.makedirs(os.path.dirname(corpus_path), exist_ok=True)
    os.makedirs(os.path.dirname(alphabet_path), exist_ok=True)

    if not force and os.path.exists(corpus_path) and os.path.exists(alphabet_path):
        with open(corpus_path, "r", encoding="utf-8") as fh:
            text = fh.read()
        with open(alphabet_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return text, data["alphabet"], data["freqs"]

    text = build_language_corpus(paths, max_chars, seed=f"{lang}|corpus")
    alphabet, freqs = derive_alphabet(text)

    with open(corpus_path, "w", encoding="utf-8") as fh:
        fh.write(text)
    with open(alphabet_path, "w", encoding="utf-8") as fh:
        json.dump({"alphabet": alphabet, "freqs": freqs}, fh, ensure_ascii=False)

    return text, alphabet, freqs
