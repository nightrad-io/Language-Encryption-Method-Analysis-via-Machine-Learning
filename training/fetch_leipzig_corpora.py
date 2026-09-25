#!/usr/bin/env python3
"""Fetch per-language corpora from the Leipzig Corpora Collection
(https://wortschatz.uni-leipzig.de/), replacing the OPUS-100 source this
project used through 2026-08-21 -- see HANDOFF.md for why: OPUS-100 turned
out to be heavily contaminated for a meaningful fraction of languages
(GNOME/KDE software-localization strings mixed into "sentences", raw URLs,
in one case outright corrupted crawl-log text), spread throughout the
corpus rather than concentrated at the start.

Source selection lives in dictionaries/leipzig_sources.json (language code
-> {data_id, url, genre, size, year}), chosen to prefer wikipedia > news >
newscrawl genre and larger sentence counts, generated from the
imvladikon/leipzig_corpora_collection index on Hugging Face. 90 of this
project's 94 non-Latin, word-segmented languages matched at that quality
bar; the other 4 (gd, oc: zero Leipzig coverage; rw, xh: only lower-tier
community/web genres, the same contamination risk category just moved
away from) are in pipeline.langcodes.UNSUPPORTED_LANGUAGES.

Writes directly into the SAME cache locations get_or_build_corpus() reads
(output/corpora/<lang>.txt, output/alphabets/<lang>.json) -- everything
downstream (training/run_pipeline.py, training/build_model_dataset.py,
client.py) is unaware of and unaffected by where a corpus actually came
from. Reuses pipeline.corpus._clean_line for the same metadata/garbage
line-filtering as the OPUS-100 path, applied as defense-in-depth even
though Leipzig's own text should already be far cleaner.

Usage:
    python training/fetch_leipzig_corpora.py
    python training/fetch_leipzig_corpora.py --languages en,fr,de
    python training/fetch_leipzig_corpora.py --force
"""
import argparse
import json
import os
import random
import shutil
import sys
import tarfile
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.corpus import _clean_line, derive_alphabet, MIN_USABLE_CORPUS_CHARS

DEFAULT_MAX_CORPUS_CHARS = 3_000_000
SOURCES_FILE = "dictionaries/leipzig_sources.json"
DOWNLOAD_CACHE_DIR = "dictionaries/leipzig_corpora"


def download(url, dest_path):
    if os.path.exists(dest_path):
        return
    tmp_path = dest_path + ".part"
    req = urllib.request.Request(url, headers={"User-Agent": "curl/8.7.1"})
    with urllib.request.urlopen(req, timeout=120) as resp, open(tmp_path, "wb") as fh:
        shutil.copyfileobj(resp, fh)
    os.rename(tmp_path, dest_path)


def extract_sentences(tar_path, extract_dir):
    with tarfile.open(tar_path, "r:gz") as tf:
        members = [m for m in tf.getmembers() if m.name.endswith("-sentences.txt")]
        if not members:
            raise ValueError(f"no *-sentences.txt member in {tar_path}")
        tf.extract(members[0], extract_dir, filter="data")
        return os.path.join(extract_dir, members[0].name)


def build_corpus_from_sentences(sentences_path, max_chars, seed):
    """Leipzig *-sentences.txt files are sorted alphabetically by sentence
    text, so a prefix up to max_chars is not a sample of the language: for
    English it was nothing but sentences starting with digits, "a...",
    "after...", "another...", which skews every word-initial/word-bigram
    statistic. Instead take a seeded uniform random sample of the cleaned
    sentences (they're already independent, shuffled-from-source units, so
    no discourse continuity is lost) and fill the cap in that order."""
    sentences = []
    with open(sentences_path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            _id, _tab, text = line.partition("\t")
            if not _tab:
                continue
            cleaned = _clean_line(text)
            if cleaned:
                sentences.append(cleaned)
    random.Random(seed).shuffle(sentences)
    parts = []
    total = 0
    for sentence in sentences:
        if total >= max_chars:
            break
        parts.append(sentence)
        total += len(sentence) + 1
    return " ".join(parts)[:max_chars]


def fetch_one(lang, source, output_dir, max_chars, force):
    corpus_path = os.path.join(output_dir, "corpora", f"{lang}.txt")
    alphabet_path = os.path.join(output_dir, "alphabets", f"{lang}.json")
    if not force and os.path.exists(corpus_path) and os.path.exists(alphabet_path):
        return "cached", None

    os.makedirs(DOWNLOAD_CACHE_DIR, exist_ok=True)
    tar_path = os.path.join(DOWNLOAD_CACHE_DIR, f"{lang}.tar.gz")
    download(source["url"], tar_path)

    extract_dir = os.path.join(DOWNLOAD_CACHE_DIR, f"_extract_{lang}")
    try:
        sentences_path = extract_sentences(tar_path, extract_dir)
        text = build_corpus_from_sentences(sentences_path, max_chars, seed=f"{lang}|corpus")
    finally:
        shutil.rmtree(extract_dir, ignore_errors=True)

    if len(text) < MIN_USABLE_CORPUS_CHARS:
        return "insufficient", len(text)

    alphabet, freqs = derive_alphabet(text)
    os.makedirs(os.path.dirname(corpus_path), exist_ok=True)
    os.makedirs(os.path.dirname(alphabet_path), exist_ok=True)
    with open(corpus_path, "w", encoding="utf-8") as fh:
        fh.write(text)
    with open(alphabet_path, "w", encoding="utf-8") as fh:
        json.dump({"alphabet": alphabet, "freqs": freqs}, fh, ensure_ascii=False)
    return "built", len(text)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--languages", default="all", help="comma-separated language codes, or 'all'")
    p.add_argument("--output-dir", default="output")
    p.add_argument("--max-corpus-chars", type=int, default=DEFAULT_MAX_CORPUS_CHARS)
    p.add_argument("--force", action="store_true")
    p.add_argument("--keep-downloads", action="store_true",
                    help="keep the downloaded .tar.gz files (default: kept anyway, this is a no-op alias for clarity)")
    args = p.parse_args()

    sources = json.load(open(SOURCES_FILE))
    languages = sorted(sources) if args.languages == "all" else args.languages.split(",")

    results = {"cached": [], "built": [], "insufficient": [], "error": []}
    for i, lang in enumerate(languages, 1):
        if lang not in sources:
            print(f"  [{i}/{len(languages)}] {lang}: SKIPPED, not in {SOURCES_FILE}", file=sys.stderr)
            continue
        try:
            status, n_chars = fetch_one(lang, sources[lang], args.output_dir, args.max_corpus_chars, args.force)
            results[status].append(lang)
            extra = f", {n_chars} chars" if n_chars is not None else ""
            print(f"  [{i}/{len(languages)}] {lang}: {status}{extra}", file=sys.stderr)
        except Exception as e:
            results["error"].append(lang)
            print(f"  [{i}/{len(languages)}] {lang}: ERROR {e}", file=sys.stderr)

    print(f"\nbuilt={len(results['built'])} cached={len(results['cached'])} "
          f"insufficient={len(results['insufficient'])} error={len(results['error'])}")
    if results["insufficient"]:
        print("insufficient:", results["insufficient"])
    if results["error"]:
        print("error:", results["error"])


if __name__ == "__main__":
    main()
