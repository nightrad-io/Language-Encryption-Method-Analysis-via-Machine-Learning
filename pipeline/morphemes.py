"""From-scratch byte-pair-encoding (BPE), used as a morpheme proxy (see
output README: this is NOT true morphological segmentation, just an
unsupervised subword unit -- no per-language morphological analyzers exist
for 99 languages without external models/downloads).

Trained fresh per text sample (not cached per language): statistics must be
derivable from the ciphertext sample alone, matching the eventual ML use
case where the model only ever sees ciphertext, never the plaintext.

Both training and encoding use the standard incremental/ranked BPE
algorithms (not the naive "rescan everything every merge" version) -- this
runs per-sample across hundreds of thousands of samples, so the constant
factor matters.
"""
from collections import Counter, defaultdict

from .tokenize import tokenize_words

EOW = "</w>"
NUM_MERGES = 150
MAX_TRAINING_WORDS = 20000


def train_bpe(text: str, num_merges: int = NUM_MERGES):
    words = tokenize_words(text)
    freqs = Counter(words).most_common(MAX_TRAINING_WORDS)
    vocab = {tuple(list(w) + [EOW]): c for w, c in freqs}

    pair_counts = Counter()
    pair_to_words = defaultdict(set)
    for symbols, count in vocab.items():
        for a, b in zip(symbols, symbols[1:]):
            pair_counts[(a, b)] += count
            pair_to_words[(a, b)].add(symbols)

    merges = []
    for _ in range(num_merges):
        if not pair_counts:
            break
        best, best_count = pair_counts.most_common(1)[0]
        if best_count < 2:
            break
        merges.append(best)
        a, b = best
        merged = a + b

        for symbols in list(pair_to_words[best]):
            count = vocab.pop(symbols, None)
            if count is None:
                continue
            for x, y in zip(symbols, symbols[1:]):
                pair_counts[(x, y)] -= count
                if pair_counts[(x, y)] <= 0:
                    del pair_counts[(x, y)]
                pair_to_words[(x, y)].discard(symbols)

            out = []
            i = 0
            while i < len(symbols):
                if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                    out.append(merged)
                    i += 2
                else:
                    out.append(symbols[i])
                    i += 1
            new_symbols = tuple(out)
            vocab[new_symbols] = vocab.get(new_symbols, 0) + count
            for x, y in zip(new_symbols, new_symbols[1:]):
                pair_counts[(x, y)] += count
                pair_to_words[(x, y)].add(new_symbols)
        pair_to_words.pop(best, None)

    return merges


def apply_bpe(word: str, rank):
    """rank: dict mapping (a, b) -> merge priority (lower = merge first),
    built once per merges list via `merge_rank`."""
    symbols = list(word) + [EOW]
    while len(symbols) > 1:
        pairs = list(zip(symbols, symbols[1:]))
        best_pair, best_rank = None, None
        for p in pairs:
            r = rank.get(p)
            if r is not None and (best_rank is None or r < best_rank):
                best_pair, best_rank = p, r
        if best_pair is None:
            break
        a, b = best_pair
        merged = a + b
        out = []
        i = 0
        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                out.append(merged)
                i += 2
            else:
                out.append(symbols[i])
                i += 1
        symbols = out
    if symbols and symbols[-1].endswith(EOW):
        symbols[-1] = symbols[-1][: -len(EOW)]
        if symbols[-1] == "":
            symbols.pop()
    return symbols


def merge_rank(merges):
    return {pair: i for i, pair in enumerate(merges)}


def segment_text(text: str, merges):
    """Segment whitespace-tokenized words in `text` into BPE morpheme-proxy
    units, returned as a flat list in reading order."""
    rank = merge_rank(merges)
    morphemes = []
    for w in tokenize_words(text):
        morphemes.extend(apply_bpe(w, rank))
    return morphemes
