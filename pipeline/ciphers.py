"""Classical/historical cipher implementations, generalized to operate over
an arbitrary per-language alphabet (not just A-Z) since this project spans
~99 languages and scripts.

Every cipher exposes `encrypt(text, alphabet, rng, corpus_text=None) ->
(ciphertext, key_repr)`. Two families:

  * substitution/polyalphabetic -- non-alphabet characters (spaces,
    punctuation) pass through unchanged, so word boundaries survive.
  * digraphic/grid and transposition -- operate on the letters-only stream
    (spaces stripped), so word boundaries do NOT survive; this is recorded
    in each cipher's `preserves_word_boundaries` metadata and downstream
    code skips word/morpheme-level stats accordingly.
"""
import math
from collections import Counter
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

# Private Use Area: guaranteed not to collide with any real-language
# character, used as a disposable pool of single-character "homophone"
# symbols for homophonic substitution (see homophonic_substitution below).
PUA_START = 0xE000
PUA_CAPACITY = 6400


@dataclass
class CipherSpec:
    id: str
    category: str
    preserves_word_boundaries: bool
    fn: Callable
    min_alphabet_size: int = 2
    max_alphabet_size: Optional[int] = None

    def applicable(self, alphabet_size: int) -> bool:
        if alphabet_size < self.min_alphabet_size:
            return False
        if self.max_alphabet_size and alphabet_size > self.max_alphabet_size:
            return False
        return True


# ---------------------------------------------------------------- helpers --

def _idx_map(alphabet):
    return {c: i for i, c in enumerate(alphabet)}


def _letters_only(text, idx):
    return [c for c in text if c in idx]


def build_square_grid(alphabet, order, grid_size=None):
    """order: the alphabet in the sequence to fill the grid (already keyed/
    shuffled by the caller). Returns (grid 2D list, canonical idx map
    char->(row,col) using each letter's first occurrence)."""
    k = len(alphabet)
    if grid_size is None:
        grid_size = math.ceil(math.sqrt(k))
    flat_size = grid_size * grid_size
    fill = list(order)
    i = 0
    while len(fill) < flat_size:
        fill.append(order[i % len(order)])
        i += 1
    fill = fill[:flat_size]
    grid = [fill[r * grid_size:(r + 1) * grid_size] for r in range(grid_size)]
    idx = {}
    for r in range(grid_size):
        for c in range(grid_size):
            ch = grid[r][c]
            if ch not in idx:
                idx[ch] = (r, c)
    return grid, idx, grid_size


def _pairs_with_filler(letters, filler):
    out = []
    i = 0
    n = len(letters)
    while i < n:
        a = letters[i]
        if i + 1 < n:
            b = letters[i + 1]
            if a == b:
                out.append((a, filler))
                i += 1
                continue
            out.append((a, b))
            i += 2
        else:
            out.append((a, filler))
            i += 1
    return out


def columnar_transpose(symbols: List[str], keylen: int, rng, filler: str) -> Tuple[List[str], List[int]]:
    rank = list(range(keylen))
    rng.shuffle(rank)
    padded = list(symbols)
    while len(padded) % keylen != 0:
        padded.append(filler)
    rows = [padded[i:i + keylen] for i in range(0, len(padded), keylen)]
    order = sorted(range(keylen), key=lambda c: rank[c])
    out = []
    for col in order:
        for row in rows:
            out.append(row[col])
    return out, rank


# --------------------------------------------------------- substitution ---

def caesar(text, alphabet, rng, corpus_text=None):
    k = len(alphabet)
    idx = _idx_map(alphabet)
    shift = rng.randrange(1, k)
    out = [alphabet[(idx[c] + shift) % k] if c in idx else c for c in text]
    return "".join(out), f"shift={shift}"


def atbash(text, alphabet, rng, corpus_text=None):
    k = len(alphabet)
    idx = _idx_map(alphabet)
    out = [alphabet[k - 1 - idx[c]] if c in idx else c for c in text]
    return "".join(out), "atbash"


def mono_substitution(text, alphabet, rng, corpus_text=None):
    idx = _idx_map(alphabet)
    perm = list(alphabet)
    rng.shuffle(perm)
    out = [perm[idx[c]] if c in idx else c for c in text]
    return "".join(out), "".join(perm)


def vigenere(text, alphabet, rng, corpus_text=None):
    k = len(alphabet)
    idx = _idx_map(alphabet)
    keylen = rng.randint(4, 10)
    key = [rng.randrange(k) for _ in range(keylen)]
    out = []
    ki = 0
    for c in text:
        if c in idx:
            out.append(alphabet[(idx[c] + key[ki % keylen]) % k])
            ki += 1
        else:
            out.append(c)
    return "".join(out), "".join(alphabet[x] for x in key)


def beaufort(text, alphabet, rng, corpus_text=None):
    k = len(alphabet)
    idx = _idx_map(alphabet)
    keylen = rng.randint(4, 10)
    key = [rng.randrange(k) for _ in range(keylen)]
    out = []
    ki = 0
    for c in text:
        if c in idx:
            out.append(alphabet[(key[ki % keylen] - idx[c]) % k])
            ki += 1
        else:
            out.append(c)
    return "".join(out), "".join(alphabet[x] for x in key)


def autokey(text, alphabet, rng, corpus_text=None):
    k = len(alphabet)
    idx = _idx_map(alphabet)
    plain_positions = [i for i, c in enumerate(text) if c in idx]
    plain_vals = [idx[text[i]] for i in plain_positions]
    L = min(rng.randint(4, 8), max(len(plain_vals), 1))
    primer = [rng.randrange(k) for _ in range(L)]
    keystream = (primer + plain_vals)[:len(plain_vals)]
    out = list(text)
    for j, pos in enumerate(plain_positions):
        out[pos] = alphabet[(plain_vals[j] + keystream[j]) % k]
    return "".join(out), "".join(alphabet[x] for x in primer)


def running_key(text, alphabet, rng, corpus_text=None):
    k = len(alphabet)
    idx = _idx_map(alphabet)
    plain_positions = [i for i, c in enumerate(text) if c in idx]
    plain_vals = [idx[text[i]] for i in plain_positions]
    n = len(plain_vals)

    key_vals = []
    offset = 0
    if corpus_text and len(corpus_text) > n + 1:
        offset = rng.randrange(0, len(corpus_text) - n)
        key_source = corpus_text[offset:offset + n * 4]
        key_vals = [idx[c] for c in key_source if c in idx][:n]
    while len(key_vals) < n:
        key_vals.append(rng.randrange(k))

    out = list(text)
    for j, pos in enumerate(plain_positions):
        out[pos] = alphabet[(plain_vals[j] + key_vals[j]) % k]
    return "".join(out), f"corpus_offset={offset}"


def porta_variant(text, alphabet, rng, corpus_text=None):
    """Generalized reciprocal-style polyalphabetic cipher in the spirit of
    the historical Porta cipher (shift-then-complement per key symbol).
    Not a reproduction of the exact 13-row Porta table, which is only
    well-defined for a 26-letter alphabet; generalized here for arbitrary
    alphabet sizes."""
    k = len(alphabet)
    idx = _idx_map(alphabet)
    keylen = rng.randint(4, 10)
    key = [rng.randrange(k) for _ in range(keylen)]
    out = []
    ki = 0
    for c in text:
        if c in idx:
            p = idx[c]
            out.append(alphabet[(k - 1 - ((p + key[ki % keylen]) % k)) % k])
            ki += 1
        else:
            out.append(c)
    return "".join(out), "".join(alphabet[x] for x in key)


def homophonic_symbol_budget(alphabet_size: int) -> int:
    """Total distinct homophone symbols homophonic_substitution will emit
    for an alphabet of this size -- depends only on alphabet size, not on
    letter frequencies, so callers (e.g. stats normalization) can compute
    it without re-deriving the frequency-based per-letter allocation."""
    extra_budget = min(300, alphabet_size * 2)
    return min(alphabet_size + extra_budget, PUA_CAPACITY)


def _homophonic_encrypt(text, alphabet, rng, corpus_text, allocation):
    """Shared engine for the homophonic cipher variants below: each
    plaintext letter maps to a POOL of ciphertext symbols ("homophones"),
    redrawn at random per occurrence. `allocation` controls how many
    homophones each letter gets:

      * "proportional" -- proportional to the letter's language frequency
        (the textbook design: flattens the ciphertext distribution toward
        uniform, since common letters get spread across many symbols).
      * "flat" -- every letter gets (approximately) the same number of
        homophones regardless of frequency. This does NOT flatten the
        distribution the same way: a common letter's few homophones each
        still carry high individual frequency, so ciphertext IC stays
        higher than the proportional variant, and a Chao1-style true-
        codespace-size estimate from a small sample behaves differently
        too (see letter_chao1_estimate in output/README.md).

    Homophone symbols are single Private-Use-Area characters, not real
    alphabet letters, so they carry no cross-language leakage and behave
    like any other single-character symbol for downstream statistics.
    """
    k = len(alphabet)
    idx = _idx_map(alphabet)
    extra_budget = min(300, k * 2)
    total_symbols = homophonic_symbol_budget(k)

    if allocation == "proportional":
        freq_source = corpus_text if corpus_text else text
        counts = Counter(c for c in freq_source if c in idx)
        if not counts:
            counts = Counter({c: 1 for c in alphabet})
        total = sum(counts.values())
        raw_extra = {c: (counts.get(c, 0) / total) * extra_budget for c in alphabet}
    else:  # "flat"
        raw_extra = {c: extra_budget / k for c in alphabet}

    floor_extra = {c: int(raw_extra[c]) for c in alphabet}
    remainder = extra_budget - sum(floor_extra.values())
    by_frac = sorted(alphabet, key=lambda c: -(raw_extra[c] - floor_extra[c]))
    for i in range(remainder):
        floor_extra[by_frac[i % len(by_frac)]] += 1
    alloc = {c: 1 + floor_extra[c] for c in alphabet}

    pool = [chr(PUA_START + n) for n in range(total_symbols)]
    rng.shuffle(pool)
    homophones = {}
    pos = 0
    for c in alphabet:
        n = min(alloc[c], total_symbols - pos) or 1
        n = max(n, 1)
        end = min(pos + n, total_symbols)
        homophones[c] = pool[pos:end] or [pool[pos % total_symbols]]
        pos = end

    out = [rng.choice(homophones[c]) if c in idx else c for c in text]
    counts_list = [len(v) for v in homophones.values()]
    return "".join(out), (f"budget={total_symbols};allocation={allocation};"
                           f"min_homophones={min(counts_list)};max_homophones={max(counts_list)}")


def homophonic_substitution(text, alphabet, rng, corpus_text=None):
    """Homophone count per letter proportional to language frequency (the
    textbook design) -- see _homophonic_encrypt for the mechanics and the
    module docstring on the flat variant for how they differ."""
    return _homophonic_encrypt(text, alphabet, rng, corpus_text, "proportional")


def homophonic_flat_substitution(text, alphabet, rng, corpus_text=None):
    """Every letter gets (about) the same number of homophones, regardless
    of frequency -- see _homophonic_encrypt."""
    return _homophonic_encrypt(text, alphabet, rng, corpus_text, "flat")


# -------------------------------------------------------------- digraphic --

def playfair(text, alphabet, rng, corpus_text=None):
    idx = _idx_map(alphabet)
    order = list(alphabet)
    rng.shuffle(order)
    grid, gidx, gsize = build_square_grid(alphabet, order)
    filler = order[0]
    letters = _letters_only(text, idx)
    out = []
    for a, b in _pairs_with_filler(letters, filler):
        ra, ca = gidx[a]
        rb, cb = gidx[b]
        if ra == rb:
            out.append(grid[ra][(ca + 1) % gsize])
            out.append(grid[rb][(cb + 1) % gsize])
        elif ca == cb:
            out.append(grid[(ra + 1) % gsize][ca])
            out.append(grid[(rb + 1) % gsize][cb])
        else:
            out.append(grid[ra][cb])
            out.append(grid[rb][ca])
    return "".join(out), "".join(order)


def two_square(text, alphabet, rng, corpus_text=None):
    idx = _idx_map(alphabet)
    order1 = list(alphabet)
    rng.shuffle(order1)
    order2 = list(alphabet)
    rng.shuffle(order2)
    grid1, gidx1, gsize = build_square_grid(alphabet, order1)
    grid2, gidx2, _ = build_square_grid(alphabet, order2, grid_size=gsize)
    filler = order1[0]
    letters = _letters_only(text, idx)
    out = []
    for a, b in _pairs_with_filler(letters, filler):
        ra, ca = gidx1[a]
        rb, cb = gidx2[b]
        out.append(grid2[ra][cb])
        out.append(grid1[rb][ca])
    return "".join(out), "".join(order1) + "|" + "".join(order2)


def four_square(text, alphabet, rng, corpus_text=None):
    idx = _idx_map(alphabet)
    plain_order = list(alphabet)
    plain_grid, plain_idx, gsize = build_square_grid(alphabet, plain_order)
    order1 = list(alphabet)
    rng.shuffle(order1)
    order2 = list(alphabet)
    rng.shuffle(order2)
    cgrid1, _, _ = build_square_grid(alphabet, order1, grid_size=gsize)
    cgrid2, _, _ = build_square_grid(alphabet, order2, grid_size=gsize)
    filler = order1[0]
    letters = _letters_only(text, idx)
    out = []
    for a, b in _pairs_with_filler(letters, filler):
        ra, ca = plain_idx[a]
        rb, cb = plain_idx[b]
        out.append(cgrid1[ra][cb])
        out.append(cgrid2[rb][ca])
    return "".join(out), "".join(order1) + "|" + "".join(order2)


def bifid(text, alphabet, rng, corpus_text=None, period=5):
    idx = _idx_map(alphabet)
    order = list(alphabet)
    rng.shuffle(order)
    grid, gidx, gsize = build_square_grid(alphabet, order)
    letters = _letters_only(text, idx)
    out = []
    for start in range(0, len(letters), period):
        chunk = letters[start:start + period]
        coords = [gidx[c] for c in chunk]
        rows = [r for r, _ in coords]
        cols = [c for _, c in coords]
        combined = rows + cols
        for i in range(len(chunk)):
            r, c = combined[2 * i], combined[2 * i + 1]
            out.append(grid[r][c])
    return "".join(out), "".join(order)


def trifid(text, alphabet, rng, corpus_text=None, period=5):
    idx = _idx_map(alphabet)
    k = len(alphabet)
    order = list(alphabet)
    rng.shuffle(order)
    cube = math.ceil(k ** (1 / 3))
    if cube < 2:
        cube = 2
    flat_size = cube ** 3
    fill = list(order)
    i = 0
    while len(fill) < flat_size:
        fill.append(order[i % len(order)])
        i += 1
    fill = fill[:flat_size]
    grid = {}
    pos_of = {}
    for n_, ch in enumerate(fill):
        layer = n_ // (cube * cube)
        rem = n_ % (cube * cube)
        row = rem // cube
        col = rem % cube
        grid[(layer, row, col)] = ch
        if ch not in pos_of:
            pos_of[ch] = (layer, row, col)

    letters = _letters_only(text, idx)
    out = []
    for start in range(0, len(letters), period):
        chunk = letters[start:start + period]
        coords = [pos_of[c] for c in chunk]
        layers = [t[0] for t in coords]
        rows = [t[1] for t in coords]
        cols = [t[2] for t in coords]
        combined = layers + rows + cols
        for i in range(len(chunk)):
            triple = (combined[3 * i], combined[3 * i + 1], combined[3 * i + 2])
            out.append(grid[triple])
    return "".join(out), "".join(order)


def hill2(text, alphabet, rng, corpus_text=None):
    k = len(alphabet)
    idx = _idx_map(alphabet)
    m = None
    for _ in range(200):
        cand = [rng.randrange(k) for _ in range(4)]
        det = (cand[0] * cand[3] - cand[1] * cand[2]) % k
        if math.gcd(det, k) == 1:
            m = cand
            break
    if m is None:
        m = [rng.randrange(k) for _ in range(4)]
    letters = _letters_only(text, idx)
    filler = alphabet[0]
    if len(letters) % 2 == 1:
        letters = letters + [filler]
    out = []
    for i in range(0, len(letters), 2):
        p1, p2 = idx[letters[i]], idx[letters[i + 1]]
        c1 = (m[0] * p1 + m[1] * p2) % k
        c2 = (m[2] * p1 + m[3] * p2) % k
        out.append(alphabet[c1])
        out.append(alphabet[c2])
    return "".join(out), ",".join(str(x) for x in m)


def adfgvx_style(text, alphabet, rng, corpus_text=None):
    idx = _idx_map(alphabet)
    order = list(alphabet)
    rng.shuffle(order)
    grid, gidx, gsize = build_square_grid(alphabet, order, grid_size=6)
    letters = _letters_only(text, idx)
    coord_symbols = []
    for c in letters:
        r, col = gidx[c]
        coord_symbols.append(str(r))
        coord_symbols.append(str(col))
    keylen = rng.randint(5, 9)
    transposed, rank = columnar_transpose(coord_symbols, keylen, rng, filler="0")
    return "".join(transposed), f"grid={''.join(order)};rank={rank}"


# ----------------------------------------------------------- transposition --

def rail_fence(text, alphabet, rng, corpus_text=None):
    idx = _idx_map(alphabet)
    letters = _letters_only(text, idx)
    rails = rng.randint(2, 6)
    if len(letters) < 2:
        return "".join(letters), f"rails={rails}"
    fences = [[] for _ in range(rails)]
    rail, direction = 0, 1
    for c in letters:
        fences[rail].append(c)
        if rail == 0:
            direction = 1
        elif rail == rails - 1:
            direction = -1
        rail += direction
    return "".join("".join(f) for f in fences), f"rails={rails}"


def columnar_transposition(text, alphabet, rng, corpus_text=None):
    idx = _idx_map(alphabet)
    letters = _letters_only(text, idx)
    keylen = rng.randint(4, 10)
    filler = alphabet[0]
    out, rank = columnar_transpose(letters, keylen, rng, filler)
    return "".join(out), f"keylen={keylen};rank={rank}"


def route_transposition(text, alphabet, rng, corpus_text=None):
    idx = _idx_map(alphabet)
    letters = _letters_only(text, idx)
    if not letters:
        return "", "cols=0"
    cols = rng.randint(5, 9)
    filler = alphabet[0]
    padded = list(letters)
    while len(padded) % cols != 0:
        padded.append(filler)
    rows = [padded[i:i + cols] for i in range(0, len(padded), cols)]
    out = []
    for c in range(cols):
        col_vals = [row[c] for row in rows]
        if c % 2 == 1:
            col_vals = col_vals[::-1]
        out.extend(col_vals)
    return "".join(out), f"cols={cols}"


CIPHERS: List[CipherSpec] = [
    CipherSpec("caesar", "substitution", True, caesar),
    CipherSpec("atbash", "substitution", True, atbash),
    CipherSpec("mono_substitution", "substitution", True, mono_substitution),
    CipherSpec("homophonic_substitution", "homophonic", True, homophonic_substitution),
    CipherSpec("homophonic_flat_substitution", "homophonic", True, homophonic_flat_substitution),
    CipherSpec("vigenere", "polyalphabetic", True, vigenere),
    CipherSpec("beaufort", "polyalphabetic", True, beaufort),
    CipherSpec("autokey", "polyalphabetic", True, autokey),
    CipherSpec("running_key", "polyalphabetic", True, running_key),
    CipherSpec("porta_variant", "polyalphabetic", True, porta_variant),
    CipherSpec("playfair", "digraphic", False, playfair, min_alphabet_size=4, max_alphabet_size=100),
    CipherSpec("two_square", "digraphic", False, two_square, min_alphabet_size=4, max_alphabet_size=100),
    CipherSpec("four_square", "digraphic", False, four_square, min_alphabet_size=4, max_alphabet_size=100),
    CipherSpec("bifid", "digraphic", False, bifid, min_alphabet_size=4, max_alphabet_size=100),
    CipherSpec("trifid", "digraphic", False, trifid, min_alphabet_size=4, max_alphabet_size=100),
    CipherSpec("hill2", "digraphic", False, hill2, min_alphabet_size=4, max_alphabet_size=100),
    CipherSpec("adfgvx_style", "digraphic", False, adfgvx_style, min_alphabet_size=4, max_alphabet_size=36),
    CipherSpec("rail_fence", "transposition", False, rail_fence, min_alphabet_size=2),
    CipherSpec("columnar_transposition", "transposition", False, columnar_transposition, min_alphabet_size=2),
    CipherSpec("route_transposition", "transposition", False, route_transposition, min_alphabet_size=2),
]

CIPHERS_BY_ID = {c.id: c for c in CIPHERS}
