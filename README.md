# Cipher & Language Identification

Given a piece of text — plaintext or enciphered with a classical cipher —
predicts which of 21 classical ciphers was used and which of 38 languages
the underlying text is in. Two-stage `HistGradientBoostingClassifier`
pipeline, trained entirely on statistics observable from the ciphertext
itself (no oracle knowledge of the true language or cipher at inference
time).

## Quick start

```bash
pip install -r requirements.txt
python predict.py --text "gur dhvpx oebja sbk whzcf bire gur ynml qbt"
python predict.py --file mystery.txt
python predict.py --file mystery.txt --cipher vigenere   # skip cipher ID if already known
python predict.py --list-ciphers
```

Or programmatically:

```python
from client import CipherLanguageClient

client = CipherLanguageClient()          # loads both models once
result = client.predict(text, top_k=3)
result["top_ciphers"]     # [(cipher_id, probability), ...]
result["top_languages"]   # [(lang_code, lang_name, probability), ...]
result.get("family_note") # present when the top language has a close-knit
                           # rival (e.g. Norwegian/Danish, Bosnian/Croatian)
                           # and the result should be read as ambiguous
                           # between them rather than a single confident answer
```

## How it works

- **Stage A** predicts the cipher (21-way) from 105 features computed
  purely from the observable symbol stream of the input text.
- **Stage B** predicts the language (38-way) from those same 105 features
  plus Stage A's predicted probability for each of the 21 ciphers (126
  inputs total) — a stacked architecture, not two independent models.

Feature families (all computed from the ciphertext alone, `pipeline/stats.py`):

| Family | What |
|---|---|
| Letter unigram | count, distinct count, index of coincidence, entropy, Chao1 species estimate, and (Stage A only) IC-normalized/redundancy/chi² against an estimated symbol universe |
| Letter bigram | count, distinct count, IC, entropy |
| Letter trigram repeats | Kasiski-style: repeated n-gram count, mean/GCD spacing between repeats |
| Letter frequency-rank vector | sorted relative frequency of the 20 most common symbols — invariant to any substitution cipher, exactly preserved under transposition |
| Unicode script composition | fraction of letters in each of 26 script buckets (Latin, Cyrillic, Greek, Arabic, Devanagari, CJK, ...) + "other" |
| Word/morpheme-level | unigram/bigram stats, type-token ratio, mean/std length, plus length-sequence IC/entropy and length-bigram stats (answers "given this word's length, how predictable is the next word's length") |
| Positional letter stats | IC/entropy of word-initial and word-final letter distributions — survives ciphers that redraw a fresh symbol per letter occurrence, as long as word boundaries survive |

Word/morpheme-level features are blank (`NaN`, handled natively by
`HistGradientBoostingClassifier`) when a cipher doesn't preserve word
boundaries or there are too few words to compute reliable statistics.

## Dataset

Training text drawn from two sources:

- **[Leipzig Corpora Collection](https://wortschatz.uni-leipzig.de/)** — 90
  languages, predominantly Wikipedia-genre sentence corpora (88 of the 90;
  2 use news/newscrawl genre where Wikipedia coverage was too small),
  10K–300K sentences per language depending on availability, capped at
  3,000,000 characters per language after cleaning.
- **[The Latin Library](https://www.thelatinlibrary.com/)** — classical
  Latin prose, for the 1 remaining language (Latin).

Text was cleaned line by line before use: normalized to NFC, lowercased,
and filtered to drop lines that are boilerplate, contain URLs/copyright
markers, or are mostly non-letter characters (crawl-log/encoding
artifacts). This replaced an earlier version of the dataset built on
OPUS-100, which turned out to have significant contamination for a
meaningful fraction of languages (software-localization strings and
web-crawl artifacts mixed into supposedly natural-language text).

For each (language, cipher, window size) combination, ciphertext samples
are drawn from independently-random offsets into that language's corpus
— not a sliding window — seeded deterministically
(`f"{lang}|{cipher}|{window_size}|{sample_id}"`) so any specific sample is
exactly reproducible.

### Languages (38)

| Family | Languages |
|---|---|
| Germanic (13) | English, German, Dutch, Afrikaans, Western Frisian, Limburgish, Danish, Swedish, Norwegian, Norwegian Bokmål, Norwegian Nynorsk, Icelandic, Yiddish |
| Romance (10) | French, Spanish, Portuguese, Italian, Catalan, Galician, Romanian, Aragonese, Walloon, Latin |
| Slavic (13) | Russian, Ukrainian, Belarusian, Polish, Czech, Slovak, Bulgarian, Macedonian, Serbian, Croatian, Bosnian, Serbo-Croatian, Slovenian |
| Hellenic (1) | Greek |
| Basque (1) | Basque |

This selection concentrates several close-knit language clusters (three
Norwegian codes, four codes across the Serbo-Croatian dialect continuum,
Dutch/Afrikaans). `client.py` detects when the top-predicted language has
a real rival from one of these clusters and reports it explicitly rather
than a single overconfident guess — see `family_note` above.

### Ciphers (21)

| Category | Ciphers |
|---|---|
| Baseline | plaintext |
| Substitution | caesar, atbash, mono_substitution |
| Homophonic substitution | homophonic_substitution, homophonic_flat_substitution |
| Polyalphabetic | vigenere, beaufort, autokey, running_key, porta_variant |
| Digraphic | playfair, two_square, four_square, bifid, trifid, hill2, adfgvx_style |
| Transposition | rail_fence, columnar_transposition, route_transposition |

## Parameters tested

- **Window sizes**: 100, 500, 1,000, 2,000, 5,000, 10,000 characters
- **Samples per (language, cipher, window) cell**: 30
- **Hyperparameter search**: Bayesian optimization (`scikit-optimize`
  `BayesSearchCV`), 100 iterations per stage, 3-fold stratified
  cross-validation, on a 60,000-row stratified subsample, scored on
  accuracy
- **Search space** (both stages): `max_iter` [50, 350], `max_leaf_nodes`
  [15, 255], `learning_rate` [0.01, 0.3] (log-uniform), `min_samples_leaf`
  [5, 100], `l2_regularization` [1e-4, 10.0] (log-uniform), `max_bins`
  [64, 255]
- **Train/test split**: 80/20, stratified, seed 42
- **Final chosen hyperparameters** (after refitting on the full training
  set; also in `models/feature_manifest.json`):

  | | Stage A (cipher) | Stage B (language) |
  |---|---|---|
  | max_iter | 207 | 350 |
  | max_leaf_nodes | 108 | 255 |
  | learning_rate | 0.0164 | 0.0163 |
  | min_samples_leaf | 99 | 100 |
  | l2_regularization | 0.000115 | 0.0407 |
  | max_bins | 255 | 255 |

## Model performance

Held-out test split (28,476 rows):

| | accuracy | macro-F1 | top-5 |
|---|---|---|---|
| Stage A (cipher, 21-way) | 57.6% | 57.9% | 98.8% |
| Stage B (language, 38-way) | 60.8% | 60.7% | 88.0% |

Fresh out-of-sample benchmark (23,730 samples generated with seeds never
used in training): cipher 57.8%, language (cipher unknown) 60.7%,
language (cipher known) 58.6%.

Both metrics climb sharply with input length — language accuracy alone
goes from ~21% at 100 characters to ~80% at 10,000 characters.

## Model files

| File | Size | What |
|---|---|---|
| `models/cipher_classifier.joblib` | ~45MB | Stage A |
| `models/language_classifier.joblib` | ~248MB | Stage B |
| `models/feature_manifest.json` | 4.4KB | exact feature column order, both stages' class lists, and hyperparameters — inference must match this exactly |

### Git LFS

`language_classifier.joblib` is well over GitHub's 100MB per-file limit,
so both `.joblib` files are tracked via [Git LFS](https://git-lfs.com/)
(`.gitattributes` is already configured). Before cloning or committing
model updates:

```bash
brew install git-lfs   # or your platform's equivalent
git lfs install
```

Without Git LFS set up, a `git clone` of this repo will check out small
text pointer files in `models/` instead of the actual model binaries.

## Known limitations

- **Training data caps out at 10,000-character windows.** Longer input is
  genuine extrapolation for a tree-based model — accuracy doesn't degrade
  gracefully past that point.
- **`--cipher` / `known_cipher=` mode measurably underperforms the
  unknown-cipher path** (~2 percentage points on language accuracy in the
  latest benchmark). Stage B was trained on Stage A's soft out-of-fold
  predicted probabilities; a hard one-hot vector at inference is a
  distribution it never saw during training.
- Language coverage is limited to the 38 languages above — no CJK,
  Southeast Asian, South Asian, Semitic, Turkic, or African-language
  coverage in this particular model.
