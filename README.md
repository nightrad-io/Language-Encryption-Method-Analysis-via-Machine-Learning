# Cipher & Language Identification

Given a piece of text — plaintext or enciphered with a classical cipher —
predicts which of 21 classical ciphers was used and which of 91 languages
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
- **Stage B** predicts the language (91-way) from those same 105 features
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

Each language's 3M-character corpus is a seeded uniform random sample of
its cleaned sentences. Leipzig's `*-sentences.txt` files are sorted
alphabetically, and the previous (38-language) model's corpora were a file
prefix: English was nothing but sentences starting with digits, "a…",
"after…", "another…", and non-Latin-script languages were front-loaded
with their Latin-script sentences. Latin files are visited in a seeded
shuffle of their sorted paths (dotfiles, code and markdown excluded).

For each (language, cipher, window size) combination, ciphertext samples
are drawn from independently-random offsets into that language's corpus
— not a sliding window — seeded deterministically
(`f"{lang}|{cipher}|{window_size}|{sample_id}"`) so any specific sample is
exactly reproducible.

### Languages (91)

| Family | Languages |
|---|---|
| Germanic (13) | English, German, Dutch, Afrikaans, Western Frisian, Limburgish, Danish, Swedish, Norwegian, Norwegian Bokmål, Norwegian Nynorsk, Icelandic, Yiddish |
| Romance (10) | French, Spanish, Portuguese, Italian, Catalan, Galician, Romanian, Aragonese, Walloon, Latin |
| Slavic (13) | Russian, Ukrainian, Belarusian, Polish, Czech, Slovak, Bulgarian, Macedonian, Serbian, Croatian, Bosnian, Serbo-Croatian, Slovenian |
| Celtic (3) | Irish, Welsh, Breton |
| Baltic (2) | Lithuanian, Latvian |
| Other Indo-European (3) | Greek, Albanian, Armenian |
| Indo-Aryan (10) | Hindi, Urdu, Bengali, Assamese, Marathi, Nepali, Gujarati, Punjabi, Odia, Sinhala |
| Iranian (4) | Persian, Kurdish, Pashto, Tajik |
| Uralic (4) | Finnish, Estonian, Hungarian, Northern Sami |
| Turkic (8) | Turkish, Azerbaijani, Kazakh, Kyrgyz, Tatar, Turkmen, Uzbek, Uyghur |
| Dravidian (4) | Tamil, Telugu, Kannada, Malayalam |
| Afroasiatic (5) | Arabic, Hebrew, Maltese, Amharic, Hausa |
| Austronesian (3) | Indonesian, Malay, Malagasy |
| Niger-Congo (3) | Igbo, Yoruba, Zulu |
| Other (6) | Georgian, Mongolian, Korean, Vietnamese, Basque, Esperanto |

Excluded: Chinese, Japanese, Thai, Khmer, Burmese and Dzongkha (no
whitespace word segmentation, so every word/morpheme feature is blank),
plus Scottish Gaelic, Occitan, Kinyarwanda and Xhosa (no Leipzig corpus at
the wikipedia/news quality bar) — see `pipeline/langcodes.py`.

Several close-knit clusters are hard to separate statistically (the
Norwegian codes and Danish, the Serbo-Croatian continuum, Malay/Indonesian,
Bengali/Assamese, Nepali/Marathi, Kazakh/Tatar, ...). `client.py` detects
when the top-predicted language has a real rival from one of these
clusters and reports it explicitly rather than a single overconfident
guess — see `family_note` above.

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
- **Samples per (language, cipher, window) cell**: 50 (559,500 rows total)
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
  | max_iter | 350 | 350 |
  | max_leaf_nodes | 185 | 255 |
  | learning_rate | 0.0100 | 0.0185 |
  | min_samples_leaf | 95 | 100 |
  | l2_regularization | 0.00292 | 0.00274 |
  | max_bins | 255 | 255 |

## Model performance

Held-out test split (111,900 rows):

| | accuracy | macro-F1 | top-5 |
|---|---|---|---|
| Stage A (cipher, 21-way) | 59.9% | 60.5% | 99.1% |
| Stage B (language, 91-way) | 75.3% | 75.7% | 90.9% |

Both correct (joint): 45.5%.

Fresh out-of-sample benchmark (`benchmarks/benchmark.py --languages all
--window-sizes 100,500,1000,5000 --n-samples 3`, 22,380 samples, seeds
never used in training): cipher 57.0%, language (cipher unknown) 68.0%,
language (cipher known) 65.0%.

Language accuracy by input length (fresh benchmark): 35.5% at 100
characters, 66.7% at 500, 78.7% at 1,000, 90.8% at 5,000. Held-out split
reaches 92.9% at 10,000.

Against the previous 38-language model, benchmarked on the same corrected
corpora and restricted to its 38 languages (9,492 samples):

| | cipher | language | language (cipher known) |
|---|---|---|---|
| previous model (38-way) | 53.2% | 49.7% | 48.3% |
| this model (91-way) | 55.6% | 56.4% | 53.7% |

The previous model's own published numbers (57.6% / 60.8%) were measured on
the alphabetically-truncated corpora it was trained on.

## Training

Everything is run from the repo root (paths are cwd-relative):

```bash
pip install -r requirements-train.txt
python training/fetch_leipzig_corpora.py                 # -> output/corpora/, output/alphabets/
# Latin: clone The Latin Library text into dictionaries/lat_text_latin_library/;
# its corpus is built on first use by the step below.
python training/build_model_dataset.py --n-samples 50    # -> output/model_dataset.csv (one language per core, --jobs)
python training/train_model.py                           # -> models/*, output/model_eval.json
python benchmarks/benchmark.py --languages all           # fresh-seed end-to-end benchmark
python -m unittest discover tests
```

## Model files

| File | Size | What |
|---|---|---|
| `models/cipher_classifier.joblib` | ~138MB | Stage A |
| `models/language_classifier.joblib` | ~581MB | Stage B |
| `models/feature_manifest.json` | 5KB | exact feature column order, both stages' class lists, and hyperparameters — inference must match this exactly |

### Git LFS

Both `.joblib` files are over GitHub's 100MB per-file limit, so they are
tracked via [Git LFS](https://git-lfs.com/)
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
  unknown-cipher path** (~3 percentage points on language accuracy in the
  latest benchmark). Stage B was trained on Stage A's soft out-of-fold
  predicted probabilities; a hard one-hot vector at inference is a
  distribution it never saw during training.
- **Digraphic ciphers carry the least language signal** (63.8% language
  accuracy on the held-out split vs. 84.9% for plaintext). Most of the
  benchmark's linguistically-unrelated confusions (Irish↔Bokmål,
  English→Welsh, Basque→Indonesian) are digraphic samples.
- **Samples overlap at long windows.** Each corpus is 3M characters, so
  50 random 10,000-character windows per cipher (and the benchmark's fresh
  seeds) share plaintext with training samples, under different keys.
  Long-window accuracy is likely optimistic.
- `no` and `nb` are nearly indistinguishable (the benchmark predicts most
  `nb` plaintext samples as `no`); read either as Norwegian (Bokmål).
