"""Known language code table used to resolve a language from a file
extension or a directory-name token. Deliberately restricted to the codes
actually present in this project's corpora (opus-100's ~99 codes + Latin)
plus a handful of common codes, so token matching stays precise rather than
guessing from a giant generic ISO-639 table.
"""

LANGUAGE_NAMES = {
    "af": "Afrikaans", "am": "Amharic", "an": "Aragonese", "ar": "Arabic",
    "as": "Assamese", "az": "Azerbaijani", "be": "Belarusian", "bg": "Bulgarian",
    "bn": "Bengali", "br": "Breton", "bs": "Bosnian", "ca": "Catalan",
    "cs": "Czech", "cy": "Welsh", "da": "Danish", "de": "German",
    "dz": "Dzongkha", "el": "Greek", "en": "English", "eo": "Esperanto",
    "es": "Spanish", "et": "Estonian", "eu": "Basque", "fa": "Persian",
    "fi": "Finnish", "fr": "French", "fy": "Western Frisian", "ga": "Irish",
    "gd": "Scottish Gaelic", "gl": "Galician", "gu": "Gujarati", "ha": "Hausa",
    "he": "Hebrew", "hi": "Hindi", "hr": "Croatian", "hu": "Hungarian",
    "hy": "Armenian", "id": "Indonesian", "ig": "Igbo", "is": "Icelandic",
    "it": "Italian", "ja": "Japanese", "ka": "Georgian", "kk": "Kazakh",
    "km": "Khmer", "kn": "Kannada", "ko": "Korean", "ku": "Kurdish",
    "ky": "Kyrgyz", "li": "Limburgish", "lt": "Lithuanian", "lv": "Latvian",
    "mg": "Malagasy", "mk": "Macedonian", "ml": "Malayalam", "mn": "Mongolian",
    "mr": "Marathi", "ms": "Malay", "mt": "Maltese", "my": "Burmese",
    "nb": "Norwegian Bokmal", "ne": "Nepali", "nl": "Dutch", "nn": "Norwegian Nynorsk",
    "no": "Norwegian", "oc": "Occitan", "or": "Odia", "pa": "Punjabi",
    "pl": "Polish", "ps": "Pashto", "pt": "Portuguese", "ro": "Romanian",
    "ru": "Russian", "rw": "Kinyarwanda", "se": "Northern Sami", "sh": "Serbo-Croatian",
    "si": "Sinhala", "sk": "Slovak", "sl": "Slovenian", "sq": "Albanian",
    "sr": "Serbian", "sv": "Swedish", "ta": "Tamil", "te": "Telugu",
    "tg": "Tajik", "th": "Thai", "tk": "Turkmen", "tr": "Turkish",
    "tt": "Tatar", "ug": "Uyghur", "uk": "Ukrainian", "ur": "Urdu",
    "uz": "Uzbek", "vi": "Vietnamese", "wa": "Walloon", "xh": "Xhosa",
    "yi": "Yiddish", "yo": "Yoruba", "zh": "Chinese", "zu": "Zulu",
    "lat": "Latin",
}

# Out of product scope, for two different reasons:
#
# 1. zh/ja/th/km/my/dz use scripts without whitespace word segmentation (see
#    pipeline/tokenize.py's NO_WHITESPACE_SEGMENTATION_LANGS -- same 6 codes,
#    different concern: that one is the linguistic fact that word-level
#    features can't be computed for these scripts; this one is the product
#    decision to drop them entirely rather than support them cipher-symbol-
#    only). Benchmarked at 11-28% language accuracy, by far the worst of the
#    101 languages, and the source of every top-20 highest-confidence wrong
#    prediction in the 2026-08-21 benchmark -- see HANDOFF.md.
# 2. gd/oc/rw/xh have no trustworthy corpus source: the OPUS-100 data this
#    project used through 2026-08-21 turned out to be heavily contaminated
#    (GNOME/KDE software-localization strings, raw URLs, and in some cases
#    outright corrupted crawl-log text -- see HANDOFF.md) and was archived
#    wholesale in favor of the Leipzig Corpora Collection
#    (dictionaries/leipzig_sources.json). gd and oc have zero coverage in
#    Leipzig; rw and xh only have lower-tier community/web genres there
#    (the same contamination risk category just moved away from), so they're
#    held to the same wikipedia/news/newscrawl bar as everything else and
#    dropped rather than let back in through a lower-quality source.
#
# Excluded from corpus/dataset generation in training/run_pipeline.py,
# training/build_model_dataset.py, and benchmarks/benchmark.py.
UNSUPPORTED_LANGUAGES = {"zh", "ja", "th", "km", "my", "dz", "gd", "oc", "rw", "xh"}

# Tokens that collide with real language codes but are common English words /
# path noise in this project's corpora, so a directory/filename token match
# on these alone is not trusted as a language label.
AMBIGUOUS_TOKENS = {
    "an", "as", "or", "no", "he", "is", "id", "ga", "am", "ha", "hi", "li",
    "ca", "da", "my", "so", "in", "it", "we", "me",
}

KNOWN_CODES = set(LANGUAGE_NAMES)

# Close-knit language clusters: pairs/small groups so linguistically similar
# that a statistical classifier genuinely can't cleanly separate them --
# used by client.py to flag "this could be any of these, and here's the
# lean" instead of reporting one overconfident guess. Chosen from real
# confusion evidence in the 2026-08-22 38-language (Germanic/Romance/
# Slavic/Hellenic/Basque) rebuild -- no/nb/nn, bs/hr/sr/sh, bg/mk, ru/uk,
# nl/af were all top confusions there, by a wide margin over anything else
# -- plus well-established mutual-intelligibility relationships for
# languages that run didn't include enough members of to surface on their
# own (cs/sk, pt/gl, ms/id). Deliberately NOT linguistic-family-complete
# (e.g. not "all Slavic languages") -- membership means genuinely easy to
# confuse, not merely related.
LANGUAGE_FAMILIES = [
    frozenset({"no", "nb", "nn", "da"}),
    frozenset({"bs", "hr", "sr", "sh"}),
    frozenset({"bg", "mk"}),
    frozenset({"ru", "uk", "be"}),
    frozenset({"cs", "sk"}),
    frozenset({"nl", "af", "fy", "li"}),
    frozenset({"pt", "gl"}),
    frozenset({"ms", "id"}),
]

LANGUAGE_FAMILY_LABELS = {
    frozenset({"no", "nb", "nn", "da"}): "mainland Scandinavian",
    frozenset({"bs", "hr", "sr", "sh"}): "Serbo-Croatian continuum",
    frozenset({"bg", "mk"}): "Bulgarian/Macedonian",
    frozenset({"ru", "uk", "be"}): "East Slavic",
    frozenset({"cs", "sk"}): "Czech/Slovak",
    frozenset({"nl", "af", "fy", "li"}): "Low Franconian/Frisian",
    frozenset({"pt", "gl"}): "Portuguese/Galician",
    frozenset({"ms", "id"}): "Malay/Indonesian",
}
