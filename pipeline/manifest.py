"""Corpus discovery: walk the dictionaries/ tree, decide which files are
usable text, and resolve a language code for each usable file.
"""
import csv
import os
import re
from dataclasses import dataclass, asdict

from .langcodes import KNOWN_CODES, AMBIGUOUS_TOKENS

BINARY_EXTENSIONS = {
    ".zip", ".gz", ".bz2", ".xz", ".7z", ".tar", ".rar",
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".webp",
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".db", ".sqlite", ".sqlite3", ".bin", ".exe", ".dll", ".so", ".dylib",
    ".pyc", ".class", ".o", ".a", ".jar", ".mp3", ".mp4", ".wav", ".avi",
    ".mov", ".ttf", ".otf", ".woff", ".woff2",
}

# Readable text, but not prose in the file's language: repo metadata, code,
# markup. The Latin Library clone's LICENSE.md, README.md and crawler .py
# (plus dotfiles like .gitignore) were being concatenated into the Latin corpus.
NON_PROSE_EXTENSIONS = {".md", ".py", ".sh", ".json", ".csv", ".html", ".htm", ".xml", ".js", ".yml", ".yaml"}

SKIP_DIR_NAMES = {".git", ".hg", ".svn", "__pycache__", "node_modules"}

TOKEN_SPLIT_RE = re.compile(r"[^a-zA-Z0-9]+")


@dataclass
class ManifestEntry:
    path: str
    size_bytes: int
    usable: bool
    reason: str
    resolved_language: str
    resolution_method: str


def _tokens(name: str):
    return [t for t in TOKEN_SPLIT_RE.split(name) if t]


def _sniff_binary(path: str) -> str:
    """Return a non-empty reason string if the file looks binary/unusable.

    Printability is judged on *decoded characters*, not raw bytes: a raw-byte
    check misclassifies any non-Latin UTF-8 text (Arabic, Russian, Chinese,
    ...) as binary, since ~half or more of its bytes are UTF-8 continuation
    bytes (0x80-0xBF) that aren't ASCII-printable on their own.
    """
    try:
        with open(path, "rb") as fh:
            chunk = fh.read(8192)
    except OSError as exc:
        return f"unreadable: {exc}"
    if not chunk:
        return "empty file"
    if b"\x00" in chunk:
        return "contains null bytes"

    text = None
    try:
        text = chunk.decode("utf-8")
    except UnicodeDecodeError:
        # chunk may be truncated mid-codepoint at the 8KB boundary.
        for trim in range(1, 4):
            try:
                text = chunk[:-trim].decode("utf-8")
                break
            except UnicodeDecodeError:
                continue
    if text is None:
        try:
            text = chunk.decode("latin-1")
        except UnicodeDecodeError:
            return "undecodable as utf-8 or latin-1"
    if not text:
        return "empty file"

    printable = sum(1 for ch in text if ch.isprintable() or ch in "\n\r\t")
    ratio = printable / len(text)
    if ratio < 0.85:
        return f"low printable ratio ({ratio:.2f})"
    return ""


def _resolve_language(path: str, root: str):
    """Resolve (language_code, method) for a file, or ("unknown", "") ."""
    base = os.path.basename(path)
    stem, ext = os.path.splitext(base)
    ext = ext.lstrip(".").lower()

    # Method a: file extension is the language code (opus-100 convention).
    if ext in KNOWN_CODES:
        return ext, "extension"

    # Some files carry the code as the last dot-separated stem segment
    # (e.g. "opus.en-fr-train.en" already handled by ext above; this
    # covers stems like "foo.en.txt" after extension stripping happened
    # upstream, or "foo_en").
    stem_tokens = _tokens(stem)
    for tok in reversed(stem_tokens):
        tl = tok.lower()
        if tl in KNOWN_CODES and tl not in AMBIGUOUS_TOKENS:
            return tl, "filename-token"

    # Method b: walk ancestor directory names (closest first), looking for
    # a whole-word token that matches a known code, excluding ambiguous
    # short English-word collisions.
    rel = os.path.relpath(path, root)
    parts = rel.split(os.sep)[:-1]  # directories only, nearest first at the end
    for dirname in reversed(parts):
        for tok in _tokens(dirname):
            tl = tok.lower()
            if tl in KNOWN_CODES and tl not in AMBIGUOUS_TOKENS:
                return tl, "dirname-token"

    return "unknown", ""


def build_manifest(dictionaries_root: str, output_csv: str):
    entries = []
    for dirpath, dirnames, filenames in os.walk(dictionaries_root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIR_NAMES]
        for fname in filenames:
            fpath = os.path.join(dirpath, fname)
            ext = os.path.splitext(fname)[1].lower()
            try:
                size = os.path.getsize(fpath)
            except OSError:
                size = 0

            if ext in BINARY_EXTENSIONS:
                entries.append(ManifestEntry(fpath, size, False, f"binary extension {ext}",
                                              "unknown", ""))
                continue
            if fname.startswith(".") or ext in NON_PROSE_EXTENSIONS:
                entries.append(ManifestEntry(fpath, size, False, "not prose (dotfile/code/markup)",
                                              "unknown", ""))
                continue

            reason = _sniff_binary(fpath)
            if reason:
                entries.append(ManifestEntry(fpath, size, False, reason, "unknown", ""))
                continue

            lang, method = _resolve_language(fpath, dictionaries_root)
            if lang == "unknown":
                entries.append(ManifestEntry(fpath, size, False, "no resolvable language label",
                                              "unknown", ""))
            else:
                entries.append(ManifestEntry(fpath, size, True, "", lang, method))

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(asdict(entries[0]).keys()) if entries else
                                 ["path", "size_bytes", "usable", "reason",
                                  "resolved_language", "resolution_method"])
        writer.writeheader()
        for e in entries:
            writer.writerow(asdict(e))

    return entries
