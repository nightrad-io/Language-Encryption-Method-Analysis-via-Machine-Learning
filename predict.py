#!/usr/bin/env python3
"""CLI wrapper around client.CipherLanguageClient -- identify the language
and cipher of a piece of text using the trained two-stage model.

Usage:
    python predict.py --text "gur dhvpx oebja sbk whzcf bire gur ynml qbt"
    python predict.py --file mystery.txt

If the cipher is already known, pass --cipher to skip cipher
identification (Stage A) and predict language only -- Stage B then runs
on the known cipher directly instead of a predicted probability
distribution. Run --list-ciphers to see valid ids.

    python predict.py --file mystery.txt --cipher vigenere

Text longer than 10,000 characters (the longest window size seen in
training) is automatically split into chunks and the results averaged --
see --no-chunk / --chunk-size to control this.

For repeated/programmatic queries (e.g. from a server), import
client.CipherLanguageClient directly instead -- it loads the models once
and exposes .predict(text).
"""
import argparse
import json
import os

from client import CipherLanguageClient, MAX_TRAINED_WINDOW


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--text", help="Text to identify")
    g.add_argument("--file", help="Path to a file containing the text")
    p.add_argument("--cipher", default=None,
                    help="If the cipher mechanism is already known, pass its id here to skip "
                         "cipher identification and predict language only. See --list-ciphers.")
    p.add_argument("--list-ciphers", action="store_true", help="Print valid --cipher ids and exit")
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--models-dir", default="models")
    p.add_argument("--eval-report", default="output/model_eval.json")
    p.add_argument("--chunk-size", type=int, default=MAX_TRAINED_WINDOW,
                    help=f"Split input longer than this many characters into chunks and average "
                         f"the results (default: {MAX_TRAINED_WINDOW}, the longest window size "
                         f"used in training)")
    p.add_argument("--no-chunk", action="store_true",
                    help="Disable chunking -- run one raw prediction over the whole input "
                         "regardless of length. Not recommended for long input; mainly useful "
                         "for diagnosing length-extrapolation behavior directly.")
    args = p.parse_args()

    if args.list_ciphers:
        manifest = json.load(open(os.path.join(args.models_dir, "feature_manifest.json")))
        for cipher_id in manifest["stage_a_classes"]:
            print(cipher_id)
        return

    if not args.text and not args.file:
        p.error("one of --text or --file is required (unless --list-ciphers)")

    text = args.text
    if args.file:
        with open(args.file, encoding="utf-8") as fh:
            text = fh.read()

    client = CipherLanguageClient(models_dir=args.models_dir, eval_report=args.eval_report)
    try:
        result = client.predict(text, top_k=args.top_k, known_cipher=args.cipher,
                                 chunk=not args.no_chunk, chunk_size=args.chunk_size)
    except ValueError as e:
        p.error(str(e))

    chunk_note = f", averaged over {result['n_chunks']} chunks" if result["n_chunks"] > 1 else ""
    print(f"Input: {result['n_chars']} characters{chunk_note}, word-boundary-preserving features "
          f"{'available' if result['word_level_applicable'] else 'not available (letters-only cipher output)'}\n")

    if result["cipher_source"] == "user_specified":
        cid, _ = result["top_ciphers"][0]
        print(f"Cipher: {cid} (user-specified, not predicted)")
    else:
        print(f"Top {args.top_k} cipher guesses:")
        for cid, prob in result["top_ciphers"]:
            print(f"  {cid:32s} {prob*100:5.1f}%")

    print(f"\nTop {args.top_k} language guesses:")
    for lid, name, prob in result["top_languages"]:
        print(f"  {lid:6s} {name:24s} {prob*100:5.1f}%")

    if "family_note" in result:
        print(f"\n{result['family_note']}")

    if "length_caveat" in result:
        print(f"\n{result['length_caveat']}")


if __name__ == "__main__":
    main()
