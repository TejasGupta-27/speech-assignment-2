"""Remove Whisper hallucinations from a transcript.

Heuristics
----------
- Drop segments with avg_logprob < -1.0 (acoustic model very uncertain).
- Drop segments with compression ratio > 2.4 (indicates repeated tokens).
- Drop segments where any token repeats > 6 times (bada-bada-bada style).
- Deduplicate consecutive identical words.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def has_excessive_repeat(text: str, thresh: int = 6) -> bool:
    toks = re.findall(r"[\u0900-\u097FA-Za-z]+", text)
    if not toks:
        return False
    from collections import Counter

    c = Counter(toks)
    return max(c.values()) > thresh


def compression_ratio(text: str) -> float:
    import zlib

    b = text.encode("utf-8")
    if not b:
        return 1.0
    return len(b) / max(1, len(zlib.compress(b)))


def clean_segments(segments):
    out = []
    for s in segments:
        t = s.get("text", "").strip()
        if not t:
            continue
        if s.get("avg_logprob", 0) < -1.0:
            continue
        if compression_ratio(t) > 2.4:
            continue
        if has_excessive_repeat(t):
            continue
        # dedupe consecutive identical tokens
        toks = re.findall(r"\S+", t)
        deduped = [toks[0]] if toks else []
        for w in toks[1:]:
            if w != deduped[-1]:
                deduped.append(w)
        s["text"] = " ".join(deduped)
        out.append(s)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_json", default="outputs/transcript.json")
    ap.add_argument("--out_json", default="outputs/transcript_clean.json")
    ap.add_argument("--out_txt", default="outputs/transcript_clean.txt")
    args = ap.parse_args()
    obj = json.loads(Path(args.in_json).read_text(encoding="utf-8"))
    before = len(obj["segments"])
    obj["segments"] = clean_segments(obj["segments"])
    after = len(obj["segments"])
    Path(args.out_json).write_text(json.dumps(obj, indent=2, ensure_ascii=False))
    Path(args.out_txt).write_text("\n".join(s["text"] for s in obj["segments"]))
    print(f"[clean] kept {after}/{before} segments  → {args.out_json}")


if __name__ == "__main__":
    main()
