"""WER evaluation, overall and per-language using LID segmentation.

Splits hypothesis by LID-predicted language segments. For each LID segment, find
the time-overlapping portion of the gold-transcript word list (using word
timestamps), and compute WER on that slice.

Outputs overall, eng-only, and hin-only WER.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import jiwer


def overlap(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default="outputs/gold_transcript.json")
    ap.add_argument("--hyp", default="outputs/transcript.json")
    ap.add_argument("--lid", default="outputs/lid_pred.json")
    ap.add_argument("--out", default="outputs/wer.json")
    args = ap.parse_args()

    gold = json.loads(Path(args.gold).read_text())
    hyp = json.loads(Path(args.hyp).read_text())
    lid = json.loads(Path(args.lid).read_text())

    gold_words = [(w["w"], float(w["s"]), float(w["e"])) for seg in gold["segments"] for w in seg.get("words", [])]
    hyp_words = [(w["w"], float(w["s"]), float(w["e"])) for seg in hyp["segments"] for w in seg.get("words", [])]

    full_ref = " ".join(w for w, *_ in gold_words)
    full_hyp = " ".join(w for w, *_ in hyp_words)
    overall = float(jiwer.wer(full_ref, full_hyp)) if full_ref and full_hyp else float("nan")

    per_lang = {}
    for lang in ("eng", "hin"):
        spans = [(s["start"], s["end"]) for s in lid["segments"] if s["label"] == lang]
        if not spans:
            per_lang[lang] = None
            continue
        ref_chunks = []
        hyp_chunks = []
        for s0, s1 in spans:
            r = " ".join(w for w, ws, we in gold_words if overlap(s0, s1, ws, we) > 0.05)
            h = " ".join(w for w, ws, we in hyp_words if overlap(s0, s1, ws, we) > 0.05)
            if r.strip() and h.strip():
                ref_chunks.append(r)
                hyp_chunks.append(h)
        if not ref_chunks:
            per_lang[lang] = None
        else:
            per_lang[lang] = {
                "wer": float(jiwer.wer(ref_chunks, hyp_chunks)),
                "n_spans": len(ref_chunks),
                "n_words_ref": sum(len(r.split()) for r in ref_chunks),
            }

    out = {"overall_wer": overall, "per_language": per_lang}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
