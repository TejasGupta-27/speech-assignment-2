"""Rebuild frame-level LID gold from Whisper word timestamps.

A word's language is inferred from script:
  - any Devanagari code point      -> hin
  - otherwise Latin letters        -> eng
  - pure punctuation / noise       -> sil

For each word we fill its (start, end) span at 100 fps with the inferred label.
Frames between words are labelled as silence.

Output:
  outputs/lid_gold.npy              (T,) int64 per-frame labels
  outputs/lid_gold_transitions.json list of (start_sec, end_sec, label)
  outputs/lid_eval_gold.json        F1 + 200ms hit-rate recomputed
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.lid import FRAME_HOP, LABEL_MAP, evaluate_f1, boundary_timing_accuracy

DEV = re.compile(r"[\u0900-\u097F]")
LAT = re.compile(r"[A-Za-z]")


def word_to_label(w: str) -> int:
    if DEV.search(w):
        return LABEL_MAP["hin"]
    if LAT.search(w):
        return LABEL_MAP["eng"]
    return LABEL_MAP["sil"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold_json", default="outputs/gold_transcript.json")
    ap.add_argument("--audio", default="data/original_denoised.wav")
    ap.add_argument("--weights", default="models/lid.pt")
    ap.add_argument("--out_npy", default="outputs/lid_gold.npy")
    ap.add_argument("--out_trans", default="outputs/lid_gold_transitions.json")
    ap.add_argument("--out_eval", default="outputs/lid_eval_gold.json")
    args = ap.parse_args()

    import librosa

    y, _ = librosa.load(args.audio, sr=16000, mono=True)
    n_frames = int(np.ceil(len(y) / (FRAME_HOP * 16000)))
    labels = np.full(n_frames, LABEL_MAP["sil"], dtype=np.int64)

    gold = json.loads(Path(args.gold_json).read_text(encoding="utf-8"))
    transitions = []
    prev_lbl = None
    for seg in gold["segments"]:
        for w in seg.get("words", []):
            lbl = word_to_label(w["w"])
            s, e = float(w["s"]), float(w["e"])
            fs, fe = int(s / FRAME_HOP), int(min(e / FRAME_HOP, n_frames))
            labels[fs:fe] = lbl
            if prev_lbl is not None and prev_lbl != lbl:
                transitions.append({"t": round(s, 3), "from": prev_lbl, "to": lbl})
            prev_lbl = lbl

    np.save(args.out_npy, labels)
    Path(args.out_trans).write_text(json.dumps(transitions, indent=2, ensure_ascii=False))
    n_eng = int((labels == LABEL_MAP["eng"]).sum())
    n_hin = int((labels == LABEL_MAP["hin"]).sum())
    n_sil = int((labels == LABEL_MAP["sil"]).sum())
    print(f"[gold-lid] frames: eng={n_eng} hin={n_hin} sil={n_sil}  transitions={len(transitions)}", flush=True)

    m = evaluate_f1(args.audio, args.weights, args.out_npy)
    t = boundary_timing_accuracy(args.audio, args.weights, args.out_npy, tol_ms=200)
    Path(args.out_eval).write_text(json.dumps({"lid": m, "timing": t}, indent=2))
    print("[gold-lid] F1/timing on word-level gold:")
    print(json.dumps({"macro_f1_eng_hin": m["macro_f1_eng_hin"], "timing": t}, indent=2))


if __name__ == "__main__":
    main()
