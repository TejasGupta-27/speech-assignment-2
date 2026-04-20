"""Confusion matrix for code-switching boundaries (predicted vs silver).

Each LID segment boundary is labelled by the (prev_lang, next_lang) pair. We count
confusions between predicted transitions and reference transitions when matched
within ±200 ms.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import lid


def transitions(labels):
    labels = np.asarray(labels)
    idx = np.where(np.diff(labels) != 0)[0] + 1
    return [(int(i), int(labels[i - 1]), int(labels[i])) for i in idx]


def main():
    audio = "data/original_denoised.wav"
    weights = "models/lid.pt"
    silver = lid.label_sequence_from_whisper(audio)
    pred = np.array(lid.predict(audio, weights)["frames"])
    n = min(len(silver), len(pred))
    silver, pred = silver[:n], pred[:n]

    t_pred = transitions(pred)
    t_gold = transitions(silver)
    tol = int(0.200 / lid.FRAME_HOP)
    labels = {0: "eng", 1: "hin", 2: "sil"}

    conf = {f"{labels[a]}->{labels[b]}": {f"{labels[c]}->{labels[d]}": 0 for c in labels for d in labels if c != d} for a in labels for b in labels if a != b}

    matched = set()
    for i_p, a_p, b_p in t_pred:
        cand = [(i_g, a_g, b_g) for i_g, a_g, b_g in t_gold if abs(i_g - i_p) <= tol and i_g not in matched]
        if not cand:
            continue
        i_g, a_g, b_g = min(cand, key=lambda x: abs(x[0] - i_p))
        matched.add(i_g)
        key_p = f"{labels[a_p]}->{labels[b_p]}"
        key_g = f"{labels[a_g]}->{labels[b_g]}"
        if key_p in conf and key_g in conf[key_p]:
            conf[key_p][key_g] += 1
    Path("outputs/switch_confusion.json").write_text(json.dumps(conf, indent=2))
    print(json.dumps(conf, indent=2))


if __name__ == "__main__":
    main()
