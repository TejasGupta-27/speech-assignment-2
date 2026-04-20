"""Ablation: flat TTS synth vs DTW-warped synth.

Runs the TTS pipeline twice on the same LRL text — once flat, once prosody-warped
from the professor reference — and writes MCD / speaker-cosine metrics.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import embed, metrics


def main():
    ref = "data/original_denoised.wav"
    flat = "outputs/output_LRL_flat.wav"
    warped = "outputs/output_LRL_cloned.wav"
    ref_voice = "data/student_voice_ref.wav"

    out = {}
    if Path(flat).exists() and Path(warped).exists():
        out["mcd_flat_vs_ref"] = metrics.mcd(ref, flat)
        out["mcd_warped_vs_ref"] = metrics.mcd(ref, warped)
        out["delta_mcd"] = out["mcd_warped_vs_ref"] - out["mcd_flat_vs_ref"]
    if Path(ref_voice).exists() and Path(flat).exists() and Path(warped).exists():
        e_ref = embed.stats_pool_dvector(_load(ref_voice))
        e_flat = embed.stats_pool_dvector(_load(flat))
        e_w = embed.stats_pool_dvector(_load(warped))
        out["cos_flat_to_ref"] = float(np.dot(e_flat, e_ref))
        out["cos_warped_to_ref"] = float(np.dot(e_w, e_ref))
    Path("outputs/ablation_prosody.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


def _load(p):
    import librosa

    y, _ = librosa.load(p, sr=16000, mono=True)
    return y


if __name__ == "__main__":
    main()
