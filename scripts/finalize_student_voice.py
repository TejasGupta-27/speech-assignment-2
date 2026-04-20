"""Finalize the pipeline after XTTS has been re-run with the student's real 60 s voice.

Steps (all assume `data/student_voice_ref.wav` is now the student's recording):
  1. DTW-warp F0/energy of the XTTS output onto the original Modi-side lecture.
  2. CMVN-match the warped output's cepstral stats against the student reference.
  3. Re-train the anti-spoof CM with bona-fide = student + Modi lecture,
     spoof = flat-XTTS + warp+CMVN synth.
  4. Recompute the full MCD grid and the ablation (flat vs warped).
  5. Refresh outputs/metrics_summary.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def make_60s(src: str, dst: str, sr: int = 22050, dur: float = 60.0) -> None:
    y, _ = librosa.load(src, sr=sr, mono=True)
    sf.write(dst, y[: int(dur * sr)].astype(np.float32), sr)


def main() -> None:
    from src import prosody, embed, metrics, antispoof

    flat = "outputs/output_LRL_xtts.wav"
    warped = "outputs/output_LRL_cloned_prewarped.wav"
    cloned = "outputs/output_LRL_cloned.wav"

    # 1) prosody warp
    print("[final] step 1/5 prosody warp …", flush=True)
    prosody.run("data/original_denoised.wav", flat, warped)

    # 2) CMVN match vs student voice ref
    print("[final] step 2/5 CMVN match α=0.5 …", flush=True)
    from scripts.cmvn_matching import cmvn_match

    cmvn_match("data/student_voice_ref.wav", warped, cloned, alpha=0.5)

    # 3) anti-spoof re-train (bona = student voice + original, spoof = XTTS + cloned)
    print("[final] step 3/5 anti-spoof CM re-train …", flush=True)
    cm_res = antispoof.train_and_eval(
        bona_paths=["data/student_voice_ref.wav", "data/original_denoised.wav"],
        spoof_paths=[flat, cloned],
        out_weights="models/cm.pt",
        feature="lfcc",
        epochs=20,
    )
    Path("outputs/cm_eer.json").write_text(json.dumps(cm_res, indent=2))

    # 4) MCD grid
    print("[final] step 4/5 MCD grid …", flush=True)
    make_60s("data/original_denoised.wav", "/tmp/prof_60s.wav", dur=60)
    make_60s("data/student_voice_ref.wav", "/tmp/ref_60s.wav", dur=60)
    make_60s(flat, "/tmp/flat_60s.wav", dur=60)
    make_60s(cloned, "/tmp/cloned_60s.wav", dur=60)
    mcd_out = {
        "self_probe": metrics.mcd("outputs/parallel_probe.wav", "outputs/parallel_probe.wav"),
        "parallel_probe_vs_xtts": metrics.mcd("outputs/parallel_probe.wav", "outputs/parallel_synth_xtts.wav"),
        "flat60_vs_prof60": metrics.mcd("/tmp/prof_60s.wav", "/tmp/flat_60s.wav"),
        "cloned60_vs_prof60": metrics.mcd("/tmp/prof_60s.wav", "/tmp/cloned_60s.wav"),
        "flat60_vs_ref60": metrics.mcd("/tmp/ref_60s.wav", "/tmp/flat_60s.wav"),
        "cloned60_vs_ref60": metrics.mcd("/tmp/ref_60s.wav", "/tmp/cloned_60s.wav"),
    }
    Path("outputs/mcd.json").write_text(json.dumps(mcd_out, indent=2))

    # ablation
    e_ref = embed.ecapa_xvector("data/student_voice_ref.wav")
    e_flat = embed.ecapa_xvector(flat)
    e_warp = embed.ecapa_xvector(cloned)
    abl = {
        "mcd_flat_vs_ref": mcd_out["flat60_vs_ref60"],
        "mcd_warped_vs_ref": mcd_out["cloned60_vs_ref60"],
        "delta_mcd": mcd_out["cloned60_vs_ref60"] - mcd_out["flat60_vs_ref60"],
        "cos_flat_to_ref": float(np.dot(e_flat, e_ref)),
        "cos_warped_to_ref": float(np.dot(e_warp, e_ref)),
    }
    Path("outputs/ablation_prosody.json").write_text(json.dumps(abl, indent=2))
    print("[final] MCD:", json.dumps(mcd_out, indent=2))
    print("[final] Ablation:", json.dumps(abl, indent=2))

    # 5) refresh summary
    print("[final] step 5/5 metrics summary …", flush=True)
    summary = {}
    for f in ["lid_eval.json", "fgsm.json", "cm_eer.json", "mcd.json", "ablation_prosody.json", "wer.json"]:
        p = Path("outputs") / f
        if p.exists():
            try:
                summary[f] = json.loads(p.read_text())
            except Exception:
                pass

    def pass_lt(v, thr):
        return (v is not None) and (v < thr)

    out = {
        "lid": {
            "macro_f1_eng_hin": summary.get("lid_eval.json", {}).get("lid", {}).get("macro_f1_eng_hin"),
            "criterion_pass_ge_0.85": (summary.get("lid_eval.json", {}).get("lid", {}).get("macro_f1_eng_hin") or 0) >= 0.85,
            "timing_hit_rate_200ms_weak_silver": summary.get("lid_eval.json", {}).get("timing", {}).get("hit_rate"),
        },
        "stt_wer": {
            "overall_wer": summary.get("wer.json", {}).get("overall_wer"),
            "eng_wer": summary.get("wer.json", {}).get("eng_wer"),
            "hin_wer": summary.get("wer.json", {}).get("hin_wer"),
        },
        "antispoof": {
            "eer": summary.get("cm_eer.json", {}).get("eer"),
            "criterion_under_10pct": pass_lt(summary.get("cm_eer.json", {}).get("eer"), 0.10),
        },
        "adversarial_fgsm": {
            "min_eps_any_flip": summary.get("fgsm.json", {}).get("min_eps_any_flip", {}).get("epsilon"),
            "min_eps_inaudible_flip_SNR_over_40dB": summary.get("fgsm.json", {}).get("min_eps_inaudible_flip"),
        },
        "mcd_pymcd": mcd_out | {
            "criterion_cloned_under_8dB_vs_ref": pass_lt(mcd_out.get("cloned60_vs_ref60"), 8.0),
        },
        "ablation_prosody": abl,
    }
    Path("outputs/metrics_summary.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
