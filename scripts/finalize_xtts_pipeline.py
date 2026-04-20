"""Once XTTS full synthesis is done, run this to:
1) Apply prosody warping onto the XTTS-cloned audio.
2) Overwrite outputs/output_LRL_cloned.wav with the XTTS+warped version.
3) Recompute MCD, speaker-cosine, and the ablation.
4) Refresh outputs/metrics_summary.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import embed, metrics, prosody


def main():
    xtts = Path("outputs/output_LRL_xtts.wav")
    if not xtts.exists():
        raise FileNotFoundError("Run scripts/tts_xtts_full.py first to produce outputs/output_LRL_xtts.wav")

    # 1) Warp XTTS output onto the professor's prosody contour.
    cloned = "outputs/output_LRL_cloned.wav"
    prosody.run("data/original_denoised.wav", str(xtts), cloned)

    # 2) keep a copy of the pre-warp XTTS as "flat" comparator for ablation
    flat = "outputs/output_LRL_xtts.wav"  # use xtts as the new "flat"

    # 3) Make 60s excerpts for the MCD grid
    import librosa
    import soundfile as sf

    for src, dst, dur in [
        ("data/original_denoised.wav", "/tmp/prof_60s.wav", 60),
        ("data/student_voice_ref.wav", "/tmp/ref_60s.wav", 60),
        (flat, "/tmp/flat_60s.wav", 60),
        (cloned, "/tmp/cloned_60s.wav", 60),
    ]:
        y, sr = librosa.load(src, sr=22050, mono=True)
        sf.write(dst, y[: int(dur * sr)].astype(np.float32), sr)

    print("[final] computing MCD grid with pymcd ...")
    mcd_out = {
        "self_probe": metrics.mcd("outputs/parallel_probe.wav", "outputs/parallel_probe.wav"),
        "probe_vs_xtts_parallel": metrics.mcd("outputs/parallel_probe.wav", "outputs/parallel_synth_xtts.wav"),
        "flat60_vs_prof60": metrics.mcd("/tmp/prof_60s.wav", "/tmp/flat_60s.wav"),
        "cloned60_vs_prof60": metrics.mcd("/tmp/prof_60s.wav", "/tmp/cloned_60s.wav"),
        "flat60_vs_ref60": metrics.mcd("/tmp/ref_60s.wav", "/tmp/flat_60s.wav"),
        "cloned60_vs_ref60": metrics.mcd("/tmp/ref_60s.wav", "/tmp/cloned_60s.wav"),
    }
    Path("outputs/mcd.json").write_text(json.dumps(mcd_out, indent=2))

    # 4) Ablation (flat XTTS vs warped XTTS)
    print("[final] computing ablation (flat vs warped XTTS) ...")
    import numpy as np

    def load(p):
        y, _ = librosa.load(p, sr=16000, mono=True)
        return y

    e_ref = embed.ecapa_xvector("data/student_voice_ref.wav")
    e_flat = embed.ecapa_xvector(flat)
    e_warp = embed.ecapa_xvector(cloned)
    abl = {
        "mcd_flat_vs_prof": metrics.mcd("/tmp/prof_60s.wav", "/tmp/flat_60s.wav"),
        "mcd_warped_vs_prof": metrics.mcd("/tmp/prof_60s.wav", "/tmp/cloned_60s.wav"),
        "cos_flat_to_ref": float(np.dot(e_flat, e_ref)),
        "cos_warped_to_ref": float(np.dot(e_warp, e_ref)),
    }
    abl["delta_mcd"] = abl["mcd_warped_vs_prof"] - abl["mcd_flat_vs_prof"]
    Path("outputs/ablation_prosody.json").write_text(json.dumps(abl, indent=2))

    # 5) Refresh summary
    print("[final] regenerating metrics_summary.json ...")
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
            "eng_f1": summary.get("lid_eval.json", {}).get("lid", {}).get("report", {}).get("eng", {}).get("f1-score"),
            "hin_f1": summary.get("lid_eval.json", {}).get("lid", {}).get("report", {}).get("hin", {}).get("f1-score"),
            "criterion_pass_ge_0.85": not pass_lt(summary.get("lid_eval.json", {}).get("lid", {}).get("macro_f1_eng_hin", 0), 0.85),
            "timing_hit_rate_200ms_weak_silver": summary.get("lid_eval.json", {}).get("timing", {}).get("hit_rate"),
        },
        "stt_wer": {
            "overall_wer": summary.get("wer.json", {}).get("overall_wer"),
            "overall_cer": summary.get("wer.json", {}).get("overall_cer"),
            "eng_wer": summary.get("wer.json", {}).get("eng_wer"),
            "hin_wer": summary.get("wer.json", {}).get("hin_wer"),
            "criterion_eng_under_15pct": pass_lt(summary.get("wer.json", {}).get("eng_wer"), 0.15),
            "criterion_hin_under_25pct": pass_lt(summary.get("wer.json", {}).get("hin_wer"), 0.25),
            "note": "No human gold. Proxy gold = whisper-small beam=5. WER inflated by greedy hypothesis silence hallucinations.",
        },
        "antispoof": {
            "eer": summary.get("cm_eer.json", {}).get("eer"),
            "threshold": summary.get("cm_eer.json", {}).get("threshold"),
            "criterion_under_10pct": pass_lt(summary.get("cm_eer.json", {}).get("eer"), 0.10),
        },
        "adversarial_fgsm": {
            "min_eps_any_flip": summary.get("fgsm.json", {}).get("min_eps_any_flip", {}).get("epsilon"),
            "feature_snr_at_min_eps_db": summary.get("fgsm.json", {}).get("min_eps_any_flip", {}).get("snr_feature_db"),
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
