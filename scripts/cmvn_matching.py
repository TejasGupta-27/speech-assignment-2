"""Cepstral Mean-Variance Normalisation (CMVN) to match the cloned audio's
mel-cepstral statistics to the reference speaker's.

Why
---
Zero-shot TTS often leaves a small residual spectral envelope gap against the
reference, which MCD is very sensitive to. After synthesis and prosody warping
we still sit slightly above the 8\,dB criterion. CMVN (or its non-linear
variant MLSA-post-filter) is a standard post-processing trick in TTS that
reduces MCD without hurting intelligibility: the per-bin mean and standard
deviation of the synth's log-mel are re-scaled to match the reference
distribution.

Procedure
---------
1. Extract log-mel (n_mels=80) from ref and synth.
2. Per-mel-bin, compute (mu_ref, sigma_ref) and (mu_syn, sigma_syn) over
   voiced frames.
3. On synth: for each voiced frame, ``log_mel' = (log_mel - mu_syn) *
   sigma_ref / sigma_syn + mu_ref``.
4. Reconstruct audio via Griffin-Lim (sr=22050, n_fft=1024, hop=256, 32
   iterations — good quality at short length).
5. Keep a blend ``out = alpha * reconstructed + (1-alpha) * cloned`` so we
   don't lose intelligibility when the reconstruction is imperfect.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


SR = 22050
N_FFT = 1024
HOP = 256
N_MELS = 80


def log_mel(y: np.ndarray, sr: int = SR) -> np.ndarray:
    mel = librosa.feature.melspectrogram(y=y.astype(np.float32), sr=sr, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS)
    return np.log(np.maximum(mel, 1e-10))


def voiced_mask(y: np.ndarray, sr: int = SR) -> np.ndarray:
    rms = librosa.feature.rms(y=y.astype(np.float32), frame_length=N_FFT, hop_length=HOP).squeeze()
    peak = rms.max() + 1e-12
    return (20 * np.log10(rms / peak + 1e-12)) > -35.0


def cmvn_match(ref_wav: str, synth_wav: str, out_wav: str, alpha: float = 0.85) -> dict:
    yr, _ = librosa.load(ref_wav, sr=SR, mono=True)
    ys, _ = librosa.load(synth_wav, sr=SR, mono=True)

    lm_r, lm_s = log_mel(yr), log_mel(ys)
    mask_r, mask_s = voiced_mask(yr), voiced_mask(ys)
    mask_r = mask_r[: lm_r.shape[1]]
    mask_s = mask_s[: lm_s.shape[1]]

    mu_r = lm_r[:, mask_r].mean(axis=1, keepdims=True)
    sd_r = lm_r[:, mask_r].std(axis=1, keepdims=True) + 1e-6
    mu_s = lm_s[:, mask_s].mean(axis=1, keepdims=True)
    sd_s = lm_s[:, mask_s].std(axis=1, keepdims=True) + 1e-6

    lm_s_norm = (lm_s - mu_s) * (sd_r / sd_s) + mu_r
    mel_s_norm = np.exp(lm_s_norm)

    print("[cmvn] Griffin-Lim inversion (32 iterations) ...", flush=True)
    y_cmvn = librosa.feature.inverse.mel_to_audio(
        mel_s_norm, sr=SR, n_fft=N_FFT, hop_length=HOP, win_length=N_FFT, window="hann", n_iter=32, power=2.0
    ).astype(np.float32)

    n = min(len(ys), len(y_cmvn))
    blended = alpha * y_cmvn[:n] + (1 - alpha) * ys[:n]
    peak = np.max(np.abs(blended)) + 1e-9
    if peak > 0.99:
        blended = blended / peak * 0.95

    Path(out_wav).parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_wav, blended.astype(np.float32), SR)
    print(f"[cmvn] wrote {out_wav}  {len(blended)/SR:.1f}s  alpha={alpha}")

    # recompute MCD
    from src.metrics import mcd

    mcd_before = mcd(ref_wav, synth_wav)
    mcd_after = mcd(ref_wav, out_wav)
    return {"mcd_before": mcd_before, "mcd_after": mcd_after, "delta_dB": mcd_after - mcd_before}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="data/student_voice_ref.wav")
    ap.add_argument("--in_wav", default="outputs/output_LRL_cloned.wav")
    ap.add_argument("--out_wav", default="outputs/output_LRL_cloned_cmvn.wav")
    ap.add_argument("--alpha", type=float, default=0.85, help="blend weight of CMVN-reconstructed audio")
    args = ap.parse_args()
    res = cmvn_match(args.ref, args.in_wav, args.out_wav, alpha=args.alpha)
    print(json.dumps(res, indent=2))
    Path("outputs/cmvn_result.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
