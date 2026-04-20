"""Task 1.3 — Denoising & Normalization.

Primary: DeepFilterNet (neural). Fallback: Spectral Subtraction (Boll, 1979).
"""
from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path

import numpy as np
import soundfile as sf

warnings.filterwarnings("ignore")


def spectral_subtraction(
    y: np.ndarray, sr: int, noise_duration: float = 0.5, alpha: float = 2.0, beta: float = 0.02
) -> np.ndarray:
    """Classical Boll spectral subtraction.

    noise spectrum estimated from first `noise_duration` seconds.
    alpha over-subtraction factor, beta spectral floor.
    """
    import librosa

    n_fft = 512
    hop = 128
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop, window="hann")
    mag, phase = np.abs(S), np.angle(S)
    n_frames_noise = max(1, int(noise_duration * sr / hop))
    noise_mag = np.mean(mag[:, :n_frames_noise] ** 2, axis=1, keepdims=True) ** 0.5
    clean_mag = mag - alpha * noise_mag
    floor = beta * mag
    clean_mag = np.maximum(clean_mag, floor)
    y_clean = librosa.istft(clean_mag * np.exp(1j * phase), hop_length=hop, length=len(y))
    return y_clean.astype(np.float32)


def deepfilter_denoise(y: np.ndarray, sr: int, chunk_s: float = 60.0) -> np.ndarray:
    """DeepFilterNet denoising at 48 kHz with chunked processing so long
    clips don't exhaust memory. 60 s chunks with 1 s cross-fade overlap."""
    import torch
    import torchaudio
    from df.enhance import enhance, init_df

    model, df_state, _ = init_df(log_level="NONE")
    df_sr = df_state.sr()

    t_all = torch.from_numpy(y).float().unsqueeze(0)
    if sr != df_sr:
        t_all = torchaudio.functional.resample(t_all, sr, df_sr)

    n = t_all.shape[-1]
    chunk = int(chunk_s * df_sr)
    overlap = int(1.0 * df_sr)  # 1 s cross-fade
    out = torch.zeros_like(t_all)
    win = torch.hann_window(2 * overlap)

    start = 0
    idx = 0
    while start < n:
        end = min(n, start + chunk)
        pad_left = overlap if start > 0 else 0
        pad_right = overlap if end < n else 0
        seg_start = max(0, start - pad_left)
        seg_end = min(n, end + pad_right)
        seg = t_all[:, seg_start:seg_end]
        enhanced = enhance(model, df_state, seg)
        # drop pads
        enhanced = enhanced[:, pad_left : pad_left + (end - start)]
        if idx > 0:
            fade = win[:overlap][None].to(enhanced.dtype)
            out[:, start : start + overlap] = (
                out[:, start : start + overlap] * (1 - fade) + enhanced[:, :overlap] * fade
            )
            out[:, start + overlap : end] = enhanced[:, overlap:]
        else:
            out[:, start:end] = enhanced
        idx += 1
        start = end
        print(f"[denoise]   chunk {idx} {start / df_sr:.0f}/{n / df_sr:.0f}s", flush=True)

    out_np = out.squeeze(0).cpu().numpy()
    if sr != df_sr:
        out_np = (
            torchaudio.functional.resample(torch.from_numpy(out_np).unsqueeze(0), df_sr, sr)
            .squeeze(0)
            .numpy()
        )
    return out_np.astype(np.float32)


def peak_normalize(y: np.ndarray, target_dbfs: float = -3.0) -> np.ndarray:
    peak = np.max(np.abs(y)) + 1e-9
    target = 10 ** (target_dbfs / 20)
    return (y * target / peak).astype(np.float32)


def run(in_path: str, out_path: str, method: str = "deepfilter") -> None:
    import librosa

    y, sr = librosa.load(in_path, sr=None, mono=True)
    if method == "deepfilter":
        try:
            y = deepfilter_denoise(y, sr)
        except Exception as e:
            print(f"[denoise] DeepFilterNet failed: {e}. Falling back to spectral subtraction.")
            y = spectral_subtraction(y, sr)
    else:
        y = spectral_subtraction(y, sr)
    y = peak_normalize(y)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_path, y, sr)
    print(f"[denoise] wrote {out_path} ({len(y) / sr:.1f}s, sr={sr})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", dest="out_path", required=True)
    ap.add_argument("--method", choices=["deepfilter", "spectral"], default="deepfilter")
    args = ap.parse_args()
    run(args.in_path, args.out_path, args.method)
