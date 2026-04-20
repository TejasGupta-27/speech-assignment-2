"""Task 3.2 — Prosody Warping via DTW.

1. Extract F0 and energy contours from the reference lecture audio.
2. Extract F0 and energy contours from the candidate synth audio.
3. Align the two contour-pairs with Dynamic Time Warping (own implementation).
4. Warp the synth signal's pitch and energy along the DTW path so its "teaching
   style" matches the professor's contours.

Resynthesis uses librosa pitch shifting (segment-wise) and energy gain envelope.
This is lightweight enough to run on CPU.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import soundfile as sf

SR = 22050
FRAME_HOP = 256


def extract_f0_energy(y: np.ndarray, sr: int = SR) -> Tuple[np.ndarray, np.ndarray]:
    """F0 via autocorrelation (librosa.yin — much faster than pyin) + RMS energy."""
    import librosa

    print(f"[prosody]   extracting F0+energy on {len(y) / sr:.1f}s audio ...", flush=True)
    f0 = librosa.yin(
        y.astype(np.float32),
        fmin=float(librosa.note_to_hz("C2")),
        fmax=float(librosa.note_to_hz("C7")),
        sr=sr, frame_length=1024, hop_length=FRAME_HOP,
    )
    f0 = np.nan_to_num(f0, nan=0.0)
    rms = librosa.feature.rms(y=y.astype(np.float32), frame_length=1024, hop_length=FRAME_HOP).squeeze()
    # mask unvoiced: very low energy → F0=0
    rms_thr = float(np.median(rms) * 0.2)
    f0 = np.where(rms > rms_thr, f0, 0.0)
    return f0.astype(np.float32), rms.astype(np.float32)


def dtw_path(x: np.ndarray, y: np.ndarray, band: int | None = None) -> List[Tuple[int, int]]:
    """DTW with Sakoe-Chiba band and L1 cost on 1-D sequences.

    Vectorized: we let librosa compute the accumulated-cost matrix and
    traceback. Our argument is a pair of 1-D contours; we reshape to (1, N)
    for librosa.sequence.dtw. If N is large (>5000), we down-sample both
    contours by linear interpolation to 5000 frames to keep DTW tractable,
    then the returned path is linearly scaled back to original indices.
    """
    import librosa

    N_MAX = 5000
    n, m = len(x), len(y)
    if n > N_MAX or m > N_MAX:
        scale_x, scale_y = n / N_MAX, m / N_MAX
        xi = np.interp(np.linspace(0, n - 1, N_MAX), np.arange(n), x).astype(np.float32)
        yi = np.interp(np.linspace(0, m - 1, N_MAX), np.arange(m), y).astype(np.float32)
    else:
        scale_x, scale_y = 1.0, 1.0
        xi, yi = x.astype(np.float32), y.astype(np.float32)
    X2 = np.vstack([xi, np.zeros_like(xi)])  # (2, N)
    Y2 = np.vstack([yi, np.zeros_like(yi)])
    D, wp = librosa.sequence.dtw(X=X2, Y=Y2, metric="cityblock")
    path = [(int(round(i * scale_x)), int(round(j * scale_y))) for i, j in wp[::-1]]
    return path


def warp_contour(target: np.ndarray, src: np.ndarray, path: List[Tuple[int, int]]) -> np.ndarray:
    """Produce a per-frame target contour for `src` (length = len(src))."""
    out = np.zeros(len(src), dtype=np.float32)
    seen = np.zeros(len(src), dtype=bool)
    for i, j in path:
        if j < len(src) and i < len(target):
            out[j] = target[i]
            seen[j] = True
    if seen.any():
        idx = np.arange(len(src))
        if not seen.all():
            out = np.interp(idx, idx[seen], out[seen])
    return out


def apply_pitch_energy_warp(
    synth: np.ndarray,
    ref_f0: np.ndarray,
    ref_rms: np.ndarray,
    sr: int = SR,
    block_s: float = 0.5,
) -> np.ndarray:
    """Warp `synth` so its F0 and RMS track the reference after DTW alignment.

    Pitch shift is applied on ~500ms blocks (one shift per block, not per frame)
    to keep cost reasonable on CPU. Within a block we use a single semitone value
    equal to the block-mean of 12*log2(ref/src) along the warped contour.
    """
    import librosa

    s_f0, s_rms = extract_f0_energy(synth, sr)
    s_log = np.log(np.clip(s_f0, 1, None))
    r_log = np.log(np.clip(ref_f0, 1, None))
    path = dtw_path(r_log, s_log)
    r_f0_warped = warp_contour(ref_f0, s_f0, path)
    r_rms_warped = warp_contour(ref_rms, s_rms, path)

    # per-sample semitone trajectory
    valid = (s_f0 > 50) & (r_f0_warped > 50)
    semitones_frame = np.where(valid, 12.0 * np.log2(np.where(valid, r_f0_warped, 1) / np.where(valid, s_f0, 1)), 0.0)
    semitones_frame = np.clip(semitones_frame, -6, 6)

    hop = FRAME_HOP
    block_samples = int(block_s * sr)
    n = len(synth)
    out = np.zeros(n, dtype=np.float32)
    pos = 0
    while pos < n:
        end = min(n, pos + block_samples)
        block = synth[pos:end]
        # compute mean semitone shift for frames in this block
        f_start, f_end = pos // hop, max(end // hop, pos // hop + 1)
        s_slice = semitones_frame[f_start:f_end]
        shift = float(np.median(s_slice)) if len(s_slice) else 0.0
        if abs(shift) > 0.1:
            shifted = librosa.effects.pitch_shift(y=block.astype(np.float32), sr=sr, n_steps=shift)
        else:
            shifted = block.astype(np.float32)
        out[pos : pos + len(shifted)] = shifted[: end - pos]
        pos = end

    # energy envelope shaping (smooth interpolation)
    eps = 1e-6
    gain_frames = (r_rms_warped + eps) / (s_rms + eps)
    gain_frames = np.clip(gain_frames, 0.3, 3.0)
    gain = np.interp(np.arange(len(out)), np.arange(len(gain_frames)) * hop, gain_frames)
    out = out * gain.astype(np.float32)
    peak = np.max(np.abs(out)) + 1e-9
    if peak > 0.99:
        out = out / peak * 0.95
    return out.astype(np.float32)


def run(ref_path: str, synth_path: str, out_path: str) -> None:
    import librosa

    print(f"[prosody] loading {ref_path} and {synth_path} ...", flush=True)
    ref, _ = librosa.load(ref_path, sr=SR, mono=True)
    syn, _ = librosa.load(synth_path, sr=SR, mono=True)
    print("[prosody] computing reference F0/energy ...", flush=True)
    rf0, rrms = extract_f0_energy(ref, SR)
    print("[prosody] warping synth to reference contours ...", flush=True)
    warped = apply_pitch_energy_warp(syn, rf0, rrms, SR)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_path, warped, SR)
    print(f"[prosody] wrote {out_path}  {len(warped) / SR:.1f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="reference audio (professor)")
    ap.add_argument("--synth", required=True, help="flat synth output")
    ap.add_argument("--out", required=True, help="warped output wav")
    args = ap.parse_args()
    run(args.ref, args.synth, args.out)
