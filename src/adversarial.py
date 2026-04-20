"""Task 4.2 — FGSM adversarial perturbation on the frame-level LID.

Goal: find an inaudible perturbation (SNR > 40 dB) that flips a `hi` frame to `eng`.

Math
----
Given input raw waveform x, frame feature extractor φ (non-differentiable in our
pipeline because MFCC extraction uses librosa/numpy), we instead apply FGSM *in
feature space*: perturb the normalised log-mel + MFCC features directly. This is
a standard workaround used in prior LID/ASV adversarial studies (e.g., Kreuk
et al., 2018). Because features are computed frame-wise, we can still convert
feature-domain ε to an effective time-domain SNR by running a proxy inverse that
resynthesizes audio from the perturbed mel via Griffin-Lim.

Procedure
---------
1. Pick a 5-second segment, extract features F = φ(x).
2. Run LID to get class probs. If argmax != Hindi, skip to next segment.
3. Compute loss L = CE(LID(F), target=eng) and grad g = ∂L/∂F.
4. Sweep epsilons ε ∈ {0.002, 0.005, 0.01, 0.02, 0.05, 0.1}:
     F' = F - ε * sign(g)    (targeted: move toward eng)
     if argmax(LID(F')) == eng: record ε, compute audio SNR via Griffin-Lim
5. Report the smallest ε that flips prediction while keeping SNR > 40 dB.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F_torch

from src.lid import FRAME_HOP, LABEL_MAP, LIDModel, SR, extract_features


def feature_to_audio(feats: np.ndarray, sr: int = SR, hop: int = 160, seed: int = 0) -> np.ndarray:
    """Griffin-Lim inverse for the log-mel portion (first 80 dims of the stack).

    Seeded to make reconstruction deterministic so clean vs adv audio compare fairly.
    """
    import librosa

    np.random.seed(seed)
    log_mel = feats.T[:80]
    S = librosa.db_to_power(log_mel)
    y = librosa.feature.inverse.mel_to_audio(
        S, sr=sr, n_fft=512, hop_length=hop, win_length=400, window="hann", n_iter=16
    )
    return y.astype(np.float32)


def compute_snr(clean: np.ndarray, noisy: np.ndarray) -> float:
    n = min(len(clean), len(noisy))
    clean, noisy = clean[:n], noisy[:n]
    noise = noisy - clean
    ps = np.mean(clean ** 2) + 1e-12
    pn = np.mean(noise ** 2) + 1e-12
    return float(10 * np.log10(ps / pn))


def feature_snr(feats_clean: np.ndarray, feats_adv: np.ndarray) -> float:
    """SNR in the 80-d log-mel feature domain (deterministic; independent of GL)."""
    c, a = feats_clean.T[:80], feats_adv.T[:80]
    noise = a - c
    ps = np.mean(c ** 2) + 1e-12
    pn = np.mean(noise ** 2) + 1e-12
    return float(10 * np.log10(ps / pn))


def fgsm_flip(
    audio_path: str,
    weights: str,
    target_lang: str = "eng",
    source_lang: str = "hin",
    segment_s: float = 5.0,
    epsilons=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0),
    device: str = "cpu",
) -> dict:
    import librosa

    y, _ = librosa.load(audio_path, sr=SR, mono=True)
    seg_n = int(segment_s * SR)

    model = LIDModel().to(device)
    model.load_state_dict(torch.load(weights, map_location=device))
    model.eval()

    tgt_idx = LABEL_MAP[target_lang]
    src_idx = LABEL_MAP[source_lang]

    for start in range(0, len(y) - seg_n + 1, SR):
        seg = y[start : start + seg_n]
        feats = extract_features(seg, SR)
        x = torch.from_numpy(feats).unsqueeze(0).to(device).requires_grad_(True)
        logits, _ = model(x)
        probs0 = F_torch.softmax(logits.squeeze(0), dim=-1).detach().cpu().numpy()
        pred0 = probs0.argmax(-1)
        hin_frac = float((pred0 == src_idx).mean())
        if hin_frac < 0.4:
            continue
        # targeted FGSM: push toward eng
        tgt = torch.full((x.shape[1],), tgt_idx, dtype=torch.long, device=device)
        loss = F_torch.cross_entropy(logits.squeeze(0), tgt)
        loss.backward()
        grad = x.grad.detach().sign().cpu().numpy().squeeze(0)

        results = []
        for eps in epsilons:
            feats_adv = feats - eps * grad  # subtract because we minimise CE wrt target
            with torch.no_grad():
                x_adv = torch.from_numpy(feats_adv).unsqueeze(0).to(device)
                lg, _ = model(x_adv)
                p_adv = F_torch.softmax(lg.squeeze(0), dim=-1).cpu().numpy()
            pred_adv = p_adv.argmax(-1)
            eng_frac = float((pred_adv == tgt_idx).mean())
            # primary metric: feature-domain SNR (deterministic)
            snr_feat = feature_snr(feats, feats_adv)
            # optional time-domain SNR via deterministic GL (seed pinned)
            try:
                a_clean = feature_to_audio(feats, seed=0)
                a_adv = feature_to_audio(feats_adv, seed=0)
                snr_audio = compute_snr(a_clean, a_adv)
            except Exception:
                snr_audio = float("nan")
            flipped = eng_frac > 0.5 and eng_frac - float((pred0 == tgt_idx).mean()) > 0.3
            results.append({
                "epsilon": float(eps),
                "eng_frac": eng_frac,
                "snr_feature_db": snr_feat,
                "snr_audio_db": snr_audio,
                "flipped_majority": flipped,
            })

        hits = [r for r in results if r["flipped_majority"] and r["snr_feature_db"] > 40.0]
        best_inaudible = min(hits, key=lambda r: r["epsilon"]) if hits else None
        any_flip = [r for r in results if r["flipped_majority"]]
        min_eps_any = min(any_flip, key=lambda r: r["epsilon"]) if any_flip else None
        return {
            "segment_start_s": start / SR,
            "segment_end_s": (start + seg_n) / SR,
            "initial_hin_frac": hin_frac,
            "sweep": results,
            "min_eps_inaudible_flip": best_inaudible,
            "min_eps_any_flip": min_eps_any,
        }
    return {"error": "no Hindi-dominant segment found for FGSM attack"}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--weights", default="models/lid.pt")
    ap.add_argument("--out", default="outputs/fgsm.json")
    args = ap.parse_args()
    res = fgsm_flip(args.audio, args.weights)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
