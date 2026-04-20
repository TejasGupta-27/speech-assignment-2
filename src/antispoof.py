"""Task 4.1 — Anti-Spoofing Countermeasure using LFCC (bona-fide vs spoof).

Feature extraction
------------------
LFCC = DCT( log( LinearFilterBank( |STFT|^2 ) ) ).
A linear-spaced triangular filter bank (not mel) captures both low and high
frequency artefacts that TTS vocoders commonly leave behind. We stack with
Δ and ΔΔ for temporal context.

Model
-----
A lightweight 1-D CNN (3 conv blocks + GAP + Linear) predicts bona-fide vs spoof.
We train on a small set of audio clips (bona fide = user voice / reference segments,
spoof = TTS output), and compute EER on a held-out split.

The `eer` utility is hand-implemented: sweep thresholds over sorted scores.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SR = 16000


def linear_filterbank(n_filters: int, n_fft: int, sr: int) -> np.ndarray:
    fmin, fmax = 0.0, sr / 2
    edges = np.linspace(fmin, fmax, n_filters + 2)
    bins = np.floor((n_fft + 1) * edges / sr).astype(int)
    fb = np.zeros((n_filters, n_fft // 2 + 1), dtype=np.float32)
    for k in range(n_filters):
        l, c, r = bins[k], bins[k + 1], bins[k + 2]
        if c == l:
            c = l + 1
        if r == c:
            r = c + 1
        for i in range(l, c):
            fb[k, i] = (i - l) / max(c - l, 1)
        for i in range(c, r):
            fb[k, i] = (r - i) / max(r - c, 1)
    return fb


def extract_lfcc(y: np.ndarray, sr: int = SR, n_fft: int = 512, hop: int = 160, n_filters: int = 40, n_ceps: int = 20) -> np.ndarray:
    import librosa
    from scipy.fftpack import dct

    S = librosa.stft(y, n_fft=n_fft, hop_length=hop, window="hann", center=True)
    pwr = (np.abs(S) ** 2).astype(np.float32)
    fb = linear_filterbank(n_filters, n_fft, sr)
    en = np.maximum(fb @ pwr, 1e-10)
    log_en = np.log(en)
    lfcc = dct(log_en, type=2, axis=0, norm="ortho")[:n_ceps]
    d1 = librosa.feature.delta(lfcc)
    d2 = librosa.feature.delta(lfcc, order=2)
    feats = np.concatenate([lfcc, d1, d2], axis=0).astype(np.float32)  # (3*n_ceps, T)
    mu, sigma = feats.mean(1, keepdims=True), feats.std(1, keepdims=True) + 1e-6
    return (feats - mu) / sigma


def extract_cqcc(y: np.ndarray, sr: int = SR, n_ceps: int = 20) -> np.ndarray:
    """CQCC via librosa CQT (approximation; full CQCC uses resampling)."""
    import librosa
    from scipy.fftpack import dct

    C = np.abs(librosa.cqt(y, sr=sr, hop_length=256, n_bins=84, bins_per_octave=12))
    log_c = np.log(np.maximum(C, 1e-10))
    cqcc = dct(log_c, type=2, axis=0, norm="ortho")[:n_ceps]
    d1 = librosa.feature.delta(cqcc)
    d2 = librosa.feature.delta(cqcc, order=2)
    feats = np.concatenate([cqcc, d1, d2], axis=0).astype(np.float32)
    mu, sigma = feats.mean(1, keepdims=True), feats.std(1, keepdims=True) + 1e-6
    return (feats - mu) / sigma


class CMNet(nn.Module):
    """Lightweight 1-D CNN anti-spoof classifier."""

    def __init__(self, in_dim: int = 60, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_dim, hidden, 5, padding=2),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, 5, padding=2),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.MaxPool1d(3),
            nn.Conv1d(hidden, 2 * hidden, 5, padding=2),
            nn.BatchNorm1d(2 * hidden),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(2 * hidden, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x).squeeze(-1)
        return self.fc(h)


def random_crop(feat: np.ndarray, T: int = 400) -> np.ndarray:
    if feat.shape[1] <= T:
        pad = np.zeros((feat.shape[0], T - feat.shape[1]), dtype=feat.dtype)
        return np.concatenate([feat, pad], axis=1)
    s = np.random.randint(0, feat.shape[1] - T + 1)
    return feat[:, s : s + T]


def compute_eer(scores: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    """EER where higher score = bona fide. Returns (eer, threshold)."""
    order = np.argsort(scores)[::-1]
    scores, labels = scores[order], labels[order]
    pos = labels == 1
    neg = labels == 0
    n_pos, n_neg = pos.sum(), neg.sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan"), float("nan")
    tpr_c = np.cumsum(pos) / n_pos  # recall-positive at each threshold
    fpr_c = np.cumsum(neg) / n_neg
    fnr = 1 - tpr_c
    fpr = fpr_c
    diff = np.abs(fnr - fpr)
    i = int(np.argmin(diff))
    return float((fnr[i] + fpr[i]) / 2), float(scores[i])


def iter_clips(wav_paths: List[Tuple[str, int]], clip_s: float = 4.0, hop_s: float = 2.0, feature: str = "lfcc"):
    import librosa

    for p, lbl in wav_paths:
        y, sr = librosa.load(p, sr=SR, mono=True)
        n = int(clip_s * sr)
        h = int(hop_s * sr)
        if len(y) < n:
            y = np.pad(y, (0, n - len(y)))
        for s in range(0, len(y) - n + 1, h):
            seg = y[s : s + n]
            feat = extract_lfcc(seg, sr) if feature == "lfcc" else extract_cqcc(seg, sr)
            yield feat, lbl, p, s / sr


def train_and_eval(
    bona_paths: List[str],
    spoof_paths: List[str],
    out_weights: str = "models/cm.pt",
    feature: str = "lfcc",
    epochs: int = 25,
    lr: float = 1e-3,
    device: str = "cpu",
) -> dict:
    np.random.seed(0)
    torch.manual_seed(0)

    all_paths = [(p, 1) for p in bona_paths] + [(p, 0) for p in spoof_paths]
    print(f"[cm] files: bona={len(bona_paths)}  spoof={len(spoof_paths)}")

    # extract all clips, then split at clip level (data is limited — 2–4 files total)
    all_items = list(iter_clips(all_paths, feature=feature))
    np.random.shuffle(all_items)
    split = max(1, int(0.7 * len(all_items)))
    train_items, test_items = all_items[:split], all_items[split:]
    print(f"[cm] clips: train={len(train_items)}  test={len(test_items)}")
    if not train_items:
        raise RuntimeError("No training clips extracted; check that audio files exist and are long enough.")

    in_dim = train_items[0][0].shape[0]
    model = CMNet(in_dim=in_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # balance classes
    pos = [x for x in train_items if x[1] == 1]
    neg = [x for x in train_items if x[1] == 0]
    k = min(len(pos), len(neg))
    print(f"[cm] train clips: bona={len(pos)}  spoof={len(neg)}  → sampling {k} each / epoch")

    model.train()
    for ep in range(epochs):
        np.random.shuffle(pos)
        np.random.shuffle(neg)
        batch = pos[:k] + neg[:k]
        np.random.shuffle(batch)
        total = 0.0
        bs = 8
        for i in range(0, len(batch), bs):
            chunk = batch[i : i + bs]
            X = np.stack([random_crop(x[0], 400) for x in chunk])
            yl = np.array([x[1] for x in chunk])
            xt = torch.from_numpy(X).float().to(device)
            yt = torch.from_numpy(yl).long().to(device)
            logits = model(xt)
            loss = F.cross_entropy(logits, yt)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item())
        if (ep + 1) % 5 == 0:
            print(f"[cm] ep {ep + 1}/{epochs}  loss={total / max(1, len(batch) // bs):.4f}")

    model.eval()
    # eval EER on test clips (score = P(bona fide))
    if not test_items:
        test_items = train_items[-16:]
    with torch.no_grad():
        X = np.stack([random_crop(x[0], 400) for x in test_items])
        yl = np.array([x[1] for x in test_items])
        xt = torch.from_numpy(X).float().to(device)
        probs = F.softmax(model(xt), dim=-1)[:, 1].cpu().numpy()
    eer, thr = compute_eer(probs, yl)
    Path(out_weights).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": model.state_dict(), "in_dim": in_dim, "feature": feature}, out_weights)
    print(f"[cm] test EER = {eer:.4f} at thr={thr:.3f}")
    return {"eer": float(eer), "threshold": float(thr), "weights": out_weights}


@torch.no_grad()
def score_file(wav_path: str, weights_path: str, device: str = "cpu") -> dict:
    import librosa

    ckpt = torch.load(weights_path, map_location=device)
    feature = ckpt.get("feature", "lfcc")
    in_dim = ckpt.get("in_dim", 60)
    model = CMNet(in_dim=in_dim).to(device)
    model.load_state_dict(ckpt["state"])
    model.eval()
    y, sr = librosa.load(wav_path, sr=SR, mono=True)
    clips = []
    clip_n = int(4 * sr)
    for s in range(0, max(1, len(y) - clip_n + 1), int(2 * sr)):
        seg = y[s : s + clip_n]
        if len(seg) < clip_n:
            seg = np.pad(seg, (0, clip_n - len(seg)))
        feat = extract_lfcc(seg, sr) if feature == "lfcc" else extract_cqcc(seg, sr)
        clips.append(random_crop(feat, 400))
    X = np.stack(clips)
    xt = torch.from_numpy(X).float().to(device)
    probs = F.softmax(model(xt), dim=-1)[:, 1].cpu().numpy()
    return {"file": wav_path, "bona_fide_prob_mean": float(probs.mean()), "clip_scores": probs.tolist()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "score"])
    ap.add_argument("--bona", nargs="*", default=[])
    ap.add_argument("--spoof", nargs="*", default=[])
    ap.add_argument("--feature", choices=["lfcc", "cqcc"], default="lfcc")
    ap.add_argument("--weights", default="models/cm.pt")
    ap.add_argument("--wav")
    ap.add_argument("--out", default="outputs/cm_score.json")
    ap.add_argument("--epochs", type=int, default=25)
    args = ap.parse_args()
    if args.cmd == "train":
        m = train_and_eval(args.bona, args.spoof, args.weights, feature=args.feature, epochs=args.epochs)
        Path("outputs").mkdir(exist_ok=True)
        Path("outputs/cm_eer.json").write_text(json.dumps(m, indent=2))
    elif args.cmd == "score":
        out = score_file(args.wav, args.weights)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=2))
        print(json.dumps(out, indent=2))
