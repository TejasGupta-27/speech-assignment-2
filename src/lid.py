"""Task 1.1 — Multi-Head Frame-Level Language Identification (English vs Hindi).

Architecture
------------
- Input: 80-d log-mel + 13-d MFCC + delta/delta-delta → 40-d projected via linear.
- Encoder: 2-layer Bi-LSTM (hidden=128) for frame context.
- Two heads (multi-head):
    (a) frame-level language logits (3 classes: eng, hin, sil/noise),
    (b) boundary regression head predicting P(switch-at-frame) for 200ms timestamping.
- Trained jointly with cross-entropy (frames) + BCE (boundary).

Frame-rate: 100 fps (10ms hop) on 16kHz audio. Time resolution 10ms ≪ 200ms target.
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

FRAME_HOP = 0.010  # 10ms
SR = 16000
LABEL_MAP = {"eng": 0, "hin": 1, "sil": 2}
INV_LABEL = {v: k for k, v in LABEL_MAP.items()}


def extract_features(y: np.ndarray, sr: int = SR) -> np.ndarray:
    """80 log-mel + 13 MFCC + Δ + ΔΔ → (T, F)."""
    import librosa

    n_fft, hop = 512, int(FRAME_HOP * sr)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop, n_mels=80)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    mfcc = librosa.feature.mfcc(S=log_mel, n_mfcc=13)
    d1 = librosa.feature.delta(mfcc)
    d2 = librosa.feature.delta(mfcc, order=2)
    feats = np.concatenate([log_mel, mfcc, d1, d2], axis=0).T  # (T, 80+13*3=119)
    mu, sigma = feats.mean(0, keepdims=True), feats.std(0, keepdims=True) + 1e-6
    return ((feats - mu) / sigma).astype(np.float32)


class LIDModel(nn.Module):
    def __init__(self, in_dim: int = 119, hidden: int = 128, num_classes: int = 3):
        super().__init__()
        self.proj = nn.Linear(in_dim, 64)
        self.lstm = nn.LSTM(64, hidden, num_layers=2, batch_first=True, bidirectional=True, dropout=0.2)
        self.cls_head = nn.Linear(2 * hidden, num_classes)
        self.boundary_head = nn.Linear(2 * hidden, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = torch.relu(self.proj(x))
        h, _ = self.lstm(h)
        return self.cls_head(h), self.boundary_head(h).squeeze(-1)


def label_sequence_from_whisper(audio_path: str, win_s: float = 8.0, hop_s: float = 4.0, model_size: str = "base") -> np.ndarray:
    """Weakly label frames by running Whisper language detection on sliding windows.

    Uses `detect_language` (a single forward pass per window, no decoder).
    Returns per-frame label array at 100 fps.
    """
    import librosa
    from faster_whisper import WhisperModel

    y, sr = librosa.load(audio_path, sr=SR, mono=True)
    n_frames = int(np.ceil(len(y) / (FRAME_HOP * sr)))
    labels = np.full(n_frames, LABEL_MAP["sil"], dtype=np.int64)
    conf = np.zeros(n_frames, dtype=np.float32)
    model = WhisperModel(model_size, device="cpu", compute_type="int8")

    step = int(hop_s * sr)
    win = int(win_s * sr)
    total = int(np.ceil((len(y) - win) / step)) + 1
    print(f"[lid] weak-labelling: {total} windows ({win_s}s, hop {hop_s}s, model={model_size})", flush=True)

    for idx, start in enumerate(range(0, len(y), step)):
        seg = y[start : start + win]
        if len(seg) < sr * 1.0:
            break
        rms = float(np.sqrt(np.mean(seg**2) + 1e-12))
        s_frame = int(start / sr / FRAME_HOP)
        e_frame = int(min(start + win, len(y)) / sr / FRAME_HOP)
        if rms < 0.005:
            # preserve previous non-silent labels if any; just mark as sil where unset
            mask = labels[s_frame:e_frame] == LABEL_MAP["sil"]
            labels[s_frame:e_frame][mask] = LABEL_MAP["sil"]
            continue
        try:
            lang, lp, _all = model.detect_language(audio=seg)
        except Exception:
            try:
                _, info = model.transcribe(seg, language=None, beam_size=1, without_timestamps=True, vad_filter=False)
                lang = info.language
                lp = info.language_probability or 0.0
            except Exception:
                lang, lp = "en", 0.0
        if lp < 0.3:
            continue
        # use weighted vote: frames in overlap region take higher-confidence label
        new_label = LABEL_MAP["eng"] if lang == "en" else (LABEL_MAP["hin"] if lang in ("hi", "ur") else None)
        if new_label is None:
            continue
        for f in range(s_frame, e_frame):
            if lp > conf[f]:
                labels[f] = new_label
                conf[f] = lp
        if idx % 20 == 0:
            print(f"[lid]   win {idx + 1}/{total}  lang={lang} conf={lp:.2f}", flush=True)
    return labels


def boundaries_from_labels(labels: np.ndarray, width: int = 5) -> np.ndarray:
    """Binary boundary target; smoothed ±width frames."""
    change = np.zeros_like(labels, dtype=np.float32)
    diff = np.abs(np.diff(labels)) > 0
    idx = np.where(diff)[0] + 1
    for i in idx:
        lo, hi = max(0, i - width), min(len(change), i + width + 1)
        change[lo:hi] = 1.0
    return change


def chunk_iter(feats: np.ndarray, labels: np.ndarray, bound: np.ndarray, T: int = 500):
    N = feats.shape[0]
    for start in range(0, N, T):
        yield feats[start : start + T], labels[start : start + T], bound[start : start + T]


def train(
    audio_paths: List[str],
    out_weights: str,
    epochs: int = 8,
    lr: float = 1e-3,
    device: str = "cpu",
    label_paths: List[str] | None = None,
) -> None:
    """Train on a list of audio files.

    If `label_paths` is supplied (one .npy per audio), those are taken as GOLD
    frame-level labels. Otherwise labels are derived weakly from Whisper.
    """
    import librosa

    model = LIDModel().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    dataset = []
    for idx, ap in enumerate(audio_paths):
        lp = label_paths[idx] if label_paths and idx < len(label_paths) else None
        if lp and Path(lp).exists():
            print(f"[lid] loading gold labels from {lp}")
            lbls = np.load(lp).astype(np.int64)
        else:
            print(f"[lid] generating weak labels via Whisper for {ap} ...")
            lbls = label_sequence_from_whisper(ap)
        y, _ = librosa.load(ap, sr=SR, mono=True)
        feats = extract_features(y)
        n = min(len(feats), len(lbls))
        feats, lbls = feats[:n], lbls[:n]
        bnd = boundaries_from_labels(lbls)
        dataset.append((feats, lbls, bnd))
        print(f"  - {ap}: frames={n}  eng={int((lbls == 0).sum())}  hin={int((lbls == 1).sum())}  sil={int((lbls == 2).sum())}")

    # class weights invert freq to help rare languages
    all_lbls = np.concatenate([d[1] for d in dataset])
    freq = np.bincount(all_lbls, minlength=3).astype(np.float32)
    w = (freq.sum() / (3 * np.maximum(freq, 1))).astype(np.float32)
    w_t = torch.tensor(w, device=device)
    print(f"[lid] class weights = {w}")

    model.train()
    for ep in range(epochs):
        total, n = 0.0, 0
        for feats, lbls, bnd in dataset:
            for fc, lc, bc in chunk_iter(feats, lbls, bnd, T=500):
                xf = torch.from_numpy(fc).unsqueeze(0).to(device)
                xl = torch.from_numpy(lc).unsqueeze(0).to(device)
                xb = torch.from_numpy(bc).unsqueeze(0).to(device)
                logits, bout = model(xf)
                l_cls = F.cross_entropy(logits.squeeze(0), xl.squeeze(0), weight=w_t)
                l_bnd = F.binary_cross_entropy_with_logits(bout.squeeze(0), xb.squeeze(0))
                loss = l_cls + 0.3 * l_bnd
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += float(loss.item())
                n += 1
        print(f"[lid] epoch {ep + 1}/{epochs}  loss={total / max(n, 1):.4f}")
    Path(out_weights).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_weights)
    print(f"[lid] saved {out_weights}")


@torch.no_grad()
def predict(audio_path: str, weights_path: str, device: str = "cpu") -> dict:
    """Return per-frame labels, boundary probs, and timestamped language segments."""
    import librosa

    model = LIDModel().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    y, _ = librosa.load(audio_path, sr=SR, mono=True)
    feats = extract_features(y)
    x = torch.from_numpy(feats).unsqueeze(0).to(device)
    logits, bout = model(x)
    probs = F.softmax(logits.squeeze(0), dim=-1).cpu().numpy()
    preds = probs.argmax(-1)
    bprob = torch.sigmoid(bout.squeeze(0)).cpu().numpy()

    segs = []
    start = 0
    for i in range(1, len(preds)):
        if preds[i] != preds[start]:
            segs.append(
                {
                    "start": round(start * FRAME_HOP, 3),
                    "end": round(i * FRAME_HOP, 3),
                    "label": INV_LABEL[int(preds[start])],
                }
            )
            start = i
    segs.append(
        {
            "start": round(start * FRAME_HOP, 3),
            "end": round(len(preds) * FRAME_HOP, 3),
            "label": INV_LABEL[int(preds[-1])],
        }
    )
    return {"frames": preds.tolist(), "boundary": bprob.tolist(), "segments": segs}


def evaluate_f1(audio_path: str, weights_path: str, gold_path: str | None = None) -> dict:
    """If gold label .npy exists (same 100fps layout) compute per-class F1 & macro-F1.

    If no gold is provided, we treat Whisper-derived labels as silver and report vs those.
    """
    from sklearn.metrics import classification_report, f1_score

    out = predict(audio_path, weights_path)
    pred = np.array(out["frames"])
    if gold_path and Path(gold_path).exists():
        gold = np.load(gold_path)
    else:
        gold = label_sequence_from_whisper(audio_path)
    n = min(len(pred), len(gold))
    pred, gold = pred[:n], gold[:n]
    report = classification_report(
        gold, pred, labels=[0, 1, 2], target_names=["eng", "hin", "sil"], output_dict=True, zero_division=0
    )
    macro = f1_score(gold, pred, labels=[0, 1], average="macro", zero_division=0)  # eng+hin macro
    return {"macro_f1_eng_hin": float(macro), "report": report}


def boundary_timing_accuracy(audio_path: str, weights_path: str, gold_path: str | None, tol_ms: int = 200) -> dict:
    """Fraction of gold language-switch boundaries within `tol_ms` of a predicted switch."""
    out = predict(audio_path, weights_path)
    pred = np.array(out["frames"])
    if gold_path and Path(gold_path).exists():
        gold = np.load(gold_path)
    else:
        gold = label_sequence_from_whisper(audio_path)
    n = min(len(pred), len(gold))
    pred, gold = pred[:n], gold[:n]

    def changes(arr):
        d = np.abs(np.diff(arr)) > 0
        return np.where(d)[0] + 1

    g_idx, p_idx = changes(gold), changes(pred)
    tol_frames = int(tol_ms / 1000 / FRAME_HOP)
    if len(g_idx) == 0:
        return {"n_gold": 0, "hit_rate": 1.0, "tol_ms": tol_ms}
    hits = 0
    for gi in g_idx:
        if p_idx.size and np.min(np.abs(p_idx - gi)) <= tol_frames:
            hits += 1
    return {"n_gold": int(len(g_idx)), "hit_rate": float(hits / len(g_idx)), "tol_ms": tol_ms}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "predict", "eval"])
    ap.add_argument("--audio", nargs="+")
    ap.add_argument("--labels", nargs="*", default=None, help="gold label .npy files parallel to --audio")
    ap.add_argument("--weights", default="models/lid.pt")
    ap.add_argument("--out", default="outputs/lid_pred.json")
    ap.add_argument("--gold", default=None)
    ap.add_argument("--epochs", type=int, default=8)
    args = ap.parse_args()
    if args.cmd == "train":
        train(args.audio, args.weights, epochs=args.epochs, label_paths=args.labels)
    elif args.cmd == "predict":
        out = predict(args.audio[0], args.weights)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[lid] wrote {args.out}  ({len(out['segments'])} segments)")
    elif args.cmd == "eval":
        m = evaluate_f1(args.audio[0], args.weights, args.gold)
        t = boundary_timing_accuracy(args.audio[0], args.weights, args.gold, tol_ms=200)
        print(json.dumps({"lid": m, "timing": t}, indent=2))
