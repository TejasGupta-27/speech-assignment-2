"""Task 3.1 — Speaker Embedding (d-vector) from 60s reference.

We extract a fixed-length speaker embedding from `student_voice_ref.wav`.
Two backends, auto-selected:
  (A) SpeechBrain ECAPA-TDNN (speechbrain/spkrec-ecapa-voxceleb) → 192-d x-vector
  (B) Fallback: self-contained d-vector from log-mel + stats-pool + small MLP
      (trainable later, but for embedding we just use mean+std of log-mel which is
       surprisingly discriminative when cosine-compared against the same speaker).

The output is saved as a .npy vector and consumed by tts.py.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

SR = 16000


def extract_logmel(y: np.ndarray, sr: int = SR, n_mels: int = 80) -> np.ndarray:
    import librosa

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=512, hop_length=160, n_mels=n_mels)
    return librosa.power_to_db(mel, ref=np.max)


def stats_pool_dvector(y: np.ndarray, sr: int = SR) -> np.ndarray:
    m = extract_logmel(y, sr)
    mu, sigma = m.mean(axis=1), m.std(axis=1)
    v = np.concatenate([mu, sigma]).astype(np.float32)
    v /= np.linalg.norm(v) + 1e-9
    return v


def ecapa_xvector(wav_path: str) -> np.ndarray:
    try:
        import torch
        from speechbrain.inference.speaker import EncoderClassifier

        cls = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="models/spkrec_ecapa",
            run_opts={"device": "cpu"},
        )
        sig = cls.load_audio(wav_path)
        emb = cls.encode_batch(sig.unsqueeze(0)).squeeze(0).squeeze(0).cpu().numpy()
        emb /= np.linalg.norm(emb) + 1e-9
        return emb.astype(np.float32)
    except Exception as e:
        print(f"[embed] ECAPA backend unavailable: {e}; using stats-pool fallback.")
        import librosa

        y, sr = librosa.load(wav_path, sr=SR, mono=True)
        return stats_pool_dvector(y, sr)


def run(wav_path: str, out_path: str, backend: str = "ecapa") -> np.ndarray:
    emb = ecapa_xvector(wav_path) if backend == "ecapa" else None
    if emb is None:
        import librosa

        y, sr = librosa.load(wav_path, sr=SR, mono=True)
        emb = stats_pool_dvector(y, sr)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, emb)
    print(f"[embed] saved {out_path}  shape={emb.shape}  ‖v‖={np.linalg.norm(emb):.3f}")
    return emb


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True)
    ap.add_argument("--out", default="models/speaker_embedding.npy")
    ap.add_argument("--backend", choices=["ecapa", "dvector"], default="ecapa")
    args = ap.parse_args()
    run(args.wav, args.out, args.backend)
