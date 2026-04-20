"""Build a synthetic code-switched Eng+Hin training/eval set with gold labels.

Strategy
--------
- Take the English 10-min lecture and split into ~30 chunks of variable length.
- Generate ~30 Hindi MMS-TTS utterances from a list of syllabus-like sentences.
- Interleave chunks (eng, hin, eng, hin, …) with short silence gaps.
- Write:
    data/codeswitched_train.wav
    data/codeswitched_train_labels.npy      (frame-level ints, 100 fps, 0=eng 1=hin 2=sil)
    data/codeswitched_eval.wav              (held-out permutation)
    data/codeswitched_eval_labels.npy
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.lid import FRAME_HOP, LABEL_MAP, SR

HIN_SENTS = [
    "आज हम MFCC के बारे में बात करेंगे।",
    "यह एक stochastic model है।",
    "Cepstrum का मतलब होता है स्पेक्ट्रम का उलटा।",
    "Viterbi algorithm से हम best path निकालते हैं।",
    "Mel-spectrogram में frequency logarithmic scale पर होती है।",
    "Transformer architecture में self-attention होता है।",
    "हम Whisper model से transcription करते हैं।",
    "WER कम करना हमारा goal है।",
    "Dynamic time warping दो time series को align करता है।",
    "LFCC anti-spoofing में useful होता है।",
    "FGSM एक adversarial attack method है।",
    "Code-switching में matrix language और embedded language होते हैं।",
    "GMM में कई Gaussians का mixture होता है।",
    "Filterbank energies log scale पर compute होती हैं।",
    "TTS में हम text से audio generate करते हैं।",
    "F0 contour speaker की pitch दिखाता है।",
    "ASR system code-mixing में challenges face करते हैं।",
    "Mel-cepstral distortion से हम synthesis quality measure करते हैं।",
    "Bayesian inference में posterior probability important होती है।",
    "Cross-entropy loss classification में use होती है।",
    "नमस्ते, आप कैसे हैं? आज का lecture शुरू करते हैं।",
    "हमारा पहला topic है phoneme recognition।",
    "Speaker verification के लिए d-vector embedding use करते हैं।",
    "ECAPA TDNN एक अच्छा speaker encoder है।",
    "Hinglish भाषा में हम दोनों भाषाओं को mix करते हैं।",
    "Phonetics और phonology अलग subjects हैं।",
    "Fricatives जैसे /s/ और /f/ continuous sounds हैं।",
    "Nasals जैसे /m/ और /n/ nasal cavity use करते हैं।",
    "Retroflex consonants हिंदी में बहुत important हैं।",
    "Schwa-deletion Devanagari के G2P में critical है।",
]


def synth_hindi(sentences, out_sr=SR):
    from transformers import AutoTokenizer, VitsModel

    tk = AutoTokenizer.from_pretrained("facebook/mms-tts-hin")
    model = VitsModel.from_pretrained("facebook/mms-tts-hin").eval()
    out = []
    for s in sentences:
        inputs = tk(s, return_tensors="pt")
        with torch.no_grad():
            wav = model(**inputs).waveform.squeeze(0).cpu().numpy()
        wav = wav.astype(np.float32)
        peak = np.max(np.abs(wav)) + 1e-9
        wav = wav / peak * 0.85
        if model.config.sampling_rate != out_sr:
            import librosa

            wav = librosa.resample(wav, orig_sr=model.config.sampling_rate, target_sr=out_sr)
        out.append(wav)
    return out


def english_chunks(audio_path, n_chunks, min_s=3.0, max_s=8.0, out_sr=SR):
    import librosa

    y, sr = librosa.load(audio_path, sr=out_sr, mono=True)
    rng = np.random.default_rng(0)
    chunks = []
    for i in range(n_chunks):
        dur = rng.uniform(min_s, max_s)
        start = rng.uniform(0, max(1.0, len(y) / sr - dur - 1.0))
        s, e = int(start * sr), int((start + dur) * sr)
        chunks.append(y[s:e].astype(np.float32))
    return chunks


def interleave(eng, hin, sr=SR, gap_s=0.25, seed=0):
    rng = np.random.default_rng(seed)
    order = []
    eng_iter = list(eng)
    hin_iter = list(hin)
    # alternate roughly with a few doubles
    cur = "eng"
    while eng_iter or hin_iter:
        if cur == "eng" and eng_iter:
            order.append(("eng", eng_iter.pop(0)))
            cur = "hin" if rng.random() > 0.2 else "eng"
        elif cur == "hin" and hin_iter:
            order.append(("hin", hin_iter.pop(0)))
            cur = "eng" if rng.random() > 0.2 else "hin"
        else:
            cur = "hin" if eng_iter == [] else "eng"
    gap = np.zeros(int(gap_s * sr), dtype=np.float32)
    audio = []
    labels = []
    for tag, chunk in order:
        audio.append(chunk)
        lbl_id = LABEL_MAP[tag]
        labels.extend([lbl_id] * int(np.ceil(len(chunk) / (FRAME_HOP * sr))))
        audio.append(gap)
        labels.extend([LABEL_MAP["sil"]] * int(np.ceil(len(gap) / (FRAME_HOP * sr))))
    a = np.concatenate(audio).astype(np.float32)
    l = np.asarray(labels, dtype=np.int64)
    # reconcile lengths
    n_frames = int(np.ceil(len(a) / (FRAME_HOP * sr)))
    if len(l) > n_frames:
        l = l[:n_frames]
    elif len(l) < n_frames:
        l = np.pad(l, (0, n_frames - len(l)), constant_values=LABEL_MAP["sil"])
    return a, l


def main():
    Path("data").mkdir(exist_ok=True)
    print("[cs] synthesizing Hindi with MMS-TTS …", flush=True)
    hin_all = synth_hindi(HIN_SENTS)
    print(f"[cs] got {len(hin_all)} Hindi clips.")
    print("[cs] sampling English chunks from lecture …", flush=True)
    eng_all = english_chunks("data/original_denoised.wav", n_chunks=len(hin_all))

    # train / eval split
    n = len(hin_all)
    split = int(0.8 * n)

    a_tr, l_tr = interleave(eng_all[:split], hin_all[:split], seed=1)
    a_ev, l_ev = interleave(eng_all[split:], hin_all[split:], seed=2)

    sf.write("data/codeswitched_train.wav", a_tr, SR)
    np.save("data/codeswitched_train_labels.npy", l_tr)
    sf.write("data/codeswitched_eval.wav", a_ev, SR)
    np.save("data/codeswitched_eval_labels.npy", l_ev)
    print(f"[cs] wrote data/codeswitched_{{train,eval}}.wav + labels.  train={len(a_tr)/SR:.1f}s  eval={len(a_ev)/SR:.1f}s")
    print(f"[cs] train label dist: eng={int((l_tr==0).sum())} hin={int((l_tr==1).sum())} sil={int((l_tr==2).sum())}")
    print(f"[cs] eval  label dist: eng={int((l_ev==0).sum())} hin={int((l_ev==1).sum())} sil={int((l_ev==2).sum())}")


if __name__ == "__main__":
    main()
