"""Task 3.3 — Synthesize LRL lecture via zero-shot voice cloning.

Preferred: Coqui-TTS XTTS-v2 (`tts_models/multilingual/multi-dataset/xtts_v2`) which
supports Hindi out of the box and accepts a speaker reference wav for zero-shot
voice cloning. We feed it Maithili text written in Devanagari; phonetically close
to Hindi so XTTS produces acceptable output.

Fallback: Meta MMS-TTS (`facebook/mms-tts-hin`) which has Hindi support but is
NOT speaker-conditioned. In that case the speaker colour comes purely from the
prosody-warping step in prosody.py.

Output is written at 24 kHz (native XTTS) or 16 kHz (MMS); we resample to 22.05 kHz
as required by the spec.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf

OUT_SR = 22050


def synth_xtts(text: str, ref_wav: str, out_path: str, language: str = "hi") -> bool:
    try:
        from TTS.api import TTS

        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False, gpu=False)
        tmp = "outputs/_tmp_xtts.wav"
        Path(tmp).parent.mkdir(parents=True, exist_ok=True)
        tts.tts_to_file(text=text, speaker_wav=ref_wav, language=language, file_path=tmp)
        import librosa

        y, sr = librosa.load(tmp, sr=OUT_SR, mono=True)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        sf.write(out_path, y, OUT_SR)
        print(f"[tts] XTTS wrote {out_path}  {len(y) / OUT_SR:.1f}s")
        return True
    except Exception as e:
        print(f"[tts] XTTS unavailable: {e}")
        return False


_MMS_CACHE = {}


def _mms_load(lang: str):
    if lang in _MMS_CACHE:
        return _MMS_CACHE[lang]
    from transformers import AutoTokenizer, VitsModel

    name = f"facebook/mms-tts-{lang}"
    tk = AutoTokenizer.from_pretrained(name)
    m = VitsModel.from_pretrained(name).eval()
    _MMS_CACHE[lang] = (tk, m)
    return tk, m


def synth_mms(text: str, out_path: str, lang: str = "hin") -> bool:
    try:
        import torch

        tk, model = _mms_load(lang)
        text = (text or "").strip()
        if not text:
            return False
        inputs = tk(text, return_tensors="pt")
        input_ids = inputs.get("input_ids")
        if input_ids is None or input_ids.dtype.is_floating_point:
            input_ids = input_ids.long() if input_ids is not None else None
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None and attention_mask.dtype.is_floating_point:
            attention_mask = attention_mask.long()
        if input_ids is None or input_ids.numel() == 0:
            return False
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask).waveform.squeeze(0).cpu().numpy()
        sr = model.config.sampling_rate
        import librosa

        y = librosa.resample(out.astype(np.float32), orig_sr=sr, target_sr=OUT_SR)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        sf.write(out_path, y, OUT_SR)
        print(f"[tts] MMS-{lang} wrote {out_path}  {len(y) / OUT_SR:.1f}s", flush=True)
        return True
    except Exception as e:
        print(f"[tts] MMS chunk failed: {e}", flush=True)
        return False


def synth_long(text: str, ref_wav: str | None, out_path: str, language: str = "hi", lrl_fallback: str = "hin", max_chars: int = 240) -> None:
    """Chunk long text on sentence/punctuation boundaries, synthesize, and concat."""
    import re

    import librosa

    chunks: list[str] = []
    buf = ""
    for sent in re.split(r"([।!?.]+)", text):
        if not sent:
            continue
        buf += sent
        if len(buf) >= max_chars and re.search(r"[।!?.]$", buf):
            chunks.append(buf.strip())
            buf = ""
    if buf.strip():
        chunks.append(buf.strip())

    wavs: list[np.ndarray] = []
    skipped = 0
    for i, c in enumerate(chunks):
        tmp = f"outputs/_tts_chunk_{i:04d}.wav"
        ok = False
        if ref_wav is not None:
            ok = synth_xtts(c, ref_wav, tmp, language=language)
        if not ok:
            ok = synth_mms(c, tmp, lang=lrl_fallback)
        if not ok:
            print(f"[tts] skipping chunk {i} (synth failed)", flush=True)
            skipped += 1
            continue
        y, _ = librosa.load(tmp, sr=OUT_SR, mono=True)
        wavs.append(y)
        gap = np.zeros(int(0.25 * OUT_SR), dtype=np.float32)
        wavs.append(gap)
    if not wavs:
        raise RuntimeError("All TTS chunks failed; no audio produced.")
    full = np.concatenate(wavs).astype(np.float32)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_path, full, OUT_SR)
    print(f"[tts] synthesized {len(chunks) - skipped}/{len(chunks)} chunks → {out_path}  ({len(full) / OUT_SR:.1f}s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", help="text to synthesize (or --text_file)")
    ap.add_argument("--text_file")
    ap.add_argument("--ref", help="speaker reference wav for zero-shot cloning")
    ap.add_argument("--out", required=True)
    ap.add_argument("--language", default="hi", help="XTTS language code (hi for Hindi/Maithili)")
    ap.add_argument("--mms_lang", default="hin")
    args = ap.parse_args()
    text = args.text or Path(args.text_file).read_text(encoding="utf-8")
    synth_long(text, args.ref, args.out, language=args.language, lrl_fallback=args.mms_lang)
