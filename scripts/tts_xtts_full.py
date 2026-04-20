"""Full-text XTTS-v2 synthesis of outputs/lrl.txt, conditioned on
`data/student_voice_ref.wav`. Chunks the text on Devanagari danda boundaries
(|) with a 220-char soft cap, which keeps XTTS within its context budget.

Writes `outputs/output_LRL_xtts.wav` at 22.05 kHz.
"""
from __future__ import annotations

import argparse
import os
import re
import time
from pathlib import Path

import numpy as np
import soundfile as sf

os.environ.setdefault("COQUI_TOS_AGREED", "1")


def chunk_text(text: str, max_chars: int = 220) -> list[str]:
    # split on Devanagari danda (।), full stop, ? or !
    out: list[str] = []
    buf = ""
    for s in re.split(r"([।!?.]+)", text):
        if not s:
            continue
        buf += s
        if len(buf) >= max_chars and re.search(r"[।!?.]$", buf.strip()):
            out.append(buf.strip())
            buf = ""
    if buf.strip():
        out.append(buf.strip())
    # further split any remaining giant chunks by whitespace groups
    final: list[str] = []
    for c in out:
        while len(c) > max_chars:
            # find last whitespace under the limit
            cut = c.rfind(" ", 0, max_chars)
            if cut <= 0:
                cut = max_chars
            final.append(c[:cut].strip())
            c = c[cut:].strip()
        if c:
            final.append(c)
    return [c for c in final if c]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text_file", default="outputs/lrl.txt")
    ap.add_argument("--speaker_wav", default="data/student_voice_ref.wav")
    ap.add_argument("--out", default="outputs/output_LRL_xtts.wav")
    ap.add_argument("--language", default="hi")
    ap.add_argument("--max_chars", type=int, default=220)
    ap.add_argument("--speed", type=float, default=1.0, help="XTTS playback speed (<1 slower, >1 faster)")
    args = ap.parse_args()

    text = Path(args.text_file).read_text(encoding="utf-8").strip()
    chunks = chunk_text(text, max_chars=args.max_chars)
    print(f"[xtts-full] {len(chunks)} chunks, max_chars={args.max_chars}", flush=True)

    import librosa
    from TTS.api import TTS

    t0 = time.time()
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False, gpu=False)
    print(f"[xtts-full] model loaded in {time.time() - t0:.1f}s", flush=True)

    OUT_SR = 22050
    pieces: list[np.ndarray] = []
    Path("outputs").mkdir(exist_ok=True)
    for i, c in enumerate(chunks):
        tmp = f"outputs/_xtts_chunk_{i:04d}.wav"
        t1 = time.time()
        try:
            tts.tts_to_file(
                text=c,
                speaker_wav=args.speaker_wav,
                language=args.language,
                file_path=tmp,
                speed=args.speed,
            )
        except Exception as e:
            print(f"[xtts-full] chunk {i} failed: {e}", flush=True)
            continue
        y, _ = librosa.load(tmp, sr=OUT_SR, mono=True)
        pieces.append(y)
        pieces.append(np.zeros(int(0.25 * OUT_SR), dtype=np.float32))
        print(f"[xtts-full] chunk {i + 1}/{len(chunks)}  dur={len(y) / OUT_SR:.1f}s  elapsed={time.time() - t1:.1f}s", flush=True)

    if not pieces:
        raise RuntimeError("no XTTS chunks succeeded")
    full = np.concatenate(pieces).astype(np.float32)
    peak = np.max(np.abs(full)) + 1e-9
    if peak > 0.99:
        full = full / peak * 0.95
    sf.write(args.out, full, OUT_SR)
    print(f"[xtts-full] wrote {args.out}  {len(full) / OUT_SR:.1f}s  total {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
