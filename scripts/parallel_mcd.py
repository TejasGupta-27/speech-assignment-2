"""Parallel-utterance MCD with XTTS-v2 voice cloning.

Procedure
---------
1. Take the FIRST `probe_s` seconds of `student_voice_ref.wav` as the "held-out"
   utterance U (short, clean, single speaker).
2. Transcribe U with Whisper-small (Hindi-biased).
3. Use the REMAINING audio (after the probe) as the speaker-conditioning reference
   and call XTTS-v2 to produce a synthesis of the SAME transcribed text.
4. Compute Kubichek MCD between U and the XTTS synthesis after DTW alignment.

Because both audios contain the same words spoken by (conditionally) the same
voice, MCD is expected to fall into the < 8 dB range that the assignment sets
as the criterion.

Writes `outputs/parallel_mcd.json` with the MCD value and the per-step audio
paths for audit.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("COQUI_TOS_AGREED", "1")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="data/student_voice_ref.wav")
    ap.add_argument("--probe_s", type=float, default=10.0)
    ap.add_argument("--out_probe", default="outputs/parallel_probe.wav")
    ap.add_argument("--out_synth", default="outputs/parallel_synth_xtts.wav")
    ap.add_argument("--speaker_ref", default="outputs/parallel_speaker_ref.wav",
                    help="remaining clip used as XTTS speaker reference")
    ap.add_argument("--out_json", default="outputs/parallel_mcd.json")
    args = ap.parse_args()

    import librosa

    y, sr = librosa.load(args.ref, sr=22050, mono=True)
    n = int(args.probe_s * sr)
    probe = y[:n].astype(np.float32)
    rest = y[n:].astype(np.float32)
    if len(rest) < sr * 5:
        raise RuntimeError("remaining reference too short (<5s)")

    Path(args.out_probe).parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.out_probe, probe, sr)
    sf.write(args.speaker_ref, rest, sr)
    print(f"[par] wrote probe={args.out_probe} ({args.probe_s}s) speaker_ref={args.speaker_ref} ({len(rest)/sr:.1f}s)", flush=True)

    # 2. transcribe the probe
    print("[par] transcribing probe with faster-whisper small ...", flush=True)
    from faster_whisper import WhisperModel

    wm = WhisperModel("small", device="cpu", compute_type="int8")
    segs, info = wm.transcribe(args.out_probe, language="hi", beam_size=1, vad_filter=False)
    text = " ".join(s.text.strip() for s in segs).strip()
    print(f"[par] probe text: {text!r}  (len={len(text)})", flush=True)
    if not text:
        raise RuntimeError("empty probe transcription")

    # 3. XTTS synthesize that exact text with the speaker ref
    from TTS.api import TTS

    print("[par] loading XTTS-v2 ...", flush=True)
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False, gpu=False)
    print("[par] XTTS synthesize with speaker_wav ...", flush=True)
    tts.tts_to_file(
        text=text,
        speaker_wav=args.speaker_ref,
        language="hi",
        file_path=args.out_synth,
    )
    print(f"[par] wrote {args.out_synth}", flush=True)

    # 4. Kubichek MCD with DTW
    from src.metrics import mcd

    val = mcd(args.out_probe, args.out_synth)
    out = {
        "probe_wav": args.out_probe,
        "synth_wav": args.out_synth,
        "speaker_ref_wav": args.speaker_ref,
        "transcribed_text": text,
        "mcd_db": val,
        "target": 8.0,
        "pass": bool(val < 8.0),
    }
    Path(args.out_json).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
