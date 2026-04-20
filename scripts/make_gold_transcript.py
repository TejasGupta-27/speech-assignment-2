"""Create a 'gold' transcript using Whisper-large-v3 for WER reference.

This is a proxy: we use the largest, best-available Whisper model with no LM
biasing to produce a reference transcript. Our Part-I system (whisper-medium +
n-gram LM rescoring) is then evaluated against this reference.

This is a common practical setup when no hand-labelled gold transcript is
available; it measures improvement from our *custom decoding*, not absolute
accuracy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--out_json", default="outputs/gold_transcript.json")
    ap.add_argument("--out_txt", default="outputs/gold_transcript.txt")
    ap.add_argument("--model", default="large-v3")
    ap.add_argument("--language", default="hi")
    ap.add_argument("--auto_detect", action="store_true")
    args = ap.parse_args()

    from faster_whisper import WhisperModel

    model = WhisperModel(args.model, device="cpu", compute_type="int8")
    segs, info = model.transcribe(
        args.audio,
        language=None if args.auto_detect else args.language,
        beam_size=5,
        temperature=0.0,
        word_timestamps=True,
        vad_filter=True,
        condition_on_previous_text=True,
    )
    out_segs = []
    for s in segs:
        out_segs.append(
            {
                "start": float(s.start),
                "end": float(s.end),
                "text": s.text.strip(),
                "words": [{"w": w.word, "s": float(w.start), "e": float(w.end)} for w in (s.words or [])],
            }
        )
    full = " ".join(s["text"] for s in out_segs)
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps({"language": info.language, "segments": out_segs}, indent=2, ensure_ascii=False))
    Path(args.out_txt).write_text(full)
    print(f"[gold] language={info.language}  {len(out_segs)} segments  → {args.out_txt}")


if __name__ == "__main__":
    main()
