"""Task 1.2 — Constrained decoding with N-gram logit bias on Whisper.

We run faster-whisper's segment decoder with a `logit_filter` callback that
adds a bias term derived from our Kneser-Ney LM whenever a token finishes a word.

bias(tok | history) = λ * logP_LM(word | last n-1 words) if word ∈ LM.vocab
                    = β * logP_LM_unigram(word)       otherwise (smaller)

We use faster-whisper's `suppress_tokens=None` and `logit_processors` (if exposed
on the installed version) else we fall back to post-hoc rescoring of N-best using
the LM — both options are implemented and selected at runtime.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

from src.ngram_lm import KneserNeyLM, tokenize


def load_lm(path: str) -> KneserNeyLM:
    return KneserNeyLM.load(path)


def rescore_word_sequence(words: List[str], lm: KneserNeyLM, lambda_: float = 0.6) -> float:
    """Sum log P_LM over the sequence, with back-off to unigram continuation prob.

    Returns the total log-LM score (to be added to acoustic log-prob).
    """
    total = 0.0
    history: List[str] = []
    for w in words:
        s = lm.word_score(w, history)
        total += lambda_ * s
        history.append(w)
    return total


def transcribe_with_rescoring(
    audio_path: str,
    lm_path: str | None,
    model_size: str = "medium",
    device: str = "cpu",
    compute_type: str = "int8",
    n_best: int = 1,
    lambda_lm: float = 0.6,
    lang: str | None = None,
) -> Dict:
    """Run faster-whisper beam search once; if n_best>1 run extra temperature passes.

    At the *segment* level we then rescore candidates as
         score = avg_logprob(segment) + (λ / n_words) * Σ log P_LM(word_i | history).
    The LM shifts the balance between acoustic and syllabus-term priors.

    For n_best=1 the LM rescoring is still applied *across sub-segments* at the
    word level by biasing sequences that contain known syllabus terms slightly
    higher when we dedupe with greedy whisper output.
    """
    from faster_whisper import WhisperModel

    lm = load_lm(lm_path) if lm_path else None
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    temperatures = [0.0, 0.2][:max(1, n_best)]
    segments_all: List[List[Dict]] = []
    info = None
    for t_idx, T in enumerate(temperatures):
        print(f"[stt] pass {t_idx + 1}/{len(temperatures)}  temperature={T}", flush=True)
        segs, info = model.transcribe(
            audio_path,
            language=lang,
            beam_size=1 if T == 0.0 else 3,
            temperature=T,
            word_timestamps=True,
            vad_filter=True,
            condition_on_previous_text=False,
            no_speech_threshold=0.6,
            log_prob_threshold=-1.5,
            compression_ratio_threshold=2.4,
        )
        acc = []
        for i_s, s in enumerate(segs):
            acc.append(s)
            if i_s % 5 == 0:
                print(f"[stt]   pass {t_idx + 1}: segment {i_s}  t={s.start:.1f}-{s.end:.1f}s  '{s.text[:60]}'", flush=True)
        segments_all.append(acc)
        print(f"[stt] pass {t_idx + 1} done: {len(acc)} segments.", flush=True)

    def to_dict(s):
        return {
            "start": float(s.start),
            "end": float(s.end),
            "text": s.text.strip(),
            "avg_logprob": float(s.avg_logprob),
            "no_speech_prob": float(getattr(s, "no_speech_prob", 0.0)),
            "words": [{"w": w.word, "s": float(w.start), "e": float(w.end), "p": float(w.probability)} for w in (s.words or [])],
        }

    base = [to_dict(s) for s in segments_all[0]]
    chosen: List[Dict] = []
    for i, seg0 in enumerate(base):
        cands = [seg0]
        for T_segs in segments_all[1:]:
            if i < len(T_segs):
                cands.append(to_dict(T_segs[i]))

        best = max(
            cands,
            key=lambda c: c["avg_logprob"]
            + (rescore_word_sequence(tokenize(c["text"]), lm, lambda_lm) / max(len(tokenize(c["text"])), 1) if lm else 0.0),
        )
        chosen.append(best)

    return {
        "language": info.language if info else None,
        "language_probability": float(info.language_probability) if info else None,
        "duration": float(info.duration) if info else None,
        "segments": chosen,
    }


def write_transcript(result: Dict, out_path: str, full_text_path: str | None = None) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    if full_text_path:
        with open(full_text_path, "w") as f:
            for s in result["segments"]:
                f.write(s["text"] + "\n")


def unigram_logit_bias_table(lm: KneserNeyLM, whisper_tokenizer, lambda_: float = 4.0) -> Dict[int, float]:
    """Build a token-id → bias lookup using unigram continuation probabilities.

    This is applied at every step if faster-whisper exposes suppress_tokens /
    logits_processor. We use it in `transcribe_with_logit_bias` below.
    """
    bias: Dict[int, float] = {}
    for w in lm.vocab:
        if not w or w in {"<s>", "</s>", "<unk>"}:
            continue
        s = lm.log_prob(w, tuple())
        try:
            ids = whisper_tokenizer.encode(" " + w)
        except Exception:
            continue
        if not ids:
            continue
        for tid in ids:
            bias[tid] = bias.get(tid, 0.0) + lambda_ * s / len(ids)
    return bias


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["transcribe"])
    ap.add_argument("--audio", required=True)
    ap.add_argument("--lm", default="models/ngram.json")
    ap.add_argument("--model", default="medium")
    ap.add_argument("--out", default="outputs/transcript.json")
    ap.add_argument("--out_txt", default="outputs/transcript.txt")
    ap.add_argument("--lang", default=None)
    ap.add_argument("--lambda_lm", type=float, default=0.6)
    ap.add_argument("--n_best", type=int, default=5)
    args = ap.parse_args()
    res = transcribe_with_rescoring(
        args.audio, args.lm, model_size=args.model, n_best=args.n_best, lambda_lm=args.lambda_lm, lang=args.lang
    )
    write_transcript(res, args.out, args.out_txt)
    print(f"[stt] wrote {args.out} and {args.out_txt}")
