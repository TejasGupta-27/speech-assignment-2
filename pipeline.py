"""End-to-end pipeline orchestrator.

Usage
-----
    python pipeline.py --stage all
    python pipeline.py --stage stt        # just Part I
    python pipeline.py --stage lrl        # Part II
    python pipeline.py --stage tts        # Part III
    python pipeline.py --stage security   # Part IV
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src import denoise, lid, stt, g2p_ipa, translate, embed, prosody, tts, antispoof, adversarial, metrics, ngram_lm, clean_transcript


def ensure_dirs():
    for d in ["data", "models", "outputs", "reports"]:
        Path(d).mkdir(exist_ok=True)


def stage_denoise(inp="data/original_segment.wav", out="data/original_denoised.wav"):
    denoise.run(inp, out, method="deepfilter")
    return out


def stage_lid(audio, weights="models/lid.pt", train=True, epochs=10):
    if train or not Path(weights).exists():
        lid.train([audio], weights, epochs=epochs)
    out = lid.predict(audio, weights)
    Path("outputs").mkdir(exist_ok=True)
    Path("outputs/lid_pred.json").write_text(json.dumps(out, indent=2))
    m = lid.evaluate_f1(audio, weights)
    t = lid.boundary_timing_accuracy(audio, weights, None, tol_ms=200)
    Path("outputs/lid_eval.json").write_text(json.dumps({"lid": m, "timing": t}, indent=2))
    return out, m, t


def stage_ngram(corpus="data/syllabus.txt", out="models/ngram.json"):
    ngram_lm.build_from_file(corpus, out, order=3)
    return out


def stage_stt(audio, lm_path, out_json="outputs/transcript.json", out_txt="outputs/transcript.txt", model="small"):
    res = stt.transcribe_with_rescoring(audio, lm_path, model_size=model, n_best=1)
    stt.write_transcript(res, out_json, out_txt)
    # clean Whisper silence hallucinations before downstream stages
    clean_transcript.main_args = None  # placeholder, use functions directly
    import argparse
    cleaned = clean_transcript.clean_segments(res["segments"])
    obj = {"language": res.get("language"), "segments": cleaned}
    Path("outputs/transcript_clean.json").write_text(json.dumps(obj, indent=2, ensure_ascii=False))
    Path("outputs/transcript_clean.txt").write_text("\n".join(s["text"] for s in cleaned))
    return res


def stage_part2(transcript_json="outputs/transcript_clean.json"):
    text = " ".join(s["text"] for s in json.loads(Path(transcript_json).read_text())["segments"])
    ipa = g2p_ipa.transcript_to_ipa(text)
    Path("outputs/ipa.json").write_text(json.dumps(ipa, indent=2, ensure_ascii=False))
    lrl = translate.translate_text(text)
    Path("outputs/lrl.json").write_text(json.dumps(lrl, indent=2, ensure_ascii=False))
    Path("outputs/lrl.txt").write_text(lrl["lrl"])
    return ipa, lrl


def stage_tts(ref_wav="data/student_voice_ref.wav", lrl_text_path="outputs/lrl.txt", out_flat="outputs/output_LRL_flat.wav", out_warped="outputs/output_LRL_cloned.wav", ref_audio="data/original_denoised.wav"):
    # Prefer the NLLB-cleaned translation (strips repetitive garbage that trips
    # XTTS into long looping generations) if it exists.
    from src import clean_transcript  # noqa: F401  (imported for side effects in pipeline)

    clean_path = "outputs/lrl_clean.txt"
    if Path(clean_path).exists():
        text_path = clean_path
    else:
        text_path = lrl_text_path
    text = Path(text_path).read_text()
    tts.synth_long(text, ref_wav if Path(ref_wav).exists() else None, out_flat, language="hi", lrl_fallback="hin")
    prosody.run(ref_audio, out_flat, out_warped)
    return out_warped


def stage_security(audio="data/original_denoised.wav", ref_wav="data/student_voice_ref.wav", cloned="outputs/output_LRL_cloned.wav", flat="outputs/output_LRL_flat.wav", weights="models/lid.pt"):
    bona = [p for p in [ref_wav, audio] if Path(p).exists()]
    spoof = [p for p in [flat, cloned] if Path(p).exists()]
    cm = None
    if bona and spoof:
        cm = antispoof.train_and_eval(bona, spoof, "models/cm.pt", feature="lfcc", epochs=20)
    adv = adversarial.fgsm_flip(audio, weights)
    Path("outputs/fgsm.json").write_text(json.dumps(adv, indent=2))
    return {"cm": cm, "fgsm": adv}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["all", "stt", "lrl", "tts", "security"], default="all")
    ap.add_argument("--audio", default="data/original_segment.wav")
    ap.add_argument("--ref", default="data/student_voice_ref.wav")
    args = ap.parse_args()
    ensure_dirs()
    denoised = stage_denoise(args.audio)
    lm = stage_ngram()
    if args.stage in ("all", "stt"):
        _, f1, timing = stage_lid(denoised, train=True, epochs=15)
        res = stage_stt(denoised, lm, model="small")
        print(f"[pipe] LID F1 (eng/hin macro)= {f1.get('macro_f1_eng_hin')}  timing hit-rate (200ms)= {timing.get('hit_rate')}")
    if args.stage in ("all", "lrl"):
        stage_part2()
    if args.stage in ("all", "tts"):
        stage_tts(ref_wav=args.ref, ref_audio=denoised)
    if args.stage in ("all", "security"):
        stage_security(audio=denoised, ref_wav=args.ref, weights="models/lid.pt")


if __name__ == "__main__":
    main()
