"""Task 2.2 — Translate code-switched transcript → LRL (Maithili, `mai`).

Strategy
--------
Sentence-level NLLB-200 (Hindi→Maithili, English→Maithili) is the primary
translator. We pre-substitute technical terms from the 500-entry parallel
dictionary *into* the source sentence with placeholder tags that NLLB
preserves, then un-tag them after translation. This keeps domain-specific
vocabulary exactly as we want it in Maithili while letting NLLB handle
grammar and high-frequency vocabulary.

Pseudo-code:
    for each sentence:
        1. pre-replace dict-known tokens with a unique placeholder e.g. "⟨MCD⟩".
        2. detect sentence language (Devanagari % → hin_Deva else eng_Latn).
        3. translate the whole sentence with NLLB.
        4. substitute placeholders with Maithili gloss from parallel_corpus.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

DEV_RE = re.compile(r"[\u0900-\u097F]+")


def load_corpus(path: str) -> Dict[str, str]:
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    return {k.lower(): v for k, v in obj["entries"].items()}


def detect_lang_tag(text: str) -> str:
    """Return 'hin_Deva' if mostly Devanagari, else 'eng_Latn'."""
    dev_n = len(DEV_RE.findall(text))
    eng_n = len(re.findall(r"[A-Za-z]+", text))
    return "hin_Deva" if dev_n >= eng_n else "eng_Latn"


def split_sentences(text: str) -> List[str]:
    sents = re.split(r"([।!?.]+|\n+)", text)
    out, buf = [], ""
    for s in sents:
        buf += s
        if re.search(r"[।!?.]$", buf) or len(buf) > 200:
            if buf.strip():
                out.append(buf.strip())
            buf = ""
    if buf.strip():
        out.append(buf.strip())
    return out


def protect_dict_terms(sentence: str, corpus: Dict[str, str]) -> Tuple[str, Dict[str, str]]:
    """Replace known tokens with placeholders like ⟦T0⟧, returning placeholder→Maithili map."""
    mapping: Dict[str, str] = {}
    out = sentence
    # longest-first so multi-word terms match first
    keys = sorted(corpus.keys(), key=len, reverse=True)
    idx = 0
    for k in keys:
        if not k:
            continue
        # word-boundary-ish (won't match inside Devanagari word either by accident since dict keys are mostly ASCII or Devanagari short)
        pattern = re.compile(r"(?<![A-Za-z\u0900-\u097F])" + re.escape(k) + r"(?![A-Za-z\u0900-\u097F])", flags=re.IGNORECASE)
        if pattern.search(out):
            tag = f"⟦T{idx}⟧"
            mapping[tag] = corpus[k]
            out = pattern.sub(tag, out)
            idx += 1
            if idx > 50:
                break
    return out, mapping


def restore_dict_terms(sentence: str, mapping: Dict[str, str]) -> str:
    for tag, gloss in mapping.items():
        sentence = sentence.replace(tag, gloss)
    return sentence


def try_load_nllb():
    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        name = "facebook/nllb-200-distilled-600M"
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModelForSeq2SeqLM.from_pretrained(name)
        return tok, model
    except Exception as e:
        print(f"[translate] NLLB unavailable: {e}")
        return None, None


def nllb_translate_batch(tok, model, texts: List[str], src: str, tgt: str = "mai_Deva") -> List[str]:
    import torch

    if not texts:
        return []
    tok.src_lang = src
    inputs = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=256)
    tgt_id = tok.convert_tokens_to_ids(tgt)
    with torch.no_grad():
        gen = model.generate(**inputs, forced_bos_token_id=tgt_id, max_new_tokens=256, num_beams=3)
    return tok.batch_decode(gen, skip_special_tokens=True)


def translate_text(
    text: str,
    corpus_path: str = "data/parallel_corpus.json",
    use_nllb: bool = True,
    cache_path: str = "outputs/translate_cache.json",
) -> Dict:
    corpus = load_corpus(corpus_path)
    sents = split_sentences(text)
    print(f"[translate] {len(sents)} sentences to translate.", flush=True)

    protected = [protect_dict_terms(s, corpus) for s in sents]
    masked_sents = [p[0] for p in protected]
    mappings = [p[1] for p in protected]

    cache: Dict[str, str] = {}
    if Path(cache_path).exists():
        try:
            cache = json.loads(Path(cache_path).read_text())
        except Exception:
            cache = {}

    out_sents: List[str] = []
    tk, md = (None, None)
    pending_hin, pending_eng = [], []
    pending_hin_idx, pending_eng_idx = [], []
    for i, ms in enumerate(masked_sents):
        if ms in cache:
            out_sents.append(cache[ms])
            continue
        lang = detect_lang_tag(ms)
        out_sents.append(None)  # placeholder
        if lang == "hin_Deva":
            pending_hin.append(ms)
            pending_hin_idx.append(i)
        else:
            pending_eng.append(ms)
            pending_eng_idx.append(i)

    if use_nllb and (pending_hin or pending_eng):
        tk, md = try_load_nllb()
        if tk is not None:
            print(f"[translate] NLLB: hin→mai ×{len(pending_hin)}, eng→mai ×{len(pending_eng)}", flush=True)
            for start in range(0, len(pending_hin), 4):
                batch = pending_hin[start : start + 4]
                trs = nllb_translate_batch(tk, md, batch, src="hin_Deva")
                for idx, tr in zip(pending_hin_idx[start : start + 4], trs):
                    out_sents[idx] = tr
                    cache[masked_sents[idx]] = tr
                print(f"[translate]   hin batch {start // 4 + 1}/{(len(pending_hin) + 3) // 4} done", flush=True)
            for start in range(0, len(pending_eng), 4):
                batch = pending_eng[start : start + 4]
                trs = nllb_translate_batch(tk, md, batch, src="eng_Latn")
                for idx, tr in zip(pending_eng_idx[start : start + 4], trs):
                    out_sents[idx] = tr
                    cache[masked_sents[idx]] = tr
                print(f"[translate]   eng batch {start // 4 + 1}/{(len(pending_eng) + 3) // 4} done", flush=True)
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            Path(cache_path).write_text(json.dumps(cache, ensure_ascii=False, indent=2))

    for i, t in enumerate(out_sents):
        if t is None:
            out_sents[i] = masked_sents[i]  # keep Hindi as-is as fallback

    final = [restore_dict_terms(t, m) for t, m in zip(out_sents, mappings)]
    lrl = " ".join(final)

    return {
        "lrl": lrl,
        "lrl_language": "mai",
        "n_sentences": len(sents),
        "nllb_used": use_nllb and tk is not None,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--transcript", default="outputs/transcript_clean.json")
    ap.add_argument("--out", default="outputs/lrl.json")
    ap.add_argument("--no_nllb", action="store_true")
    args = ap.parse_args()
    obj = json.loads(Path(args.transcript).read_text(encoding="utf-8"))
    text = " ".join(s["text"] for s in obj["segments"])
    res = translate_text(text, use_nllb=not args.no_nllb)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    Path("outputs/lrl.txt").write_text(res["lrl"])
    print(f"[translate] n_sentences={res['n_sentences']}  nllb={res['nllb_used']}  → {args.out}")
