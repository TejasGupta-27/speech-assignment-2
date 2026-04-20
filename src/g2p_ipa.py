"""Task 2.1 — Hinglish (English+Devanagari+Romanized-Hindi) to IPA.

Why a custom mapper
-------------------
Off-the-shelf G2Ps fail on code-switching:
  - phonemizer/espeak assumes a single language/voice per pass.
  - Indic G2Ps don't handle English spelling.
  - English G2Ps can't read Devanagari or romanized Hindi like "kya", "stochastic ka".

We build three layers:
  1. script router: detect per-word script (Latin / Devanagari / mixed).
  2. English G2P: phonemizer(espeak-ng, lang=en-us) with technical-term overrides.
  3. Hindi G2P: hand-written Devanagari-to-IPA table + schwa-deletion rules,
     plus a romanized-Hindi (Hinglish spelling) layer.

Output: space-separated IPA words, with '| ' inserted at code-switch boundaries.
"""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Tuple

DEV_VOWELS = {
    "अ": "ə", "आ": "aː", "इ": "ɪ", "ई": "iː", "उ": "ʊ", "ऊ": "uː",
    "ऋ": "ɾɪ", "ए": "eː", "ऐ": "ɛː", "ओ": "oː", "औ": "ɔː",
}
DEV_MATRA = {
    "ा": "aː", "ि": "ɪ", "ी": "iː", "ु": "ʊ", "ू": "uː", "ृ": "ɾɪ",
    "े": "eː", "ै": "ɛː", "ो": "oː", "ौ": "ɔː", "ं": "̃", "ः": "h", "ँ": "̃",
}
DEV_CONS = {
    "क": "k", "ख": "kʰ", "ग": "g", "घ": "gʱ", "ङ": "ŋ",
    "च": "tʃ", "छ": "tʃʰ", "ज": "dʒ", "झ": "dʒʱ", "ञ": "ɲ",
    "ट": "ʈ", "ठ": "ʈʰ", "ड": "ɖ", "ढ": "ɖʱ", "ण": "ɳ",
    "त": "t̪", "थ": "t̪ʰ", "द": "d̪", "ध": "d̪ʱ", "न": "n",
    "प": "p", "फ": "pʰ", "ब": "b", "भ": "bʱ", "म": "m",
    "य": "j", "र": "ɾ", "ल": "l", "व": "ʋ",
    "श": "ʃ", "ष": "ʂ", "स": "s", "ह": "ɦ",
    "क़": "q", "ख़": "x", "ग़": "ɣ", "ज़": "z", "ड़": "ɽ", "ढ़": "ɽʱ", "फ़": "f",
}
HALANT = "्"
TECH_IPA_OVERRIDES = {  # hand-curated to match expected pronunciation in a Speech course
    "stochastic": "stəˈkæstɪk",
    "cepstrum": "ˈkɛpstɹəm",
    "mfcc": "ˌɛm.ɛfˌsiːˈsiː",
    "lfcc": "ˌɛl.ɛfˌsiːˈsiː",
    "cqcc": "ˌsiːˌkjuːˌsiːˈsiː",
    "whisper": "ˈwɪspɚ",
    "wav2vec": "ˈwɑːv.tuː.vɛk",
    "transformer": "tɹænsˈfɔɹmɚ",
    "hmm": "ˌeɪtʃˌɛmˈɛm",
    "gmm": "ˌdʒiːˌɛmˈɛm",
    "viterbi": "vɪˈtɚbi",
    "hinglish": "ˈhɪŋglɪʃ",
    "code-switching": "koʊdˈswɪtʃɪŋ",
    "phoneme": "ˈfoʊniːm",
    "grapheme": "ˈgɹæfiːm",
}
ROMAN_HINDI = {  # common Hinglish rime chunks; longest-match first
    "aa": "aː", "ee": "iː", "oo": "uː",
    "ai": "ɛː", "au": "ɔː", "ou": "oː", "ei": "eː",
    "sh": "ʃ", "ch": "tʃ", "th": "t̪ʰ", "dh": "d̪ʱ", "ph": "pʰ", "kh": "kʰ", "gh": "gʱ", "bh": "bʱ",
    "ny": "ɲ", "ng": "ŋ",
    "a": "ə", "e": "ɛ", "i": "ɪ", "o": "ɔ", "u": "ʊ",
    "k": "k", "g": "g", "t": "ʈ", "d": "ɖ", "n": "n", "p": "p", "b": "b", "m": "m",
    "y": "j", "r": "ɾ", "l": "l", "v": "ʋ", "w": "ʋ", "s": "s", "h": "ɦ", "f": "f", "z": "z", "j": "dʒ", "c": "k", "x": "ks", "q": "k",
}
ROMAN_KEYS = sorted(ROMAN_HINDI.keys(), key=len, reverse=True)
HINGLISH_HINTS = {
    "kya", "hai", "nahi", "nahin", "toh", "mera", "tera", "uska", "hum", "hamara", "woh", "yeh",
    "bhi", "kyun", "kaise", "bolna", "chalo", "karo", "karta", "karti", "kiya", "par", "par", "aur",
    "lekin", "agar", "matlab", "theek", "thik", "achha", "acha", "haan", "haanji", "namaste", "samajh",
}


def is_devanagari(s: str) -> bool:
    return any("\u0900" <= c <= "\u097f" for c in s)


def is_english_word(s: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z][A-Za-z'\-]*", s))


def detect_lang(word: str) -> str:
    if is_devanagari(word):
        return "hi"
    if word.lower() in HINGLISH_HINTS:
        return "hi-roman"
    return "en"


def dev_word_to_ipa(w: str) -> str:
    """Devanagari → IPA with schwa-deletion heuristic."""
    out: List[str] = []
    i = 0
    s = unicodedata.normalize("NFC", w)
    while i < len(s):
        c = s[i]
        nxt = s[i + 1] if i + 1 < len(s) else ""
        if c in DEV_CONS:
            out.append(DEV_CONS[c])
            if nxt == HALANT:
                i += 2
                continue
            if nxt in DEV_MATRA:
                out.append(DEV_MATRA[nxt])
                i += 2
                continue
            out.append("ə")
            i += 1
        elif c in DEV_VOWELS:
            out.append(DEV_VOWELS[c])
            i += 1
        elif c in DEV_MATRA:
            out.append(DEV_MATRA[c])
            i += 1
        else:
            i += 1
    # schwa-deletion: delete ə if preceded by a cluster end at word-final or between two consonants
    joined = "".join(out)
    joined = re.sub(r"ə$", "", joined)  # word-final schwa
    return joined


def roman_hindi_to_ipa(w: str) -> str:
    s = w.lower()
    i = 0
    out: List[str] = []
    while i < len(s):
        matched = False
        for k in ROMAN_KEYS:
            if s.startswith(k, i):
                out.append(ROMAN_HINDI[k])
                i += len(k)
                matched = True
                break
        if not matched:
            i += 1
    return "".join(out)


def english_to_ipa(w: str, phonemizer_backend=None) -> str:
    low = w.lower().strip(".,!?:;\"'()[]")
    if low in TECH_IPA_OVERRIDES:
        return TECH_IPA_OVERRIDES[low]
    if phonemizer_backend is not None:
        try:
            return phonemizer_backend.phonemize([low], strip=True)[0]
        except Exception:
            pass
    # very small fallback: spelling-based English pseudo-IPA
    out = low
    pairs = [("ph", "f"), ("ch", "tʃ"), ("sh", "ʃ"), ("th", "θ"), ("ng", "ŋ"), ("ee", "iː"), ("oo", "uː"),
             ("ea", "iː"), ("ai", "eɪ"), ("ay", "eɪ"), ("ou", "aʊ"), ("ow", "aʊ"),
             ("a", "æ"), ("e", "ɛ"), ("i", "ɪ"), ("o", "ɒ"), ("u", "ʌ"),
             ("y", "i"), ("c", "k"), ("x", "ks"), ("q", "k"), ("j", "dʒ"), ("w", "w"), ("r", "ɹ")]
    for src, tgt in pairs:
        out = out.replace(src, tgt)
    return out


def load_phonemizer():
    try:
        from phonemizer.backend import EspeakBackend

        return EspeakBackend("en-us", preserve_punctuation=False, with_stress=True)
    except Exception as e:
        print(f"[g2p] phonemizer unavailable: {e}; using rule-based fallback.")
        return None


def transcript_to_ipa(text: str) -> Dict:
    """Convert a code-switched transcript into IPA with boundary markers."""
    phon = load_phonemizer()
    tokens = re.findall(r"[A-Za-z\u0900-\u097F'\-]+|[^\sA-Za-z\u0900-\u097F]", text)
    ipa_words: List[str] = []
    switches: List[int] = []
    prev_lang = None
    for tok in tokens:
        if not re.search(r"[A-Za-z\u0900-\u097F]", tok):
            ipa_words.append(tok)
            continue
        lang = detect_lang(tok)
        if prev_lang is not None and lang != prev_lang:
            switches.append(len(ipa_words))
            ipa_words.append("|")
        if lang == "en":
            ipa_words.append(english_to_ipa(tok, phon))
        elif lang == "hi":
            ipa_words.append(dev_word_to_ipa(tok))
        elif lang == "hi-roman":
            ipa_words.append(roman_hindi_to_ipa(tok))
        prev_lang = lang
    return {"ipa": " ".join(ipa_words), "boundaries": switches, "n_tokens": len(tokens)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", help="raw text")
    ap.add_argument("--transcript_json", help="STT output json")
    ap.add_argument("--out", default="outputs/ipa.json")
    args = ap.parse_args()
    if args.transcript_json:
        obj = json.loads(Path(args.transcript_json).read_text())
        text = " ".join(s["text"] for s in obj["segments"])
    else:
        text = args.text
    res = transcript_to_ipa(text)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(f"[g2p] wrote {args.out}  ({len(res['ipa'])} chars, {len(res['boundaries'])} switches)")
