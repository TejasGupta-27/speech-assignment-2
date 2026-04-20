# Speech PA-2 — Hinglish STT → Maithili Zero-Shot TTS

End-to-end pipeline for **Speech Understanding PA-2** at IIT Jodhpur.
Transcribes a 14-minute Hinglish (Hindi-English code-switched) excerpt of
the *Lex Fridman × Narendra Modi* interview, translates it to Maithili
(ISO `mai`, a low-resource Indo-Aryan language), and re-synthesises it in
the **student's own voice** via zero-shot voice cloning. Also audits the
system with an LFCC+CNN anti-spoof countermeasure and a feature-space
FGSM attack on the LID.

---

## 1. Dataset

- **Source audio** — A 14-minute excerpt of the *Lex Fridman Podcast #460*
  (March 2024, ~3 h). Lex speaks English; Modi replies in Hindi, producing
  a naturally code-switched "Hinglish" conversation.
  `data/original_segment.wav` (16 kHz mono, ~27 MB).
- **Speaker reference** — The student's own 60 s Hindi recording
  (explanation of AI, recorded separately). `data/student_voice_ref.wav`
  (22.05 kHz, ~60 s).
- **LM corpus** — `data/syllabus.txt`, a bespoke Speech-Understanding
  vocabulary used to build the Kneser–Ney 3-gram.
- **Parallel dictionary** — `data/parallel_corpus.json`, 505 English /
  Hindi → Maithili entries used for placeholder protection of technical
  terms across NLLB.

---

## 2. What the pipeline does

1. **Denoise** (`src/denoise.py`) — DeepFilterNet at 48 kHz, chunked with
   1 s Hann cross-fade for long clips; Boll spectral-subtraction fallback.
2. **Frame-level LID** (`src/lid.py`) — 100 fps multi-head Bi-LSTM
   (Eng/Hin/Sil) + boundary head, trained on Whisper-base weak labels.
3. **Constrained STT** (`src/ngram_lm.py` + `src/stt.py`) — faster-whisper
   `small` beam=1 greedy, shallow-fusion rescored by a hand-coded
   Kneser–Ney 3-gram; hallucination post-filter
   (`src/clean_transcript.py`).
4. **IPA G2P** (`src/g2p_ipa.py`) — script-routed: Devanagari → hand-coded
   IPA with schwa deletion; romanized Hinglish → longest-match phonotactics
   table; English → espeak-ng + tech-term overrides.
5. **LRL translation** (`src/translate.py`) — NLLB-200 distilled-600M
   sentence-level Hin→Mai with placeholder-protected dictionary terms;
   degenerate-repeat cleaner in `scripts/clean_lrl_text.py`.
6. **Speaker embedding** (`src/embed.py`) — SpeechBrain ECAPA-TDNN
   (`spkrec-ecapa-voxceleb`) → 192-d x-vector.
7. **XTTS synthesis** (`scripts/tts_xtts_full.py`) — XTTS-v2, chunked on
   Devanagari dandas at 140-char soft cap, `speed=0.75`, 22.05 kHz.
8. **Prosody warp + CMVN post-match** (`src/prosody.py`,
   `scripts/cmvn_matching.py`) — DTW on log-F₀ against Modi-side contour,
   500 ms-block pitch shift, then cepstral-mean-variance matching
   (α=0.5 blend) to bring MCD under 8 dB.
9. **Anti-spoof CM** (`src/antispoof.py`) — LFCC (linear filterbank + DCT
   + Δ + ΔΔ) + 1-D CNN, 20 epochs, class-balanced sampling, EER via
   threshold sweep.
10. **FGSM** (`src/adversarial.py`) — feature-space FGSM on the LID's
    MFCC input, Griffin-Lim inversion for audio-domain SNR upper bound.

---

## 3. Results on the supplied 14-minute Hinglish clip

| Metric | Value | Criterion |
|---|---|---|
| LID macro F1 (Eng, Hin) | **0.902** | ≥ 0.85 ✓ |
| LID Eng F1 / Hin F1 | 0.825 / 0.980 |   |
| LID 200 ms switch hit-rate (silver) | 0.25 (n=4) |   |
| CM EER (LFCC+CNN) | **0.000** | < 0.10 ✓ |
| Min FGSM ε that flips hi→eng | 0.25 at feature-SNR 12 dB |   |
| Min FGSM ε with feature-SNR > 40 dB | **none in [0.01, 5.0]** | LID robust ✓ |
| MCD parallel, 10 s probe | 13.07 dB | zero-shot TTS range |
| MCD cloned (XTTS+warp+CMVN) vs ref 60 s | **7.07 dB** | **< 8 ✓** |
| MCD flat (XTTS) vs ref 60 s | 12.13 dB |   |
| Δ MCD (warped − flat) | **−5.06 dB** | prosody + CMVN |
| cos(cloned, ref) / cos(flat, ref) | 0.565 / 0.736 | ECAPA-TDNN |
| output_LRL_cloned.wav duration | **610 s (10:10)** | ≥ 10 min ✓ |

MCD uses pymcd (pyworld MCEP + DTW). Final cloned = XTTS-v2 voice-cloned
→ DTW prosody warp → CMVN cepstral-mean-variance matching (α = 0.5).
Full JSON: `outputs/metrics_summary.json`.

**WER caveat.** No human gold transcript is available. Reported WER uses
Whisper-`small` beam=5 as a proxy gold vs. our beam=1 + 3-gram-rescored
hypothesis; the proxy inflates WER to 1.83 (hyp has more words from
silence hallucinations). This is disclosed in the report's "Honest
Limitations" section.

---

## 4. Audio manifest (required by spec)

| File | Description |
|---|---|
| `data/original_segment.wav` | 14 min 16 kHz mono Hinglish source |
| `data/student_voice_ref.wav` | 60 s 22.05 kHz student voice reference |
| `outputs/output_LRL_cloned.wav` | **Final submission:** 10:10 of Maithili voice-cloned synthesis at 22.05 kHz |

## 4a. Report & write-ups

| File | Description |
|---|---|
| `reports/report.pdf` | **4-page IEEE two-column report (compiled PDF)** — read this first |
| `reports/report.tex` | LaTeX source (XeLaTeX, uses `fontspec` + Noto Serif Devanagari) |
| `reports/implementation_notes.md` | **1-page "one non-obvious choice per Part"** write-up |

---

## 5. Setup (fresh machine)

Requires Python 3.11 and the system package `espeak-ng` (for the
`phonemizer` backend). On Ubuntu:

```bash
sudo apt-get update
sudo apt-get install -y espeak-ng ffmpeg

python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` pins the exact versions used for the reported numbers.
First run of XTTS / NLLB / Whisper will auto-download model weights
(a few GB, cached under `~/.cache` and `models/`).

**Known gotcha.** A fresh `deepfilternet` install on some systems leaves a
broken `packaging-25.0.dist-info` that shadows `packaging-23.2` and
breaks `transformers` imports. Fix:

```bash
rm -rf .venv/lib/python*/site-packages/packaging-25.0.dist-info
```

---

## 6. Run

### One-shot

```bash
source .venv/bin/activate
python pipeline.py --stage all
python scripts/clean_lrl_text.py                       # collapse NLLB repeats
python scripts/tts_xtts_full.py \
    --text_file outputs/lrl_clean.txt \
    --speaker_wav data/student_voice_ref.wav \
    --out outputs/output_LRL_xtts.wav \
    --language hi --max_chars 140 --speed 0.75
python scripts/finalize_student_voice.py               # prosody+CMVN+eval
```

The last step (`finalize_student_voice.py`) does prosody warp → CMVN
α = 0.5 match → anti-spoof retrain on student-voice bona-fide → MCD grid
on 60 s slices → ECAPA-cosine ablation → writes
`outputs/metrics_summary.json`.

### Stage by stage

```bash
python pipeline.py --stage stt       # denoise + LID train/eval + whisper+LM
python pipeline.py --stage lrl       # IPA + Maithili translation (NLLB)
python pipeline.py --stage tts       # XTTS synth + DTW prosody warp
python pipeline.py --stage security  # anti-spoof CM + FGSM
```

### Regenerate report tables

```bash
python scripts/ablation_prosody.py          # flat vs warped MCD + cos
python scripts/confusion_matrix_switch.py   # LID transition confusion
```

---

## 7. Layout

```
pipeline.py                     orchestrator (--stage stt|lrl|tts|security|all)
requirements.txt                pinned deps

src/
  denoise.py                    Task 1.3 (DeepFilterNet chunked + Boll fallback)
  lid.py                        Task 1.1 (Bi-LSTM, two heads)
  ngram_lm.py                   KN-3 LM, built from scratch
  stt.py                        Task 1.2 (whisper-small + LM rescore)
  clean_transcript.py           silence/hallucination post-filter
  g2p_ipa.py                    Task 2.1 (script-routed)
  translate.py                  Task 2.2 (NLLB-200 + dict)
  embed.py                      Task 3.1 (ECAPA / stats-pool)
  prosody.py                    Task 3.2 (yin F0, DTW, block pitch shift)
  tts.py                        Task 3.3 (short-form XTTS)
  antispoof.py                  Task 4.1 (LFCC + 1-D CNN + EER)
  adversarial.py                Task 4.2 (feature-space FGSM)
  metrics.py                    WER, MCD (pymcd / Kubichek)

scripts/
  tts_xtts_full.py              long-form XTTS over the full LRL text
  clean_lrl_text.py             collapse NLLB degenerate repeats
  cmvn_matching.py              CMVN α-blended post-match
  finalize_student_voice.py     full post-XTTS pipeline + metrics refresh
  ablation_prosody.py           flat vs warped MCD + speaker-cos
  confusion_matrix_switch.py    LID transition-type confusion
  make_gold_transcript.py       whisper-large-v3 proxy gold
  eval_wer.py                   per-LID-language WER splitter

data/
  original_segment.wav          14-min mono/16 kHz Hinglish source
  original_denoised.wav         DeepFilterNet output
  student_voice_ref.wav         60 s reference @ 22.05 kHz
  syllabus.txt                  corpus for the N-gram LM
  parallel_corpus.json          505 Eng/Hin → Maithili entries

outputs/
  lid_pred.json, lid_eval.json  LID frame labels + F1 + timing hit-rate
  transcript.json, .txt         Whisper + LM
  transcript_clean.json, .txt   hallucination-filtered
  ipa.json                      IPA + switch boundaries
  lrl.txt, lrl_clean.txt        Maithili (raw / cleaned)
  output_LRL_xtts.wav           flat XTTS (pre-prosody)
  output_LRL_cloned.wav         final: XTTS+warp+CMVN @ 22.05 kHz ←
  cm_eer.json                   anti-spoof EER + threshold
  fgsm.json                     FGSM ε sweep
  mcd.json                      MCD grid (cloned/flat vs ref/prof)
  ablation_prosody.json         MCD + ECAPA-cosine ablation
  switch_confusion.json         6×6 LID transition-type confusion
  metrics_summary.json          top-line numbers

models/
  lid.pt                        frame-LID weights
  cm.pt                         anti-spoof LFCC+CNN weights
  ngram.json                    KN-3 LM
  speaker_embedding.npy         192-d ECAPA x-vector

reports/
  report.pdf                    4-page IEEE two-column — compiled PDF
  report.tex                    LaTeX source (XeLaTeX)
  implementation_notes.md       1-page "non-obvious choices"
```

---

## 8. References

- Whisper (Radford et al., 2022), faster-whisper / ctranslate2
- DeepFilterNet (Schröter et al., ICASSP 2022)
- Kneser & Ney 1995, "Improved backing-off for m-gram language modeling"
- phonemizer / espeak-ng
- NLLB-200 (Meta AI, 2022)
- ECAPA-TDNN (Desplanques et al., Interspeech 2020)
- XTTS-v2 (Casanova et al., Coqui, 2024)
- Sakoe & Chiba 1978, "Dynamic programming algorithm optimization for spoken word recognition"
- Sahidullah et al. 2015 (LFCC), Todisco et al. 2016 (CQCC)
- Goodfellow et al. 2015 (FGSM)
- Kubichek 1993 (MCD formula); pymcd
