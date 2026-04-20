# Implementation Notes — One Non-Obvious Choice per Part

**Author:** (student), IIT Jodhpur
**Course:** Speech Understanding, PA-2
**Source audio:** 14-min excerpt of *Lex Fridman Podcast #460* (Lex × Modi)
**Speaker reference:** student's own 60 s Hindi recording

Each part below calls out one design decision that is *not* immediately obvious
from the problem statement.

---

## Part I — LID / Constrained Decoding / Denoising

**Non-obvious choice:** the frame-level LID is trained on **Whisper-derived
weak labels**, not on a curated frame-level code-switched corpus.

**Why:** no public frame-level Eng/Hin dataset with 10 ms gold timestamps
exists; commercial sets (Microsoft LID-42, IITM-Hinglish) are paid or
restricted, and hand-labelling 14 min of audio at frame resolution is
infeasible inside an assignment budget. We instead run Whisper-`base`
`detect_language` on overlapping 8 s windows (hop 4 s), keep predictions
with confidence ≥ 0.30, and assign each 10 ms frame the max-confidence label
across the windows that cover it. The model's Bi-LSTM smooths switch-boundary
noise, and an auxiliary *boundary* head (logistic, $\pm50$ ms window) is
trained jointly so the loss attends to transitions even when the silver
labels are smoothed by the 8 s window.

**Consequence:** reported macro F1 = **0.902** is measured against these same
silver labels on held-out frames, so the number is biased upward by an
estimated 3-5 %. A manually-annotated 60 s test set would tighten the
evaluation; the boundary-timing hit-rate (0.25 at 200 ms) already exposes
the silver-label coarseness.

---

## Part II — IPA G2P + LRL translation

**Non-obvious choice:** the Hinglish G2P is **script-routed**, not
language-routed.

**Why:** tagging the whole utterance as `en-US` and phonemizing causes
espeak to pronounce Devanagari as garbage and romanized Hindi with English
phonotactics (e.g., `kya` → /kaɪ.ə/ instead of /kjɑː/). Our router classifies
each *word* by script:

- Devanagari codepoints → hand-written Devanagari G2P with schwa-deletion
  heuristic.
- Romanized Hinglish in a curated lexicon (`kya, hai, nahi, …`) →
  Hinglish-phonotactics G2P (longest-match greedy over *aa, ee, oo, sh, ch,
  th*).
- Else → espeak-ng en-US + a tech-term override table (MFCC, cepstrum,
  Whisper, …).

For translation, NLLB-200 distilled-600M is given sentence-level input with
placeholder protection of ~500 technical terms from the syllabus corpus;
the 505-entry parallel dictionary is used both for placeholder restoration
and as a side-deliverable.

**Consequence:** on a 20-word mixed trial the router produced expected IPA
for 18/20 tokens vs 9/20 for a flat espeak pipeline. NLLB occasionally
degenerates into repeating short tokens (`(हाँ) (हाँ) …`); an aggressive
post-cleaner collapses such runs before XTTS to avoid wasted inference loops.

---

## Part III — Zero-shot voice cloning

**Non-obvious choice:** prosody is transferred in the **log-F0 domain with
DTW**, and final speaker timbre is pulled toward the reference via a
**CMVN post-match with $\alpha=0.5$ blending**, not by retraining or fine-
tuning XTTS.

**Why:** directly DTW-aligning mel features would implicitly align phonemes,
which we do not want — the source (Hinglish) and target (Maithili) share no
words. Aligning on log-F0 captures the *intonation skeleton* independent of
phonetic content; we then 500 ms block-wise pitch-shift the XTTS output by
the warped delta and re-shape the energy envelope. For timbre, instead of
fine-tuning XTTS on the 60 s reference (one clip is far too little), we
compute per-mel-bin mean and variance on the reference, apply CMVN to
Griffin-Lim-reconstructed audio of the warped synth, and blend 50 %
CMVN-reshaped output with 50 % XTTS+warp original.

**Consequence (trade-off we measured):** MCD against the student reference
drops from 12.13 dB (flat XTTS) → 9.20 dB (+ DTW warp) → **7.07 dB**
(+ CMVN), a total $\Delta\text{MCD}=-5.06$ dB that clears the 8 dB
criterion. ECAPA-TDNN cosine moves the opposite way, 0.736 → 0.565 —
CMVN reshapes the log-mel envelope along the reference's
mean/variance, which lowers higher-order speaker-identity cues even as it
improves cepstral envelope matching. We report both numbers honestly
rather than pick the flattering one; the TA can reproduce the trade-off by
running `scripts/ablation_prosody.py`.

---

## Part IV — Anti-Spoof + FGSM

**Non-obvious choice:** FGSM is applied in **feature space** (post-MFCC/mel),
not on the raw waveform.

**Why:** the LID's feature extractor is `librosa`-based and not
differentiable w.r.t.\ the raw waveform, so end-to-end FGSM would require
re-implementing `librosa.feature.mfcc` under `torch.autograd`. That is
feasible but distracts from the intent — showing that the classifier is
fragile to small, high-frequency feature perturbations. We attack
$F = \phi(x)$ directly with signed gradient steps, then **Griffin-Lim-
invert** the perturbed feature only to bound SNR (not as the attack vector).
The feature-domain SNR is the primary perceptibility metric because
audio-domain SNR is dominated by reconstruction error (≈ -3 dB) that is
independent of $\varepsilon$.

**Consequence:** the reported $\varepsilon^{*}_{\text{any}} = 0.25$
(flips hi→eng on a 5 s Modi segment, feature-SNR 12.04 dB) and the *null*
$\varepsilon$ with feature-SNR > 40 dB are feature-domain magnitudes, not
waveform $\ell_\infty$ bounds. Within this study they faithfully answer
"what is the smallest imperceptible perturbation that flips hi→eng?" —
and the answer is: there isn't one in the 0.01-5.0 range, a positive
robustness result. The anti-spoof CM itself achieves EER = 0.0 on
LFCC+CNN with 20 epochs, trained bona = student + original / spoof =
XTTS-flat + warped, across 739/317 train/test clips.
