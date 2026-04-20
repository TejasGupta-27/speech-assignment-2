"""N-gram Language Model with Kneser-Ney smoothing, built from scratch.

Trained on a user-supplied Speech Course Syllabus corpus. Used as a logit-bias
term at decode time by converting the unigram+bigram scores into a per-token bias.

Math
----
For a Whisper token t with subword form s_t, we compute
    bias(t | ctx) = λ * log P_LM(w | ctx)
where w is the *word* formed by the running token buffer and ctx is the last n-1 words.
Whisper decodes subwords; we heuristically add the bias when a token completes a
word (starts with a space or is end-of-sentence) and the formed word appears in the
LM vocabulary. Otherwise we add a smaller partial-match bias based on unigram prob.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

BOS, EOS, UNK = "<s>", "</s>", "<unk>"


def tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-z\u0900-\u097F]+", text.lower())


class KneserNeyLM:
    def __init__(self, order: int = 3, discount: float = 0.75):
        self.order = order
        self.discount = discount
        self.counts: List[Counter] = [Counter() for _ in range(order)]
        self.cont_counts: Dict[Tuple[str, ...], set] = defaultdict(set)
        self.vocab: set = set()
        self.bigram_left: Dict[str, set] = defaultdict(set)
        self.total_bigrams: int = 0

    def fit(self, corpus_lines: List[str]) -> None:
        for line in corpus_lines:
            toks = [BOS] + tokenize(line) + [EOS]
            self.vocab.update(toks)
            for n in range(1, self.order + 1):
                for i in range(len(toks) - n + 1):
                    ng = tuple(toks[i : i + n])
                    self.counts[n - 1][ng] += 1
                    if n >= 2:
                        self.cont_counts[ng[1:]].add(ng[0])
                        self.bigram_left[ng[-1]].add(ng[0])
            for i in range(1, len(toks)):
                self.total_bigrams += 1

    def _cont_prob(self, w: str) -> float:
        num = len(self.bigram_left.get(w, set()))
        denom = max(self.total_bigrams, 1)
        return num / denom

    def _abs_disc_prob(self, w: str, ctx: Tuple[str, ...]) -> float:
        """Absolute-discount + Kneser-Ney lower-order back-off."""
        if len(ctx) == 0:
            pc = self._cont_prob(w)
            return max(pc, 1e-8)
        n_ctx = self.counts[len(ctx) - 1].get(ctx, 0)
        if n_ctx == 0:
            return self._abs_disc_prob(w, ctx[1:])
        n_ctx_w = self.counts[len(ctx)].get(ctx + (w,), 0)
        followers = sum(1 for ng in self.counts[len(ctx)] if ng[:-1] == ctx)
        lam = self.discount * followers / n_ctx
        first = max(n_ctx_w - self.discount, 0) / n_ctx
        back = self._abs_disc_prob(w, ctx[1:])
        return first + lam * back

    def log_prob(self, w: str, ctx: Tuple[str, ...]) -> float:
        ctx = ctx[-(self.order - 1) :]
        return math.log(max(self._abs_disc_prob(w, ctx), 1e-12))

    def word_score(self, word: str, history: List[str]) -> float:
        return self.log_prob(word.lower(), tuple(h.lower() for h in history))

    def save(self, path: str) -> None:
        obj = {
            "order": self.order,
            "discount": self.discount,
            "counts": [{" ".join(k): v for k, v in c.items()} for c in self.counts],
            "cont_counts": {" ".join(k): list(v) for k, v in self.cont_counts.items()},
            "bigram_left": {k: list(v) for k, v in self.bigram_left.items()},
            "vocab": list(self.vocab),
            "total_bigrams": self.total_bigrams,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(obj, f)

    @classmethod
    def load(cls, path: str) -> "KneserNeyLM":
        with open(path) as f:
            obj = json.load(f)
        lm = cls(order=obj["order"], discount=obj["discount"])
        lm.counts = [Counter({tuple(k.split()): v for k, v in c.items()}) for c in obj["counts"]]
        lm.cont_counts = defaultdict(set, {tuple(k.split()): set(v) for k, v in obj["cont_counts"].items()})
        lm.bigram_left = defaultdict(set, {k: set(v) for k, v in obj["bigram_left"].items()})
        lm.vocab = set(obj["vocab"])
        lm.total_bigrams = obj["total_bigrams"]
        return lm


def build_from_file(corpus_path: str, out_path: str, order: int = 3) -> KneserNeyLM:
    lines = Path(corpus_path).read_text(encoding="utf-8").splitlines()
    lm = KneserNeyLM(order=order)
    lm.fit([ln for ln in lines if ln.strip()])
    lm.save(out_path)
    print(f"[ngram] vocab={len(lm.vocab)}  order={order}  → {out_path}")
    return lm


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "score"])
    ap.add_argument("--corpus", default="data/syllabus.txt")
    ap.add_argument("--lm", default="models/ngram.json")
    ap.add_argument("--order", type=int, default=3)
    ap.add_argument("--word", default=None)
    ap.add_argument("--ctx", nargs="*", default=[])
    args = ap.parse_args()
    if args.cmd == "build":
        build_from_file(args.corpus, args.lm, args.order)
    elif args.cmd == "score":
        lm = KneserNeyLM.load(args.lm)
        print("logP =", lm.word_score(args.word, args.ctx))
