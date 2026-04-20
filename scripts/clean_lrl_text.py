"""Clean repetitive garbage from the Maithili translation (outputs/lrl.txt).

NLLB-200 sometimes degenerates into repeating short tokens (\"(हाँ) (हाँ)...\"
or \". . .\"). We collapse such runs before feeding XTTS, otherwise XTTS
wastes inference time looping through the repeat.

Writes `outputs/lrl_clean.txt`.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path


def collapse_repeats(text: str, min_run: int = 5) -> str:
    """Keep most content; collapse:
    - identical whitespace-separated tokens repeating ≥ `min_run` times,
    - hyphen/en-dash separated repeats of the same short Devanagari group
      (e.g. `जे-जे-जे-...`),
    - paren-repeated tokens like `(हाँ) (हाँ) ...`,
    - extreme single-character runs within a word.
    """
    # Collapse hyphen/en-dash glued Devanagari repeats: (X-){4,} -> X-X
    text = re.sub(
        r"((?:[\u0900-\u097Fa-zA-Z]{1,8}[-\u2013]){4,})",
        lambda m: "-".join(m.group(1).strip("-").split("-")[:2]) + "-",
        text,
    )
    # whitespace-token run collapse
    toks = re.findall(r"\S+", text)
    out: list[str] = []
    i = 0
    while i < len(toks):
        run_end = i + 1
        while run_end < len(toks) and toks[run_end] == toks[i]:
            run_end += 1
        n = run_end - i
        if n >= min_run and len(toks[i]) <= 10:
            out.extend([toks[i]] * 2)
        else:
            out.extend(toks[i:run_end])
        i = run_end
    cleaned = " ".join(out)

    cleaned = re.sub(r"(\s*([।!?.]\s*)){5,}", ". . ", cleaned)
    cleaned = re.sub(r"(\([^\(\)]{1,12}\)\s*){4,}", r"\1\1", cleaned)
    cleaned = re.sub(r"(.)\1{6,}", r"\1\1\1", cleaned)
    return cleaned.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_path", default="outputs/lrl.txt")
    ap.add_argument("--out_path", default="outputs/lrl_clean.txt")
    args = ap.parse_args()

    text = Path(args.in_path).read_text(encoding="utf-8").strip()
    out = collapse_repeats(text)
    Path(args.out_path).write_text(out, encoding="utf-8")
    print(f"[clean-lrl] in={len(text)}  out={len(out)}  saved {args.out_path}")


if __name__ == "__main__":
    main()
