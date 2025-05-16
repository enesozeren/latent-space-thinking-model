#!/usr/bin/env python3
"""
check_contamination_midmatch.py

Detect potential overlap between

  • open-r1/OpenR1-Math-220k      (default split)
  • openai/gsm8k  –  main / test

The algorithm:
  1. Pick N random GSM-8K test questions (parameter --sample_size)
  2. Normalize text (lower-case, collapse whitespace)
  3. Extract the 'middle chunk' of each GSM question:
        • central <fraction> of characters  (parameter --mid_fraction)
        • but at least <min_chars> long     (parameter --min_chars)
  4. Check whether that middle chunk occurs verbatim
     in any OpenR1-Math question (Python substring match).
  5. Print every hit.

Dependencies
------------
    pip install datasets tqdm
"""

import argparse
import random
import re
from datasets import load_dataset
from pathlib import Path
from tqdm import tqdm


# ───────────────────────────── helpers ──────────────────────────────
def normalize(text: str) -> str:
    """Lower-case and collapse whitespace."""
    return re.sub(r"\s+", " ", text.strip().lower())


def middle_slice(text: str, fraction: float = 0.4, min_chars: int = 30) -> str:
    """
    Return the central *fraction* of the string (at least min_chars long).
    """
    text = text.strip()
    n = len(text)
    keep = max(int(n * fraction), min_chars)
    if keep >= n:
        return text
    start = (n - keep) // 2
    end = start + keep
    return text[start:end]


# ───────────────────────────── script ───────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Check GSM-8K ↔ OpenR1-Math overlap")
    p.add_argument(
        "-n",
        "--sample_size",
        type=int,
        default=1300,
        help="How many GSM-8K *test* questions to examine (default: 50)",
    )
    p.add_argument(
        "--mid_fraction",
        type=float,
        default=0.5,
        help="Fraction of each GSM question (central part) to search for (default: 0.4)",
    )
    p.add_argument(
        "--min_chars",
        type=int,
        default=30,
        help="Ensure the middle chunk is at least this long (default: 30)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling (default: 42)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print("⏬  Loading datasets from the Hugging Face Hub …")
    gsm_ds = load_dataset("openai/gsm8k", "main", split="test")
    math_ds = load_dataset("open-r1/OpenR1-Math-220k", split="train")  # default

    print(f"   GSM-8K test size : {len(gsm_ds):,}")
    print(f"   OpenR1-Math size: {len(math_ds):,}\n")

    # ── sample GSM-8K test questions ─────────────────────────────────────────
    random.seed(args.seed)
    indices = random.sample(range(len(gsm_ds)), min(args.sample_size, len(gsm_ds)))
    gsm_sample = gsm_ds.select(indices)

    # ── normalize OpenR1-Math questions once up-front ────────────────────────
    math_questions_norm = [normalize(row["problem"]) for row in math_ds]

    print(
        f"🔍  Searching {len(gsm_sample)} GSM-8K questions "
        f"against {len(math_questions_norm):,} OpenR1-Math rows …"
    )
    hits = []

    for gsm_row in tqdm(gsm_sample, unit="gsm q", ncols=75):
        gsm_q_norm = normalize(gsm_row["question"])
        mid = middle_slice(
            gsm_q_norm, fraction=args.mid_fraction, min_chars=args.min_chars
        )

        # Skip very short mids (may happen if the original question is tiny)
        if len(mid) < args.min_chars:
            continue

        # substring search
        for math_q_norm in math_questions_norm:
            if mid in math_q_norm:
                hits.append(
                    {
                        "gsm_question": gsm_row["question"],
                        "gsm_mid": mid,
                        "openr1_question": math_q_norm,
                    }
                )
                break  # stop at first match for this GSM question

    # ── report ─────────────────────────────────────────────────────────────
    print(f"\n✨  Found {len(hits)} potential overlaps\n")

    for h in hits:
        print("─" * 78)
        print("[GSM-8K]  (middle slice marked with «»)")
        # highlight the slice in the GSM text so you can see what matched
        highlighted = h["gsm_question"].replace(h["gsm_mid"], f"«{h['gsm_mid']}»")
        print(highlighted)
        print("\n[OpenR1-Math]")
        print(h["openr1_question"])
        print()

    if not hits:
        print("No overlaps detected with the chosen parameters.")


if __name__ == "__main__":
    main()
