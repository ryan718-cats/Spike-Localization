#!/usr/bin/env python3
"""
Re-run SpikeNet2 on buffered iEEG clips and compare to CSV spikenet2_prob.

Uses +0.5 s delay alignment (see spikenet_timing.py). Short ±7 s clips can
differ from full-EDF runs due to filter edges; default buffer is 30 s each side.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ieeg_clip_io import DEFAULT_CLIPS_CSV_NAME, load_clips_csv
from spikenet_timing import (
    RECHECK_BUFFER_SEC,
    prob_at_saved_timestamp,
    sn2_times_sec,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Recheck spikenet2_prob with correct timing")
    parser.add_argument("--csv", type=Path, default=Path(__file__).resolve().parent / DEFAULT_CLIPS_CSV_NAME)
    parser.add_argument("--buffer-sec", type=float, default=RECHECK_BUFFER_SEC)
    parser.add_argument("--prob-tol", type=float, default=0.08, help="Max |new - saved| to count as match")
    parser.add_argument("--limit", type=int, default=0, help="Max rows (0 = all spike/midline)")
    parser.add_argument("--show-wrong-alignment", action="store_true",
                        help="Also print error if comparing without +0.5 s delay")
    args = parser.parse_args()

    if not os.environ.get("IEEG_USERNAME") or not os.environ.get("IEEG_PASSWORD"):
        print("Set IEEG_USERNAME and IEEG_PASSWORD.", file=sys.stderr)
        return 2

    # Heavy imports (model) only when running recheck.
    from example_loading import predict_sn2_for_ieeg_window

    df = load_clips_csv(args.csv)
    work = df[df["clip_type"].isin(["spike", "midline"])].copy()
    if args.limit > 0:
        work = work.head(args.limit)

    mismatches: list[str] = []
    wrong_align_examples: list[str] = []
    ok = 0
    skipped = 0

    for i, row in work.iterrows():
        ieeg = str(row["ieeg_file_name"]).strip()
        saved_t = float(row["timestamp_sec"])
        saved_prob = float(row["spikenet2_prob"])
        buf = float(args.buffer_sec)
        start_sec = max(0.0, saved_t - buf)
        end_sec = saved_t + buf

        try:
            sn2, chunk_start = predict_sn2_for_ieeg_window(ieeg, start_sec, end_sec)
        except Exception as exc:
            skipped += 1
            print(f"SKIP row {i} {ieeg}: {exc}")
            continue

        new_prob, idx = prob_at_saved_timestamp(sn2, chunk_start, saved_t)
        delta = abs(new_prob - saved_prob)
        if delta <= args.prob_tol:
            ok += 1
        else:
            mismatches.append(
                f"row {i} {ieeg} t={saved_t:.3f}: saved={saved_prob:.4f} "
                f"recheck={new_prob:.4f} (idx={idx}, |Δ|={delta:.4f})"
            )

        if args.show_wrong_alignment and sn2.size:
            times_no_delay = chunk_start + np.arange(len(sn2)) * (8 / 128)
            bad_idx = int(np.argmin(np.abs(times_no_delay - saved_t)))
            bad_prob = float(sn2[bad_idx])
            if abs(bad_prob - saved_prob) < abs(new_prob - saved_prob):
                wrong_align_examples.append(
                    f"row {i}: no-delay align prob={bad_prob:.4f} closer to saved than "
                    f"correct {new_prob:.4f}"
                )

    print(f"Compared {ok + len(mismatches)} clips ({skipped} skipped).")
    print(f"  Match (|Δ| <= {args.prob_tol}): {ok}")
    print(f"  Mismatch: {len(mismatches)}")
    for line in mismatches[:25]:
        print(f"  - {line}")
    if len(mismatches) > 25:
        print(f"  ... and {len(mismatches) - 25} more")
    if wrong_align_examples:
        print("\nWithout +0.5 s delay looked closer (timing bug indicator):")
        for line in wrong_align_examples[:10]:
            print(f"  - {line}")

    print(
        "\nNote: recheck uses ±buffer iEEG + lab preprocess; full-EDF pipeline may "
        "still differ slightly from original export."
    )
    return 0 if not mismatches else 1


if __name__ == "__main__":
    raise SystemExit(main())
