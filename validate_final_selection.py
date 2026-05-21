#!/usr/bin/env python3
"""Validate final_selected_200_events.csv against selection rules."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from ieeg_clip_io import DEFAULT_CLIPS_CSV_NAME, load_clips_csv, spike_channel_for_row

MIDLINE_CHANNELS = {"Fz", "Cz", "Pz"}
NON_MIDLINE_SPIKE_CHANNELS = {
    "Fp1", "F3", "C3", "P3", "F7", "T3", "T5", "O1",
    "Fp2", "F4", "C4", "P4", "F8", "T4", "T6", "O2",
}
EXPECTED_GROUPS = {
    "no_spike": 30,
    "midline": 10,
    **{f"spike_{ch}": 10 for ch in sorted(NON_MIDLINE_SPIKE_CHANNELS)},
}


def validate(df: pd.DataFrame) -> list[str]:
    errors: list[str] = []

    if len(df) != 200:
        errors.append(f"Row count is {len(df)}, expected 200")

    counts = df["selection_group"].value_counts().to_dict()
    for grp, expected in EXPECTED_GROUPS.items():
        got = counts.get(grp, 0)
        if got != expected:
            errors.append(f"selection_group {grp}: got {got}, expected {expected}")

    spike_mid = df[df["clip_type"].isin(["spike", "midline"])]
    for _, row in spike_mid.iterrows():
        ch1 = spike_channel_for_row(row)
        assigned = row.get("assigned_channel")
        if isinstance(assigned, str) and assigned.strip() and assigned.strip() != ch1:
            errors.append(
                f"Row {row.name}: assigned_channel {assigned!r} != best_channel_1 {ch1!r}"
            )
        if row["clip_type"] == "midline" and ch1 not in MIDLINE_CHANNELS:
            errors.append(f"Row {row.name}: midline clip but best_channel_1={ch1}")
        if row["clip_type"] == "spike" and ch1 in MIDLINE_CHANNELS:
            errors.append(f"Row {row.name}: non-midline spike but best_channel_1={ch1}")

    sm = spike_mid.copy()
    sm["_ts"] = sm["timestamp_sec"].astype(float)
    for fname, grp in sm.groupby("ieeg_file_name"):
        times = sorted(grp["_ts"].tolist())
        for i in range(len(times) - 1):
            if times[i + 1] - times[i] < 20.0:
                errors.append(
                    f"{fname}: spike/midline times {times[i]} and {times[i+1]} "
                    f"within 20s (should be >20s apart, ±10s rule)"
                )

    no_spike = df[df["clip_type"] == "no_spike"]
    for _, row in no_spike.iterrows():
        gap = row.get("gap_sec")
        if pd.notna(gap) and float(gap) < 30.0:
            errors.append(f"Row {row.name}: gap_sec={gap} < 30")
        left = row.get("left_spike_time_sec")
        right = row.get("right_spike_time_sec")
        ts = row.get("timestamp_sec")
        if pd.notna(left) and pd.notna(right) and pd.notna(ts):
            mid = (float(left) + float(right)) / 2.0
            if abs(mid - float(ts)) > 0.01:
                errors.append(
                    f"Row {row.name}: timestamp_sec {ts} != midpoint {mid}"
                )

    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path(__file__).resolve().parent / DEFAULT_CLIPS_CSV_NAME,
    )
    args = parser.parse_args()
    df = load_clips_csv(args.csv)
    errors = validate(df)
    if errors:
        print(f"FAILED: {len(errors)} issue(s)")
        for e in errors[:50]:
            print(f"  - {e}")
        if len(errors) > 50:
            print(f"  ... and {len(errors) - 50} more")
        return 1
    print("OK: final_selected_200_events.csv passes all checks.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
