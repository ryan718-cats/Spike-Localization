#!/usr/bin/env python3
"""
Export ±7 s scalp EEG clips as local EDF files from final_selected_200_events.csv.

Filenames use the GUI-compatible pattern:
  <ieeg_file_name>_spike_at_<timestamp_sec>.edf

Sidecar JSON records patient_id, epilepsy location/laterality, and spike channel
(best_channel_1 only).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from ieeg_clip_io import (
    CLIP_HALF_WINDOW_SEC,
    DEFAULT_CLIPS_CSV_NAME,
    build_selections_csv,
    clip_edf_basename,
    clip_gui_label,
    fetch_ieeg_window_uv,
    load_clips_csv,
    row_to_option,
    spike_channel_for_row,
    write_clip_edf,
)


def main() -> int:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Build local EDF clips for spike review GUI")
    parser.add_argument(
        "--csv",
        type=Path,
        default=root / DEFAULT_CLIPS_CSV_NAME,
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=root / "clip_edfs",
        help="Directory for exported EDF files",
    )
    parser.add_argument(
        "--username",
        default=os.environ.get("IEEG_USERNAME"),
        help="iEEG.org username (or set IEEG_USERNAME)",
    )
    parser.add_argument(
        "--password-env",
        default="IEEG_PASSWORD",
        help="Environment variable holding iEEG password",
    )
    parser.add_argument(
        "--password",
        default=None,
        help="iEEG password (prefer env var)",
    )
    parser.add_argument(
        "--half-window-sec",
        type=float,
        default=CLIP_HALF_WINDOW_SEC,
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip clips whose EDF already exists",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only first N rows (for testing)",
    )
    args = parser.parse_args()

    password = args.password or os.environ.get(args.password_env)
    if not args.username or not password:
        print(
            "Set --username and password via --password or "
            f"${args.password_env}",
            file=sys.stderr,
        )
        return 2

    df = load_clips_csv(args.csv)
    if args.limit:
        df = df.head(args.limit)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []
    failures: list[str] = []

    for i, row in tqdm(df.iterrows(), total=len(df), desc="EDF clips"):
        opt = row_to_option(row, int(i))
        edf_name = f"{clip_edf_basename(row)}.edf"
        edf_path = args.out_dir / edf_name
        meta_path = edf_path.with_suffix(".json")

        sidecar = {
            "gui_label": clip_gui_label(row, int(i)),
            "patient_id": opt["patient_id"],
            "epilepsy_location": opt["epilepsy_location"],
            "epilepsy_laterality": opt["epilepsy_laterality"],
            "clip_type": opt["clip_type"],
            "selection_group": opt["selection_group"],
            "best_channel_1": opt["best_channel_1"],
            "spikenet2_prob": opt.get("spikenet2_prob"),
            "ieeg_file_name": opt["ieeg_file_name"],
            "timestamp_sec": opt["spike_time_sec"],
            "center_sec": opt["spike_time_sec"],
            "half_window_sec": args.half_window_sec,
        }

        if args.skip_existing and edf_path.exists():
            manifest.append({**sidecar, "edf_path": str(edf_path), "status": "skipped"})
            continue

        try:
            data_uv, labels, fs = fetch_ieeg_window_uv(
                args.username,
                password,
                opt["dataset_id"],
                opt["spike_time_sec"],
                half_window_sec=args.half_window_sec,
            )
            write_clip_edf(edf_path, data_uv, labels, fs)
            meta_path.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
            manifest.append({**sidecar, "edf_path": str(edf_path), "status": "ok"})
        except Exception as exc:
            failures.append(f"{edf_name}: {exc}")
            manifest.append({**sidecar, "edf_path": str(edf_path), "status": f"error: {exc}"})

    manifest_path = args.out_dir / "clip_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    selections_path = root / "clip_selections.csv"
    build_selections_csv(df, selections_path)

    print(f"\nWrote {sum(1 for m in manifest if m.get('status') == 'ok')} EDFs to {args.out_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"GUI selections: {selections_path}")
    if failures:
        print(f"\n{len(failures)} failures:")
        for f in failures[:20]:
            print(f"  {f}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
