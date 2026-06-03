#!/usr/bin/env python3
"""
Launch the spike-clip review GUI.

Default: blinded folder-scan review of pre-processed clips in
./clip_edfs_preprocessed/ (every clip_*.edf becomes a clip option, no CSV).

Usage:
    python run_spike_review_gui.py
    python run_spike_review_gui.py --clip-edf-dir D:\\path\\to\\clips
    # CSV-driven review (legacy), when a clips CSV is available:
    python run_spike_review_gui.py --clips-csv final_selected_200_events.csv \
        --clip-edf-dir clip_edfs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SZ_GUI_DIR = ROOT / "sz-gui" / "seizure_gui"
DEFAULT_EDF_DIR = ROOT / "clip_edfs_preprocessed"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SZ_GUI_DIR))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Spike clip review GUI (local EDF only)."
    )
    parser.add_argument(
        "--clip-edf-dir",
        type=Path,
        default=DEFAULT_EDF_DIR,
        help="Directory containing the clip EDFs to review (default: ./clip_edfs_preprocessed).",
    )
    parser.add_argument(
        "--clips-csv",
        type=Path,
        default=None,
        help="Optional clips CSV (e.g. final_selected_200_events.csv) for legacy "
        "CSV-driven review. When omitted, the clip folder is scanned directly.",
    )
    args = parser.parse_args()

    if not args.clip_edf_dir.is_dir():
        print(
            f"Clip EDF directory not found: {args.clip_edf_dir}\n"
            f"Run extract_clip_edfs.py first, or pass --clip-edf-dir.",
            file=sys.stderr,
        )
        return 2

    from sz_gui import main as gui_main

    if args.clips_csv is not None:
        if not args.clips_csv.exists():
            print(f"Clips CSV not found: {args.clips_csv}", file=sys.stderr)
            return 2
        sys.argv = [
            "sz_gui",
            "--clips_csv",
            str(args.clips_csv.resolve()),
            "--clip_edf_dir",
            str(args.clip_edf_dir.resolve()),
        ]
    else:
        # Blinded folder-scan review: no CSV, list every *.edf in the folder.
        sys.argv = [
            "sz_gui",
            "--clip_scan_dir",
            str(args.clip_edf_dir.resolve()),
        ]

    gui_main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
