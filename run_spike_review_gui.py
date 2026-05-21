#!/usr/bin/env python3
"""
Launch spike-clip review GUI for final_selected_200_events.csv.

Recommended (password not on command line / process list):
  $env:IEEG_USERNAME = "your_user"
  $env:IEEG_PASSWORD = "your_password"
  python run_spike_review_gui.py

Or:  .\\launch_gui.ps1
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from ieeg_clip_io import DEFAULT_CLIPS_CSV_NAME

ROOT = Path(__file__).resolve().parent
SZ_GUI_DIR = ROOT / "sz-gui" / "seizure_gui"
DEFAULT_CSV = ROOT / DEFAULT_CLIPS_CSV_NAME

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SZ_GUI_DIR))


def main() -> int:
    parser = argparse.ArgumentParser(description="Spike clip review GUI (200 clips)")
    parser.add_argument("--clips-csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument(
        "--clip-edf-dir",
        type=Path,
        default=None,
        help="Optional: load clip EDFs from this folder when present",
    )
    parser.add_argument(
        "--ieeg-username",
        default=None,
        help="iEEG.org username (or set IEEG_USERNAME env var)",
    )
    parser.add_argument(
        "--ieeg-password",
        default=None,
        help="Discouraged: prefer IEEG_PASSWORD env var (not passed to child argv)",
    )
    parser.add_argument(
        "--no-auto-edf-dir",
        action="store_true",
        help="Do not auto-use ./clip_edfs even if it exists",
    )
    args = parser.parse_args()

    if not args.clips_csv.exists():
        print(f"Clips CSV not found: {args.clips_csv}", file=sys.stderr)
        return 2

    username = args.ieeg_username or os.environ.get("IEEG_USERNAME")
    password = args.ieeg_password or os.environ.get("IEEG_PASSWORD")
    if not username:
        print(
            "Missing iEEG username. Set IEEG_USERNAME or pass --ieeg-username.",
            file=sys.stderr,
        )
        return 2
    if not password:
        print(
            "Missing iEEG password. Set IEEG_PASSWORD env var "
            "(do not commit passwords to scripts).",
            file=sys.stderr,
        )
        return 2

    # Keep password out of subprocess argv (visible in `ps` / Task Manager).
    os.environ["IEEG_PASSWORD"] = password

    clip_edf_dir = args.clip_edf_dir
    if clip_edf_dir is None and not args.no_auto_edf_dir:
        default_edf = ROOT / "clip_edfs"
        if default_edf.is_dir():
            clip_edf_dir = default_edf

    from sz_gui import main as gui_main

    argv = [
        "sz_gui",
        "--clips_csv",
        str(args.clips_csv.resolve()),
        "--ieeg_username",
        username,
        "--ieeg_password_env",
        "IEEG_PASSWORD",
    ]
    if clip_edf_dir is not None:
        argv.extend(["--clip_edf_dir", str(clip_edf_dir.resolve())])

    sys.argv = argv
    gui_main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
