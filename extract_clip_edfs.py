# Copy local clip EDFs from ./clip_edfs/ to ./clip_edfs_preprocessed/ with no
# filtering/resampling, but with polarity flipped (× −1) to match lab convention.
#
# By default the output clips are shuffled and named from ieeg_file_name +
# timestamp_sec (e.g. EMU1049_Day08_1_32621.8438.edf). Review order is
# randomized; clip_shuffle_key.csv records the mapping.
#
# Usage:
#   python extract_clip_edfs.py
#   python extract_clip_edfs.py --seed 42            # reproducible shuffle
#   python extract_clip_edfs.py --no-shuffle         # keep original names/order
#   python extract_clip_edfs.py --input-dir clip_edfs --output-dir clip_edfs_preprocessed
#
# !pip install pyedflib mne

import argparse
import csv
import os
import random
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pyedflib

KEY_CSV_NAME = "clip_shuffle_key.csv"

# Fixed default so reshuffles are reproducible (same blinded name -> same clip).
# Pass a different --seed to deliberately re-randomize.
DEFAULT_SEED = 42


def _format_clip_timestamp_sec(timestamp_sec: float) -> str:
    return f"{float(timestamp_sec):.4f}".rstrip("0").rstrip(".")


def output_name_for_clip(edf_path: Path, criteria_row: dict) -> str:
    """Output EDF basename from criteria or source filename (EMU…_time.edf)."""
    ieeg = (criteria_row.get("ieeg_file_name") or "").strip()
    ts_raw = (criteria_row.get("timestamp_sec") or "").strip()
    if ieeg and ts_raw:
        return f"{ieeg}_{_format_clip_timestamp_sec(float(ts_raw))}.edf"
    stem = edf_path.stem
    for prefix in ("spike_", "no_spike_", "midline_"):
        if stem.lower().startswith(prefix):
            return f"{stem[len(prefix):]}.edf"
    return edf_path.name


def clip_type_from_name(name: str) -> str:
    """Recover the clip label encoded in the source filename prefix."""
    stem = name.lower()
    if stem.startswith("no_spike_"):
        return "no_spike"
    if stem.startswith("spike_"):
        return "spike"
    if stem.startswith("midline_"):
        return "midline"
    return "unknown"

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

_REAL_STDERR = sys.stderr


def log(msg):
    print(msg, file=_REAL_STDERR, flush=True)


def read_edf_uv(edf_path: Path):
    """Read an EDF into (n_samples, n_channels) physical values plus labels/fs.

    Channels are required to share a single sample rate (true for the clip
    EDFs: 18 scalp channels at one fs). Physical values are returned in the
    file's native dimension, which for these clips is uV.
    """
    reader = pyedflib.EdfReader(str(edf_path))
    try:
        labels = list(reader.getSignalLabels())
        n_ch = reader.signals_in_file
        fs_values = {round(float(reader.getSampleFrequency(i)), 6) for i in range(n_ch)}
        if len(fs_values) != 1:
            raise ValueError(f"mixed sample rates in {edf_path.name}: {sorted(fs_values)}")
        fs = float(fs_values.pop())
        signals = [reader.readSignal(i) for i in range(n_ch)]
    finally:
        reader.close()

    n_samples = min(len(s) for s in signals)
    data = np.column_stack([s[:n_samples] for s in signals]).astype(np.float64)
    return data, labels, fs


def write_clip_edf(out_path: Path, data_uv: np.ndarray, labels: list[str], fs_int: int) -> None:
    """Write (n_samples, n_channels) uV data as an EDF+ file (no flipping)."""
    n_ch = data_uv.shape[1]
    headers = []
    for j, ch in enumerate(labels):
        col = data_uv[:, j]
        pmin = float(np.min(col))
        pmax = float(np.max(col))
        if pmin == pmax:
            pmin, pmax = pmin - 1.0, pmax + 1.0
        headers.append({
            "label": ch,
            "dimension": "uV",
            "sample_frequency": fs_int,
            "physical_min": pmin,
            "physical_max": pmax,
            "digital_min": -32768,
            "digital_max": 32767,
        })

    with pyedflib.EdfWriter(str(out_path), n_ch, file_type=pyedflib.FILETYPE_EDFPLUS) as writer:
        writer.setSignalHeaders(headers)
        writer.writeSamples(data_uv.T)


def process_one(edf_path: Path, out_path: Path, *, overwrite: bool) -> str:
    if out_path.exists() and not overwrite:
        return "exists"

    data, labels, fs = read_edf_uv(edf_path)

    # Pass through unchanged (no notch / bandpass / resample / Pz / reorder).
    fs_int = int(round(fs))
    n_samples = data.shape[0]
    trimmed_n = (n_samples // fs_int) * fs_int
    if trimmed_n == 0:
        raise ValueError(f"window shorter than 1 s ({n_samples} samples @ {fs_int} Hz)")
    if trimmed_n != n_samples:
        log(f"   trimmed {n_samples} -> {trimmed_n} samples (fs={fs_int})")
    segment = data[:trimmed_n]
    segment = segment * -1.0

    if not np.isfinite(segment).all():
        raise ValueError("non-finite samples in clip")

    write_clip_edf(out_path, segment.astype(np.float64), list(labels), fs_int)

    # Verify the file re-opens.
    try:
        reader = pyedflib.EdfReader(str(out_path))
        reader.close()
    except Exception as verify_exc:
        if out_path.exists():
            try:
                os.remove(out_path)
            except OSError:
                pass
        raise RuntimeError(f"pyedflib could not reopen output ({verify_exc!r})")

    return "saved"


def load_criteria(csv_path: Path) -> tuple[list[str], dict[str, dict]]:
    """Load the events CSV and index each row by clip EDF basename.

    Uses ``edf_filename`` when present; otherwise reconstructs
    ``<selection_type>_<ieeg_file_name>_<timestamp_sec:.4f>.edf``.
    """
    with open(csv_path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        columns = list(reader.fieldnames or [])
        by_basename: dict[str, dict] = {}
        for row in reader:
            edf_name = str(row.get("edf_filename", "")).strip()
            if edf_name:
                basename = edf_name if edf_name.lower().endswith(".edf") else f"{edf_name}.edf"
            else:
                sel = str(row.get("selection_type", "")).strip()
                ieeg = str(row.get("ieeg_file_name", "")).strip()
                ts_raw = str(row.get("timestamp_sec", "")).strip()
                try:
                    ts = float(ts_raw)
                except ValueError:
                    continue
                basename = f"{sel}_{ieeg}_{ts:.4f}.edf"
            by_basename[basename] = dict(row)
    return columns, by_basename


def main() -> int:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Copy local clip EDFs with optional blinded shuffle (no signal processing)."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=here / "clip_edfs",
        help="Folder of source clip EDFs (default: ./clip_edfs).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=here / "clip_edfs_preprocessed",
        help="Folder for output clip EDFs (default: ./clip_edfs_preprocessed).",
    )
    parser.add_argument(
        "--criteria-csv",
        type=Path,
        default=here / "selected_events_with_criteria.csv",
        help="CSV of per-event detail to merge into the shuffle key "
             "(default: ./selected_events_with_criteria.csv).",
    )
    parser.add_argument(
        "--no-shuffle",
        dest="shuffle",
        action="store_false",
        help="Keep original filenames/order instead of blinded clip_NNN.edf names.",
    )
    parser.set_defaults(shuffle=True)
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Seed for the shuffle (default {DEFAULT_SEED}, reproducible). "
             f"Pass a different value to re-randomize.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-process and overwrite outputs that already exist.",
    )
    args = parser.parse_args()

    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir

    if not input_dir.is_dir():
        sys.exit(f"Input directory not found: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load per-event criteria to enrich the shuffle key.
    criteria_columns: list[str] = []
    criteria_by_basename: dict[str, dict] = {}
    if args.criteria_csv and args.criteria_csv.is_file():
        criteria_columns, criteria_by_basename = load_criteria(args.criteria_csv)
        log(f"Loaded criteria for {len(criteria_by_basename)} events "
            f"from {args.criteria_csv.name}")
    else:
        log(f"WARN: criteria CSV not found ({args.criteria_csv}); "
            f"key will omit event detail.")

    edf_files = sorted(p for p in input_dir.glob("*.edf") if p.is_file())
    total = len(edf_files)
    if total == 0:
        sys.exit(f"No .edf files found in {input_dir}")

    # Build (source -> output name). When shuffling, only review order changes.
    sources = list(edf_files)
    if args.shuffle:
        random.Random(args.seed).shuffle(sources)
    assignments = [
        (
            src,
            output_name_for_clip(src, criteria_by_basename.get(src.name, {})),
        )
        for src in sources
    ]
    if args.shuffle:
        key_path = output_dir / KEY_CSV_NAME
        if key_path.exists() and not args.overwrite:
            sys.exit(
                f"{key_path} already exists. Re-run with --overwrite to reshuffle, "
                f"or use --no-shuffle."
            )
        for stale in output_dir.glob("*.edf"):
            stale.unlink()
    else:
        key_path = None

    log(f"Copying {total} clip(s) -> {output_dir} (polarity × −1, no filtering)")
    if args.shuffle:
        seed_msg = "random" if args.seed is None else f"seed={args.seed}"
        log(f"Shuffle ON ({seed_msg}); key -> {key_path}")

    batch_start = time.time()
    saved = existing = failed = 0
    key_rows: list[dict] = []

    for i, (edf_path, out_name) in enumerate(assignments, start=1):
        out_path = output_dir / out_name
        clip_t0 = time.time()
        try:
            result = process_one(edf_path, out_path, overwrite=args.overwrite)
        except Exception as exc:
            failed += 1
            log(f"[{i:>3}/{total}] FAIL {edf_path.name}: {type(exc).__name__}: {exc}")
            continue

        clip_dt = time.time() - clip_t0
        if result == "exists":
            existing += 1
            log(f"[{i:>3}/{total}] SKIP (exists) {out_name}")
        else:
            saved += 1
            size_kb = os.path.getsize(out_path) / 1024.0
            elapsed = time.time() - batch_start
            log(f"[{i:>3}/{total}] saved {out_name} <- {edf_path.name} "
                f"({size_kb:.1f} KB, {clip_dt:.1f}s, batch {elapsed/60:.1f} min)")

        criteria_row = criteria_by_basename.get(edf_path.name, {})
        key_row: dict = {
            "review_order": i,
            "output_name": out_name,
            "original_name": edf_path.name,
            "clip_type": clip_type_from_name(edf_path.name),
        }
        for col in criteria_columns:
            key_row[col] = criteria_row.get(col, "")
        key_rows.append(key_row)

    if key_path is not None and key_rows:
        reserved = {"review_order", "output_name", "original_name", "clip_type"}
        key_fieldnames = (
            ["review_order", "output_name", "original_name", "clip_type"]
            + [c for c in criteria_columns if c not in reserved]
        )
        with open(key_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=key_fieldnames)
            writer.writeheader()
            writer.writerows(key_rows)
        log(f"Wrote shuffle key: {key_path}")

    log(f"\nDone. Saved {saved}, skipped {existing} already-present, {failed} failed.")
    return 1 if failed and saved == 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
