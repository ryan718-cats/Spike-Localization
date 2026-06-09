# Build blinded review clips from ./clip_edfs/ → ./clip_edfs_preprocessed/.
#
# Input (clip_edfs/):
#   - *.edf  (spike_ / no_spike_ / midline_ prefixed exports, 14 s each)
#   - selected_events_with_criteria.csv  (one row per EDF; source key)
#
# Output (clip_edfs_preprocessed/):
#   - Blinded EDFs named <ieeg_file_name>_<timestamp_sec>.edf (shuffled order)
#   - clip_shuffle_key.csv  (review_order, output_name, original_name, metadata)
#
# Processing: +0.5 s time shift (7–8 s spike band), polarity × −1, no filtering.
#
# Usage:
#   python extract_clip_edfs.py
#   python extract_clip_edfs.py --overwrite          # replace existing outputs
#   python extract_clip_edfs.py --seed 42            # reproducible shuffle
#   python extract_clip_edfs.py --no-shuffle         # keep original names/order
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

from clip_timing import CLIP_LEGACY_SYMMETRIC_SHIFT_SEC

KEY_CSV_NAME = "clip_shuffle_key.csv"
INPUT_KEY_CSV_NAME = "selected_events_with_criteria.csv"

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


def shift_clip_to_review_band(
    data: np.ndarray, fs: int, shift_sec: float = CLIP_LEGACY_SYMMETRIC_SHIFT_SEC
) -> np.ndarray:
    """Shift symmetric ±7 s clips so SpikeNet time moves from 7.0 s → 7.5 s (7–8 s band)."""
    n = int(round(float(shift_sec) * fs))
    if n <= 0 or data.shape[0] <= n:
        return data
    pad = np.zeros((n, data.shape[1]), dtype=data.dtype)
    return np.vstack([pad, data[:-n, :]])


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
    segment = shift_clip_to_review_band(segment, fs_int)
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


def criteria_basename(row: dict) -> str | None:
    """Resolve the source EDF filename for a criteria row."""
    edf_name = str(row.get("edf_filename", "")).strip()
    if edf_name:
        return edf_name if edf_name.lower().endswith(".edf") else f"{edf_name}.edf"
    sel = str(row.get("selection_type", row.get("clip_type", ""))).strip()
    ieeg = str(row.get("ieeg_file_name", "")).strip()
    ts_raw = str(row.get("timestamp_sec", "")).strip()
    if not (sel and ieeg and ts_raw):
        return None
    try:
        ts = float(ts_raw)
    except ValueError:
        return None
    return f"{sel}_{ieeg}_{ts:.4f}.edf"


def load_criteria(csv_path: Path) -> tuple[list[str], dict[str, dict]]:
    """Load selected_events_with_criteria.csv indexed by source EDF basename."""
    with open(csv_path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        columns = list(reader.fieldnames or [])
        by_basename: dict[str, dict] = {}
        for row in reader:
            basename = criteria_basename(row)
            if basename:
                by_basename[basename] = dict(row)
    return columns, by_basename


def validate_inputs(
    input_dir: Path,
    edf_files: list[Path],
    criteria_by_basename: dict[str, dict],
) -> None:
    """Ensure every EDF has criteria metadata and vice versa."""
    edf_names = {p.name for p in edf_files}
    csv_names = set(criteria_by_basename)
    missing_edf = sorted(csv_names - edf_names)
    missing_csv = sorted(edf_names - csv_names)
    errors: list[str] = []
    if missing_edf:
        errors.append(
            f"{len(missing_edf)} CSV row(s) have no matching EDF in {input_dir} "
            f"(e.g. {missing_edf[0]})"
        )
    if missing_csv:
        errors.append(
            f"{len(missing_csv)} EDF(s) in {input_dir} are missing from the key CSV "
            f"(e.g. {missing_csv[0]})"
        )
    if errors:
        raise SystemExit("\n".join(errors))


def build_shuffle_key_row(
    review_order: int,
    edf_path: Path,
    out_name: str,
    criteria_row: dict,
    criteria_columns: list[str],
) -> dict:
    clip_type = (
        (criteria_row.get("clip_type") or criteria_row.get("selection_type") or "")
        .strip()
        or clip_type_from_name(edf_path.name)
    )
    key_row: dict = {
        "review_order": review_order,
        "output_name": out_name,
        "original_name": edf_path.name,
        "clip_type": clip_type,
    }
    for col in criteria_columns:
        key_row[col] = criteria_row.get(col, "")
    if not str(key_row.get("clip_type", "")).strip():
        key_row["clip_type"] = clip_type
    if not str(key_row.get("selection_type", "")).strip() and clip_type:
        key_row["selection_type"] = clip_type
    return key_row


def clear_output_clips(output_dir: Path) -> None:
    for stale in output_dir.glob("*.edf"):
        stale.unlink()
    key_path = output_dir / KEY_CSV_NAME
    if key_path.is_file():
        key_path.unlink()


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
        default=None,
        help=f"Source key CSV (default: <input-dir>/{INPUT_KEY_CSV_NAME}).",
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
    if args.criteria_csv is None:
        args.criteria_csv = input_dir / INPUT_KEY_CSV_NAME

    if not input_dir.is_dir():
        sys.exit(f"Input directory not found: {input_dir}")
    if not args.criteria_csv.is_file():
        sys.exit(
            f"Key CSV not found: {args.criteria_csv}\n"
            f"Expected {INPUT_KEY_CSV_NAME} inside {input_dir}."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    criteria_columns, criteria_by_basename = load_criteria(args.criteria_csv)
    log(
        f"Loaded {len(criteria_by_basename)} event(s) from "
        f"{args.criteria_csv.name}"
    )

    edf_files = sorted(p for p in input_dir.glob("*.edf") if p.is_file())
    total = len(edf_files)
    if total == 0:
        sys.exit(f"No .edf files found in {input_dir}")
    validate_inputs(input_dir, edf_files, criteria_by_basename)

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
    key_path = output_dir / KEY_CSV_NAME if args.shuffle else None
    if args.shuffle and key_path is not None and key_path.exists() and not args.overwrite:
        sys.exit(
            f"{key_path} already exists. Re-run with --overwrite to reshuffle, "
            f"or use --no-shuffle."
        )
    if args.overwrite and args.shuffle:
        clear_output_clips(output_dir)

    log(
        f"Copying {total} clip(s) -> {output_dir} "
        f"(+{CLIP_LEGACY_SYMMETRIC_SHIFT_SEC}s time shift for 7–8 s band, polarity × −1, no filtering)"
    )
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

        criteria_row = criteria_by_basename[edf_path.name]
        key_rows.append(
            build_shuffle_key_row(
                i, edf_path, out_name, criteria_row, criteria_columns
            )
        )

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
