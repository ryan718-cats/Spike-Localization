"""
Shared helpers for SpikeNet2 clip review: iEEG download, EDF export, filenames, CSV manifest.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

CORE_EEG_CHANNELS = [
    "C3", "C4", "Cz", "F3", "F4", "F7", "F8", "Fp1", "Fp2",
    "Fz", "O1", "O2", "P3", "P4", "Pz", "T3", "T4", "T5", "T6",
]

CLIP_HALF_WINDOW_SEC = 7.0
DEFAULT_CLIPS_CSV_NAME = "final_selected_200_events.csv"


def timeseries_sample_hz(details) -> float:
    """iEEG.org: sample_frequency replaced deprecated sample_rate."""
    if hasattr(details, "sample_frequency"):
        return float(details.sample_frequency)
    return float(details.sample_rate)


def sanitize_token(value: str, max_len: int = 40) -> str:
    s = re.sub(r"[^\w\-.]+", "_", str(value).strip())
    return s[:max_len] if s else "unknown"


def spike_channel_for_row(row: pd.Series) -> str:
    """Use best_channel_1 only (ignore best_channel_2)."""
    ch = row.get("best_channel_1")
    if isinstance(ch, str) and ch.strip():
        return ch.strip()
    assigned = row.get("assigned_channel")
    if isinstance(assigned, str) and assigned.strip():
        return assigned.strip()
    return "unknown"


def clip_event_stem(
    ieeg_file_name: str,
    timestamp_sec: float,
    *,
    split_token: str = "_spike_at_",
) -> str:
    """Stable id for a clip: <ieeg_file_name>_spike_at_<timestamp_sec>."""
    ieeg = str(ieeg_file_name).strip()
    t = float(timestamp_sec)
    return f"{ieeg}{split_token}{t:.6f}".rstrip("0").rstrip(".")


def clip_edf_basename(row: pd.Series, *, split_token: str = "_spike_at_") -> str:
    """
    EDF filename stem compatible with sz_gui selections.csv parsing:
      <ieeg_file_name>_spike_at_<timestamp_sec>
    Metadata is encoded in the GUI label / sidecar JSON, not the basename.
    """
    return clip_event_stem(
        str(row["ieeg_file_name"]),
        float(row["timestamp_sec"]),
        split_token=split_token,
    )


def clip_annotation_relpath(row: pd.Series) -> Path:
    """annotations/<patient_id>/<ieeg_file_name>_spike_at_<timestamp_sec>.json"""
    pid = sanitize_token(str(row.get("patient_id", "unknown")))
    stem = clip_edf_basename(row)
    return Path(pid) / f"{stem}.json"


def clip_timestamp_display(row: pd.Series) -> str:
    """Timestamp for GUI labels, in seconds."""
    return f"{float(row['timestamp_sec']):.3f}"


def clip_gui_label(row: pd.Series, index: int | None = None) -> str:
    """Display name: clip number (1–200) + iEEG file stem."""
    num = f"{int(index) + 1:03d}" if index is not None else "???"
    ieeg = str(row["ieeg_file_name"]).strip()
    return f"{num} {ieeg}"


def row_to_option(row: pd.Series, index: int) -> dict[str, Any]:
    ieeg = str(row["ieeg_file_name"]).strip()
    dataset_id = ieeg if ieeg.lower().endswith(".edf") else f"{ieeg}.edf"
    basename = clip_edf_basename(row)
    return {
        "dataset_id": dataset_id,
        "spike_time_sec": float(row["timestamp_sec"]),
        "annotation_stem": basename,
        "annotation_relpath": str(clip_annotation_relpath(row)),
        "recording_name": f"{basename}.edf",
        "clip_type": str(row.get("clip_type", "")),
        "selection_group": str(row.get("selection_group", "")),
        "patient_id": row.get("patient_id"),
        "epilepsy_location": row.get("epilepsy_location"),
        "epilepsy_laterality": row.get("epilepsy_laterality"),
        "best_channel_1": spike_channel_for_row(row),
        "assigned_channel": row.get("assigned_channel"),
        "spikenet2_prob": row.get("spikenet2_prob"),
        "best_prob_1": row.get("best_prob_1"),
        "confidence_priority": row.get("confidence_priority"),
        "selected_from_preferred_confidence_range": row.get(
            "selected_from_preferred_confidence_range"
        ),
        "ieeg_file_name": ieeg,
        "timestamp_hhmmss": row.get("timestamp_hhmmss"),
        "row_index": index,
        "left_spike_time_sec": row.get("left_spike_time_sec"),
        "right_spike_time_sec": row.get("right_spike_time_sec"),
        "gap_sec": row.get("gap_sec"),
        "left_spike_channel": row.get("left_spike_channel"),
        "right_spike_channel": row.get("right_spike_channel"),
    }


def fetch_ieeg_window_uv(
    username: str,
    password: str,
    dataset_id: str,
    center_sec: float,
    half_window_sec: float = CLIP_HALF_WINDOW_SEC,
    channels: list[str] | None = None,
) -> tuple[np.ndarray, list[str], float]:
    """Download referential scalp EEG segment; return (n_samples, n_ch) µV, labels, fs."""
    from ieeg.auth import Session

    channels = channels or CORE_EEG_CHANNELS
    dataset_id = dataset_id.strip()
    window_start = max(0.0, float(center_sec) - float(half_window_sec))
    window_end = float(center_sec) + float(half_window_sec)
    duration_sec = window_end - window_start
    if duration_sec <= 0:
        raise ValueError(f"Invalid window for {dataset_id} at {center_sec}s")

    chunk_sec = 60.0
    raw_chunks: list[np.ndarray] = []

    with Session(username, password) as session:
        try:
            ds = session.open_dataset(dataset_id)
        except Exception:
            ds = session.open_dataset(dataset_id.removesuffix(".edf"))

        all_labels = list(ds.get_channel_labels())
        if not all_labels:
            raise ValueError(f"No channels in {dataset_id}")

        first_details = ds.get_time_series_details(all_labels[0])
        fs = timeseries_sample_hz(first_details)
        vcf = float(getattr(first_details, "voltage_conversion_factor", 1.0))

        selected = [ch for ch in channels if ch in all_labels]
        if not selected:
            raise ValueError(f"No core EEG channels in {dataset_id}")
        channel_ids = list(ds.get_channel_indices(selected))

        start_usec = int(window_start * 1e6)
        stop_usec = int(window_end * 1e6)
        chunk_usec = int(chunk_sec * 1e6)
        clip_start = start_usec
        while clip_start < stop_usec:
            dur = min(chunk_usec, stop_usec - clip_start)
            chunk = np.asarray(ds.get_data(clip_start, dur, channel_ids), dtype=np.float32)
            chunk = np.nan_to_num(chunk, nan=0.0, posinf=0.0, neginf=0.0)
            raw_chunks.append(chunk)
            clip_start += dur

    raw = np.concatenate(raw_chunks, axis=0)
    # Volts → µV (polarity flip applied at display in sz_gui)
    data_uv = raw * vcf * 1e6
    return data_uv.astype(np.float32), selected, fs


def write_clip_edf(
    path: Path,
    data_uv: np.ndarray,
    channel_labels: list[str],
    fs: float,
) -> None:
    import pyedflib

    path.parent.mkdir(parents=True, exist_ok=True)
    n_ch = data_uv.shape[1]
    headers = []
    for ch in channel_labels:
        col = data_uv[:, channel_labels.index(ch)]
        pmin = float(np.min(col))
        pmax = float(np.max(col))
        if pmin == pmax:
            pmin, pmax = pmin - 1.0, pmax + 1.0
        headers.append({
            "label": ch,
            "dimension": "uV",
            "sample_rate": int(fs),
            "physical_min": pmin,
            "physical_max": pmax,
            "digital_min": -32768,
            "digital_max": 32767,
        })

    with pyedflib.EdfWriter(str(path), n_ch, file_type=pyedflib.FILETYPE_EDFPLUS) as writer:
        writer.setSignalHeaders(headers)
        writer.writeSamples(data_uv.T)


def build_selections_csv(clips_df: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    """Write sz_gui-compatible selections.csv (Filename column)."""
    rows = []
    for i, row in clips_df.iterrows():
        rows.append({"Filename": f"{clip_edf_basename(row)}.edf", "Response": ""})
    out = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    return out


def load_clips_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {
        "clip_type", "ieeg_file_name", "timestamp_sec", "patient_id",
        "epilepsy_location", "epilepsy_laterality", "selection_group",
        "best_channel_1",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {sorted(missing)}")
    return df
