"""
iEEG.org download + scalp preprocessing (lab notebook pipeline).

Matches: notch 60 Hz → HP 0.5 Hz → downsample to 128 Hz → synthetic Pz → new_channel_order.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch, resample, sosfilt

from ieeg_clip_io import timeseries_sample_hz

# 18 referential channels (no Pz in download)
CHANNELS_TO_INCLUDE = [
    "C3", "C4", "Cz", "F3", "F4", "F7", "F8", "Fp1", "Fp2",
    "Fz", "O1", "O2", "P3", "P4", "T3", "T4", "T5", "T6",
]

CURRENT_CHANNEL_ORDER = [
    "C3", "C4", "Cz", "F3", "F4", "F7", "F8", "Fp1", "Fp2",
    "Fz", "O1", "O2", "P3", "P4", "T3", "T4", "T5", "T6", "Pz",
]

NEW_CHANNEL_ORDER = [
    "Fp1", "F3", "C3", "P3", "F7", "T3", "T5", "O1", "Fz",
    "Cz", "Pz", "Fp2", "F4", "C4", "P4", "F8", "T4", "T6", "O2",
]

# Pz = mean of these columns in 18-ch layout (same indices as lab script on downsampled_data)
PZ_SOURCE_INDICES = [2, 12, 13, 10, 11]

TARGET_FS = 128
NOTCH_HZ = 60.0
HP_CUTOFF_HZ = 0.5
HP_ORDER = 4
DOWNSAMPLE_LP_ORDER = 5


def get_ieeg_data(
    username: str,
    password: str,
    ieeg_filename: str,
    start_time_usec: int,
    stop_time_usec: int,
    select_electrodes: list[str] | None = None,
) -> tuple[pd.DataFrame, int]:
    """Download a window from iEEG.org (60 s chunks if needed)."""
    from ieeg.auth import Session

    start_time_usec = int(start_time_usec)
    stop_time_usec = int(stop_time_usec)
    duration = stop_time_usec - start_time_usec
    select_electrodes = select_electrodes or CHANNELS_TO_INCLUDE

    with Session(username, password) as session:
        try:
            ds = session.open_dataset(ieeg_filename)
        except Exception:
            ds = session.open_dataset(ieeg_filename.removesuffix(".edf"))

        all_channel_labels = list(ds.get_channel_labels())
        if not isinstance(select_electrodes[0], str):
            raise ValueError("select_electrodes must be a list of strings.")

        channel_ids = [i for i, e in enumerate(all_channel_labels) if e in select_electrodes]
        if not channel_ids:
            raise ValueError("None of the requested channels were in the dataset.")
        channel_names = [all_channel_labels[i] for i in channel_ids]

        try:
            data = ds.get_data(start_time_usec, duration, channel_ids)
        except Exception:
            clip_size = int(60 * 1e6)
            clip_start = start_time_usec
            chunks: list[np.ndarray] = []
            while clip_start + clip_size < stop_time_usec:
                chunks.append(ds.get_data(clip_start, clip_size, channel_ids))
                clip_start += clip_size
            chunks.append(
                ds.get_data(clip_start, stop_time_usec - clip_start, channel_ids)
            )
            data = np.concatenate(chunks, axis=0)

        ch0 = channel_names[0]
        details = ds.get_time_series_details(ch0)
        fs = int(timeseries_sample_hz(details))

    return pd.DataFrame(np.asarray(data), columns=channel_names), fs


def notch_filter(data: np.ndarray, hz: float, fs: float) -> np.ndarray:
    b, a = iirnotch(hz, Q=30, fs=fs)
    return filtfilt(b, a, data, axis=0)


def high_pass_filter(
    data: np.ndarray, cutoff: float, fs: float, order: int = HP_ORDER
) -> np.ndarray:
    nyquist = 0.5 * fs
    b, a = butter(order, cutoff / nyquist, btype="high", analog=False)
    return filtfilt(b, a, data, axis=0)


def downsample_with_filter(
    data: np.ndarray,
    original_fs: int,
    target_fs: int = TARGET_FS,
    cutoff: float | None = None,
    order: int = DOWNSAMPLE_LP_ORDER,
) -> np.ndarray:
    num_samples, _ = data.shape
    downsample_factor = original_fs // target_fs
    if downsample_factor < 2:
        raise ValueError(
            f"Downsampling factor must be at least 2. original_fs={original_fs}, "
            f"target_fs={target_fs}"
        )
    if cutoff is None:
        cutoff = target_fs / 2.0
    sos = butter(order, cutoff / (0.5 * original_fs), btype="low", output="sos")
    filtered = sosfilt(sos, data, axis=0)
    return resample(filtered, num_samples // downsample_factor, axis=0)


def _align_columns(
    segment: np.ndarray, channel_names: list[str]
) -> tuple[np.ndarray, list[str]]:
    """Column order expected by Pz indices and reorder step (18 ch, no Pz)."""
    order = [ch for ch in CURRENT_CHANNEL_ORDER if ch != "Pz"]
    missing = [ch for ch in order if ch not in channel_names]
    if missing:
        raise ValueError(f"Missing channels for lab layout: {missing}")
    idx = [channel_names.index(ch) for ch in order]
    return segment[:, idx], order


def preprocess_scalp_segment(
    segment: np.ndarray,
    channel_names: list[str],
    fs: int,
    *,
    target_fs: int = TARGET_FS,
) -> tuple[np.ndarray, list[str], float]:
    """
    Lab pipeline: notch → HP → downsample → synthetic Pz → new_channel_order.

    Returns (n_samples, n_channels) float32, channel names, output sample rate.
    """
    segment = np.asarray(segment, dtype=np.float64)
    if np.isnan(segment).any():
        segment = np.nan_to_num(segment, nan=0.0)

    segment, _ = _align_columns(segment, list(channel_names))

    segment = notch_filter(segment, NOTCH_HZ, fs)
    segment = high_pass_filter(segment, HP_CUTOFF_HZ, fs)

    out_fs = float(fs)
    if fs > target_fs:
        if fs // target_fs >= 2:
            segment = downsample_with_filter(segment, fs, target_fs)
            out_fs = float(target_fs)
        else:
            n_out = int(round(segment.shape[0] * target_fs / fs))
            segment = resample(segment, n_out, axis=0)
            out_fs = float(target_fs)

    pz_mean = np.mean(segment[:, PZ_SOURCE_INDICES], axis=1, keepdims=False)
    with_pz = np.column_stack((segment, pz_mean))

    reorder_index = [CURRENT_CHANNEL_ORDER.index(ch) for ch in NEW_CHANNEL_ORDER]
    reordered = with_pz[:, reorder_index]

    return reordered.astype(np.float32), list(NEW_CHANNEL_ORDER), out_fs


def load_and_preprocess_window(
    username: str,
    password: str,
    ieeg_filename: str,
    start_time_usec: int,
    stop_time_usec: int,
    select_electrodes: list[str] | None = None,
) -> tuple[np.ndarray, list[str], float]:
    """get_iEEG_data + preprocess_scalp_segment."""
    df, fs = get_ieeg_data(
        username,
        password,
        ieeg_filename,
        start_time_usec,
        stop_time_usec,
        select_electrodes,
    )
    return preprocess_scalp_segment(df.values, list(df.columns), fs)
