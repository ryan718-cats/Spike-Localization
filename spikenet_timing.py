"""
SpikeNet2 score timeline vs saved spike timestamps.

Saved CSV ``timestamp_sec`` values include the detection pipeline delay:

    spike_time_sec = chunk_start_sec + peak_idx * STEP_SEC + SPIKENET_DELAY_SEC

Do NOT align saved times to ``start_sec + arange(n) * STEP_SEC`` (off by 0.5 s).
"""

from __future__ import annotations

import numpy as np

SPIKENET_FQ = 128
SPIKENET_STEP_SAMPLES = 8
SPIKENET_STEP_SEC = SPIKENET_STEP_SAMPLES / SPIKENET_FQ  # 0.0625 s
SPIKENET_DELAY_SEC = 0.5

# Default buffer when re-running SpikeNet on a short clip (filter edge effects).
RECHECK_BUFFER_SEC = 30.0


def sn2_times_sec(chunk_start_sec: float, n_scores: int) -> np.ndarray:
    """
    Wall-clock time (s) for each SpikeNet2 score index.

    Matches saved spike timestamps from the original detector.
    """
    n = int(n_scores)
    if n <= 0:
        return np.array([], dtype=np.float64)
    return (
        float(chunk_start_sec)
        + np.arange(n, dtype=np.float64) * SPIKENET_STEP_SEC
        + SPIKENET_DELAY_SEC
    )


def model_window_times_sec(chunk_start_sec: float, n_scores: int) -> np.ndarray:
    """Score times without SPIKENET_DELAY_SEC (equivalent to saved_time - 0.5)."""
    n = int(n_scores)
    if n <= 0:
        return np.array([], dtype=np.float64)
    return float(chunk_start_sec) + np.arange(n, dtype=np.float64) * SPIKENET_STEP_SEC


def index_for_saved_timestamp(
    sn2: np.ndarray,
    chunk_start_sec: float,
    saved_time_sec: float,
) -> int:
    """Index of the SN2 score whose timeline is closest to ``saved_time_sec``."""
    if sn2.size == 0:
        raise ValueError("empty SN2 trace")
    times = sn2_times_sec(chunk_start_sec, len(sn2))
    return int(np.argmin(np.abs(times - float(saved_time_sec))))


def prob_at_saved_timestamp(
    sn2: np.ndarray,
    chunk_start_sec: float,
    saved_time_sec: float,
) -> tuple[float, int]:
    """
    Probability at the saved event time (correct +0.5 s alignment).

    Equivalent to::

        model_time = saved_time_sec - SPIKENET_DELAY_SEC
        idx = argmin(|model_window_times - model_time|)
    """
    idx = index_for_saved_timestamp(sn2, chunk_start_sec, saved_time_sec)
    return float(sn2[idx]), idx


def peak_index_to_saved_time(chunk_start_sec: float, peak_idx: int) -> float:
    """Convert SpikeNet peak index to CSV ``timestamp_sec``."""
    return (
        float(chunk_start_sec)
        + int(peak_idx) * SPIKENET_STEP_SEC
        + SPIKENET_DELAY_SEC
    )
