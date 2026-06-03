"""
Clip window timing for spike review.

Legacy exports used symmetric ±7 s around ``timestamp_sec``, so the saved
SpikeNet time (delay included) sits at **7.0 s** in a 14 s clip.

Review targets the **7–8 s** band. Use asymmetric export (7.5 s pre / 6.5 s post)
or shift legacy symmetric clips by +0.5 s when building ``clip_edfs_preprocessed``.
"""

CLIP_EXPORT_PRE_SEC = 7.5
CLIP_EXPORT_POST_SEC = 6.5

CLIP_REVIEW_BAND_START_SEC = 7.0
CLIP_REVIEW_BAND_END_SEC = 8.0

# Symmetric ±7 s clips: shift waveform later so t=7.0 → t=7.5 in the file.
CLIP_LEGACY_SYMMETRIC_SHIFT_SEC = 0.5
