# Spike clip review GUI (200 clips)

Manual review of `final_selected_200_events.csv`: ±7 s scalp EEG windows, referential or banana montage, 10–20 schematic marking, trace-row clear, and prev/next navigation.

## Quick start

```powershell
cd c:\Users\ryanc\Downloads\Conrad_Lab\progress_sheet\Carlos_GUI
pip install -r requirements.txt
```
## Workflow

1. **`final_selected_200_events.csv`** — source of truth (200 clips). Channel assignment uses **`best_channel_1` only**.

2. **`build_clip_edfs.py`** — optional local EDF export.

3. **`run_spike_review_gui.py`** — opens the review GUI.

## SpikeNet2 timestamps (`timestamp_sec` / `spikenet2_prob`)

Saved spike times include a **+0.5 s** detection delay:

`timestamp_sec = chunk_start_sec + peak_idx × (8/128) + 0.5`

When re-running SpikeNet2 and comparing to `spikenet2_prob`, use `spikenet_timing.prob_at_saved_timestamp()` (or add **+0.5 s** to the score time axis). Comparing without the delay can pick the wrong index and a very different probability.

Recheck script (optional, needs iEEG credentials + model):

```powershell
python recheck_spikenet2_probs.py --buffer-sec 30 --limit 5
```

Uses ±30 s around each event by default to reduce filter edge effects vs short clips.

## Controls

| Control | Behavior |
|--------|----------|
| ↑ / ↓ | Amplitude (1.5× per press; range ~1e-6 to 1e6) |
| Montage | **As recorded** — referential channels as loaded; **Banana** — double-banana chain pairs (default) |
| Clip dropdown | `001 EMU3005_Day01_1` … `200 …` (number + ieeg file name); each clip tracked internally by timestamp |
| 10–20 map | Click to add/remove individual contacts |
| Trace row click | Clear marks for that row's contacts (does not add marks) |
| Annotations | Auto-saved to `sz-gui/seizure_gui/annotations/<patient_id>/<ieeg_file_name>_spike_at_<timestamp_sec>.json` |
| Clip dropdown colors | Blue = saved annotation exists for that clip |
