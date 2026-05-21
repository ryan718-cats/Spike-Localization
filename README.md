# Spike clip review GUI (200 clips)

Manual review of `final_selected_200_events.csv`: ±7 s scalp EEG windows, banana montage, 10–20 schematic marking, trace-row clear, and prev/next navigation.

## Quick start

```powershell
cd c:\Users\ryanc\Downloads\Conrad_Lab\progress_sheet\Carlos_GUI
pip install -r requirements.txt
```
## Workflow

1. **`final_selected_200_events.csv`** — source of truth (200 clips). Channel assignment uses **`best_channel_1` only**.

2. **`build_clip_edfs.py`** — optional local EDF export.

3. **`run_spike_review_gui.py`** — opens the review GUI.

## Controls

| Control | Behavior |
|--------|----------|
| ↑ / ↓ | Amplitude (1.5× per press; range ~1e-6 to 1e6) |
| Clip dropdown | `001 EMU3005_Day01_1` … `200 …` (number + ieeg file name); each clip tracked internally by timestamp |
| 10–20 map | Click to add/remove individual contacts |
| Trace row click | Clear marks for that row's contacts (does not add marks) |
| Annotations | Auto-saved to `sz-gui/seizure_gui/annotations/<patient_id>/<ieeg_file_name>_spike_at_<timestamp_sec>.json` |
| Clip dropdown colors | Blue = saved annotation exists for that clip |
