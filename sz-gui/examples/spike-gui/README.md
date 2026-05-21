# Spike Annotation GUI

A graphical user interface for manual spike annotation by clinicians. This tool displays iEEG traces with pre-stimulation, during stimulation, and post-stimulation periods for visual spike marking.

## Features

- **iEEG.org Time Display**: Shows the actual stimulation timestamp on ieeg.org for cross-reference (see [IEEG_TIME_DISPLAY.md](IEEG_TIME_DISPLAY.md))
- **Bipolar Montage Display**: Shows data in bipolar montage for better spike visualization
- **Interpolated vs Raw Signal**: Toggle between interpolated (what spike detector sees) and raw signal
- **Gain Adjustment**: Use up/down arrow keys to adjust display gain for better visibility
- **Interactive Spike Marking**: Click on traces to mark spikes, click again nearby to remove
- **Sequential Processing**: Automatically moves through EDF files in sequence
- **Progress Tracking**: Automatically resumes from the first uncompleted file
- **Scrollable Display**: View 15 seconds at a time with scrollbar for full 51-second window
- **Keyboard Shortcuts**:
  - `Enter`: Save annotations and move to next file
  - `↑/↓`: Increase/decrease gain
  - `←/→`: Scroll left/right

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Step 1: Create EDF Files for Annotation

First, create the EDF files from the patient data:

```bash
python create_annotation_edfs.py --n_files 200
```

This will:
- Identify stim-recording pairs with non-zero spike rates
- Load iEEG data for each pair (10s pre + 30s stim + 11s post)
- Apply filtering (60Hz notch + 1-70Hz bandpass)
- Convert to bipolar montage
- Save as EDF files in `results/spike_annotation_edfs/`

Options:
- `--n_files`: Number of files to create (default: 200)
- `--output_dir`: Custom output directory

### Step 2: Run the Annotation GUI

Launch the GUI to start annotating:

```bash
python spike_annotation_gui.py --edf_dir /path/to/edfs
```

Or select the directory from the GUI menu after launching:

```bash
python spike_annotation_gui.py
```

### Step 3: Annotate Spikes

1. **Mark Spikes**: Click on the trace where you see a spike
2. **Remove Spikes**: Click near an existing spike marker to remove it
3. **Adjust View**: 
   - Use up/down arrows to change gain
   - Use the scrollbar to navigate through the 51-second recording
   - Toggle "Show Interpolated Signal" to see raw vs interpolated data
4. **Save and Continue**: Press Enter to save annotations and move to next file

### Step 4: Review Annotations

Annotations are saved in JSON format in the `annotations/` subdirectory:
- Each file is named: `{original_edf_name}_annotations.json`
- Contains spike times, channel indices, and metadata
- Progress is tracked in `annotation_progress.json`

## Data Structure

### EDF Files
Each EDF file contains:
- **Time 0-10s**: Pre-stimulation baseline
- **Time 10-40s**: During stimulation (30 1Hz pulses)
- **Time 40-51s**: Post-stimulation period

Red dashed lines indicate stimulation times (at 10s, 11s, 12s, ... 39s).

### Annotation Files
Each annotation JSON contains:
```json
{
  "edf_file": "path/to/file.edf",
  "timestamp": "2024-10-08T...",
  "patient_id": 211,
  "stim_channel": "LF1-LF2",
  "ieeg_stim_time_seconds": 13401.304687,
  "n_channels": 10,
  "channel_names": ["LA1-LA2", "LA2-LA3", ...],
  "sampling_rate": 512,
  "spikes": [[channel_idx, time_in_seconds], ...],
  "n_spikes": 42
}
```

## Workflow Details

### Bipolar Montage
The data is displayed in bipolar montage (sequential channel differencing) which:
- Enhances local signal features
- Reduces common-mode artifacts
- Matches standard clinical EEG review practices

### Interpolated vs Raw Signal
- **Interpolated**: Shows what the automated spike detector sees after stitching segments
  - Gaps of 0.06s are linearly interpolated at each stimulation pulse
  - This matches the preprocessing in `stim-spike_detection-10s.py`
- **Raw**: Shows the original data without interpolation
  - Use this to verify whether spikes occur in real data or artifact regions

### Progress Tracking
- The GUI automatically tracks which files have been completed
- On restart, it resumes from the first uncompleted file
- You can also use Previous/Next buttons to navigate manually

## Troubleshooting

### "No EDF files found"
- Make sure you ran `create_annotation_edfs.py` first
- Check that the `--edf_dir` path is correct

### "Could not apply bipolar montage"
- The GUI will fall back to raw channel display
- This may happen if channel naming is non-standard

### "Failed to load file"
- Check that the EDF file is not corrupted
- Verify that MNE can read the file format

## Output

After annotation, you'll have:
1. **Annotations**: JSON files with spike times and channels
2. **Progress**: Record of completed files
3. **Statistics**: Spike counts per file available in annotations

These annotations can be used to:
- Validate automated spike detection algorithms
- Create ground truth datasets
- Compare manual vs automated spike detection

## Notes

- The GUI is optimized for files with 10-20 channels after bipolar montage
- Larger files may be slower to render
- All annotations are saved automatically when moving to the next file
- You can re-annotate files by selecting them manually (Previous button)

