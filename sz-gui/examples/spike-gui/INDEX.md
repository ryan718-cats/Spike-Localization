# Spike Annotation GUI - Complete Index

This directory contains a complete spike annotation system for manual review and marking of spikes in iEEG data by clinicians.

## File Structure

```
spike-gui/
├── spike_annotation_gui.py       # Main GUI application
├── create_annotation_edfs.py     # Generate EDF files from patient data
├── analyze_annotations.py        # Analyze completed annotations
├── validate_edfs.py              # Validate EDF file integrity
├── run_gui.sh                    # Quick launcher script
├── requirements.txt              # Python dependencies
├── README.md                     # Full documentation
├── QUICKSTART.md                 # Quick start guide
├── INDEX.md                      # This file
├── annotation_progress.json      # Progress tracking (auto-generated)
└── annotations/                  # Annotation output directory (auto-generated)
    └── *.json                    # Individual annotation files
```

## Components

### 1. Data Preparation

**`create_annotation_edfs.py`**
- Identifies stim-recording pairs with non-zero spike rates
- Downloads iEEG data (10s pre + 30s during + 11s post stimulation)
- Applies filtering (60Hz notch + 1-70Hz bandpass)
- Converts to bipolar montage
- Saves as EDF files for annotation

**Usage:**
```bash
python create_annotation_edfs.py --n_files 200 --output_dir /path/to/output
```

**Key Features:**
- Automatically selects pairs with highest spike activity
- Handles multiple iEEG file formats
- Channel name formatting (adds leading zeros)
- Comprehensive error handling

---

### 2. GUI Application

**`spike_annotation_gui.py`**
- Interactive spike annotation interface
- Real-time visualization with PyQt5 and pyqtgraph
- Bipolar montage display
- Interpolated vs raw signal toggle
- Gain adjustment
- Progress tracking and resumption

**Usage:**
```bash
python spike_annotation_gui.py --edf_dir /path/to/edfs
```

**Key Features:**
- **Display**: 15-second visible window, 51-second total
- **Interaction**: Click to mark/unmark spikes
- **Navigation**: Scrollbar, keyboard shortcuts
- **Persistence**: Auto-save, resume from last position
- **Visualization**: 
  - Red dashed lines for stimulation times
  - Blue triangle markers for annotated spikes
  - Channel labels on y-axis
  - Time relative to stimulation on x-axis

**Keyboard Shortcuts:**
- `Enter`: Save and next file
- `↑/↓`: Adjust gain
- `←/→`: Scroll left/right

---

### 3. Analysis Tools

**`analyze_annotations.py`**
- Loads all completed annotations
- Computes summary statistics
- Generates visualizations
- Exports CSV summary

**Usage:**
```bash
python analyze_annotations.py --annotations_dir ./annotations --output_dir ./analysis
```

**Outputs:**
- `annotation_statistics.pdf/png`: Distribution plots
- `temporal_distribution.pdf/png`: Spike timing histogram
- `annotation_summary.csv`: Detailed summary table
- Console output with statistics

**`validate_edfs.py`**
- Validates EDF file integrity
- Checks data structure
- Identifies corrupted files
- Reports statistics

**Usage:**
```bash
python validate_edfs.py --edf_dir /path/to/edfs
```

**Checks:**
- File readability
- Duration (~51 seconds expected)
- Channel count (≥2 expected)
- Sampling rate (100-2000 Hz)
- Data validity (no NaNs)
- Channel names

---

### 4. Utilities

**`run_gui.sh`**
- Quick launcher script
- Automatically finds EDF directory
- Provides helpful error messages

**Usage:**
```bash
bash run_gui.sh
```

**`requirements.txt`**
- Python package dependencies
- Versions specified for reproducibility

**Installation:**
```bash
pip install -r requirements.txt
```

**Dependencies:**
- PyQt5: GUI framework
- pyqtgraph: Fast plotting
- mne: EEG data I/O
- numpy, scipy: Numerical operations
- pandas: Data management
- tqdm: Progress bars

---

## Workflow

### Complete Annotation Pipeline

```
1. Prepare Data
   ↓
   create_annotation_edfs.py
   ↓
   [200 EDF files created]

2. Validate Files (optional)
   ↓
   validate_edfs.py
   ↓
   [Confirmation of file integrity]

3. Annotate Spikes
   ↓
   spike_annotation_gui.py
   ↓
   [Manual spike marking by clinician]
   ↓
   [Progress auto-saved in annotation_progress.json]
   ↓
   [Annotations saved in annotations/*.json]

4. Analyze Results
   ↓
   analyze_annotations.py
   ↓
   [Statistics, plots, and CSV output]
```

### Typical Session

1. **First time:**
   ```bash
   pip install -r requirements.txt
   python create_annotation_edfs.py --n_files 200
   ```

2. **Daily annotation:**
   ```bash
   bash run_gui.sh
   # or
   python spike_annotation_gui.py --edf_dir ../results/spike_annotation_edfs
   ```

3. **After completing annotations:**
   ```bash
   python analyze_annotations.py
   ```

---

## Data Formats

### Input: Patient Data
- **Pre-post merged data**: `pre_post_merged_data-10s-seg.csv`
- **Spike detections**: 
  - `baseline_pre_stim/HUP*_spike_output.csv`
  - `baseline_post_stim/HUP*_spike_output.csv`
  - `stim-spike-detection-10s/HUP*_spike_output.csv`
- **Metadata**: 
  - `master_pt_list_erin.xlsx`
  - `first_stim_times_per_channel.csv`

### Intermediate: EDF Files
- **Format**: European Data Format (EDF)
- **Structure**: 
  - Channels: Bipolar montage (sequential differencing)
  - Duration: 51 seconds (10 pre + 30 stim + 11 post)
  - Filtering: 60Hz notch + 1-70Hz bandpass
- **Naming**: `HUP{patient}_{stim_ch}_{recording_ch}.edf`

### Output: Annotations
- **Format**: JSON
- **Structure**:
  ```json
  {
    "edf_file": "path/to/file.edf",
    "timestamp": "2024-10-08T12:34:56",
    "n_channels": 10,
    "channel_names": ["LA1-LA2", "LA2-LA3", ...],
    "sampling_rate": 512,
    "spikes": [[channel_idx, time_in_seconds], ...],
    "n_spikes": 42
  }
  ```
- **Naming**: `{edf_filename}_annotations.json`

### Progress Tracking
- **File**: `annotation_progress.json`
- **Structure**:
  ```json
  {
    "completed_files": ["/path/to/file1.edf", ...],
    "last_updated": "2024-10-08T12:34:56"
  }
  ```

---

## Technical Details

### Signal Processing

**Bipolar Montage:**
- Implemented in `tools/iEEG_helper_functions.py::automatic_bipolar_montage()`
- Sequential differencing: Ch1-Ch2, Ch2-Ch3, etc.
- Enhances local features, reduces common-mode noise

**Interpolation:**
- Simulates stitched segments from `stim-spike_detection-10s.py`
- 0.06-second gaps at each stimulation pulse
- Linear interpolation between gap boundaries
- 30 gaps total (one per 1Hz stimulation pulse)

**Filtering:**
- Notch filter: 60Hz, 120Hz, 180Hz (Q=30)
- Bandpass: 1-70Hz (4th order Butterworth)
- Applied before bipolar montage and EDF export

### GUI Architecture

**Backend:**
- PyQt5 for windowing and controls
- pyqtgraph for high-performance plotting
- MNE for EDF I/O

**Display:**
- Fixed 15-second visible window
- Scrollable 51-second total range
- Multiple channels with vertical offset
- Real-time gain adjustment

**Interaction:**
- Mouse click detection via scene coordinates
- Channel mapping via y-coordinate
- Spike proximity detection (0.2s threshold)
- Keyboard event handling for shortcuts

**Data Management:**
- Lazy loading of EDF files
- On-demand interpolation
- JSON serialization for annotations
- Set-based progress tracking

---

## Customization

### Adjusting Parameters

**In `spike_annotation_gui.py`:**
```python
# Line ~46-51: Display parameters
self.visible_duration = 15.0      # Visible window (seconds)
self.total_duration = 51.0        # Total duration
self.spike_click_threshold = 0.2  # Spike detection radius (seconds)
```

**In `create_annotation_edfs.py`:**
```python
# Line ~281-282: Time window
start_time = stim_time - 10.0  # Pre-stim duration
end_time = stim_time + 41.0    # Post-stim duration (30 + 11)
```

### Adding Features

**New visualization options:**
- Modify `update_plot()` in `spike_annotation_gui.py`
- Add to control panel in `initUI()`

**Custom analysis:**
- Create new script based on `analyze_annotations.py`
- Load JSON files from `annotations/` directory
- Access spike data: `data['spikes']` is list of `[channel_idx, time]`

---

## Troubleshooting

### Common Issues

**"Import Error: No module named X"**
- Solution: `pip install -r requirements.txt`

**"No EDF files found"**
- Solution: Run `create_annotation_edfs.py` first
- Check `--edf_dir` path is correct

**"Could not apply bipolar montage"**
- GUI falls back to raw channels
- May occur with non-standard channel naming
- Check `automatic_bipolar_montage()` in `iEEG_helper_functions.py`

**"GUI is slow"**
- Reduce number of visible channels
- Increase `visible_duration` for fewer updates
- Close other applications

**"Annotations not saving"**
- Check write permissions in `annotations/` directory
- Verify disk space
- Check console for error messages

---

## Testing

### Verify Installation

```bash
# Test imports
python -c "import PyQt5, pyqtgraph, mne, numpy, pandas; print('OK')"

# Validate EDF files
python validate_edfs.py --edf_dir /path/to/edfs

# Test GUI (no EDF files needed)
python spike_annotation_gui.py
```

### Sample Workflow Test

```bash
# Create 5 test files
python create_annotation_edfs.py --n_files 5 --output_dir ./test_edfs

# Validate them
python validate_edfs.py --edf_dir ./test_edfs

# Annotate
python spike_annotation_gui.py --edf_dir ./test_edfs
# (Mark a few spikes, press Enter to save)

# Analyze
python analyze_annotations.py --annotations_dir ./annotations
```

---

## References

### Related Code

- **Spike detection**: `code/spike_detection_scripts/stim-spike_detection-10s.py`
- **Validation**: `code/change-validation/change_validation_10s.py`
- **Helper functions**: `tools/iEEG_helper_functions.py`
- **Configuration**: `code/config.py`

### Data Sources

- **iEEG data**: iEEG.org via ieeg.auth.Session
- **Patient metadata**: `data/pt-metadata/master_pt_list_erin.xlsx`
- **Spike detections**: `results/spike_detection_derivatives/`
- **Pre-post data**: `datasets/pre_post_merged_data-10s-seg.csv`

---

## Version History

**v1.0 (2024-10-08)**
- Initial release
- Core annotation GUI
- EDF creation pipeline
- Analysis tools
- Documentation

---

## Contact & Support

For issues or questions:
1. Check QUICKSTART.md and README.md
2. Review this INDEX.md for technical details
3. Examine console output for error messages
4. Check annotation_progress.json for state

---

*Last updated: 2024-10-08*

