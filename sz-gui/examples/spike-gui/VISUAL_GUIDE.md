# Visual Guide - Spike Annotation GUI

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    SPIKE ANNOTATION SYSTEM                       │
└─────────────────────────────────────────────────────────────────┘

┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐
│  Patient Data    │      │  iEEG.org Data   │      │ Spike Detections │
│  - CSV files     │      │  - Raw signals   │      │  - Pre baseline  │
│  - Metadata      │      │  - Channel info  │      │  - Post baseline │
└────────┬─────────┘      └────────┬─────────┘      │  - During stim   │
         │                         │                 └────────┬─────────┘
         │                         │                          │
         └─────────────┬───────────┴──────────────────────────┘
                       │
                       ▼
            ┌──────────────────────┐
            │ create_annotation_   │
            │      edfs.py         │
            │                      │
            │ • Identify pairs     │
            │ • Download data      │
            │ • Apply filters      │
            │ • Bipolar montage    │
            │ • Save as EDF        │
            └──────────┬───────────┘
                       │
                       ▼
            ┌──────────────────────┐
            │   200 EDF Files      │
            │                      │
            │ 10s pre + 30s stim   │
            │     + 11s post       │
            └──────────┬───────────┘
                       │
          ┌────────────┼────────────┐
          │            │            │
          ▼            ▼            ▼
   ┌───────────┐  ┌───────────┐  ┌───────────────┐
   │ validate_ │  │  spike_   │  │   Clinician   │
   │  edfs.py  │  │annotation_│  │    Reviews    │
   │           │  │  gui.py   │  │               │
   │ Check     │  │           │  │ • Visual      │
   │ integrity │  │ Interactive│  │ • Mark spikes │
   └───────────┘  │  marking  │  │ • Validate    │
                  └─────┬─────┘  └───────────────┘
                        │
                        ▼
                ┌───────────────┐
                │  Annotations  │
                │  (JSON files) │
                └───────┬───────┘
                        │
                        ▼
                ┌───────────────┐
                │  analyze_     │
                │annotations.py │
                │               │
                │ • Statistics  │
                │ • Plots       │
                │ • CSV export  │
                └───────┬───────┘
                        │
                        ▼
                ┌───────────────┐
                │    Results    │
                │               │
                │ • Plots (PDF) │
                │ • Summary CSV │
                └───────────────┘
```

## GUI Layout

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Spike Annotation GUI                                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  File: HUP213_LA1-LA2_LE3.edf        Progress: 42/200  [Select Dir]     │
│                                                                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  [✓] Show Interpolated Signal     Gain: 1.50x  (Use ↑↓ arrows)          │
│                                           Spikes marked: 15              │
│                                                                           │
├─────────────────────────────────────────────────────────────────────────┤
│                        iEEG Trace Display                                │
│  │                                                                        │
│  │ LA1-LA2  ─────▼─────────────────────────────────────                 │
│  │                |                                                      │
│  │ LA2-LA3  ──────────────────────────▼─────────────                    │
│  │                                     |                                 │
│  │ LA3-LA4  ───────────────────────────────────                         │
│  │                                                                       │
│  │ LA4-LA5  ────────▼──────────────────────────                         │
│  │                  |                                                    │
│  │          ...     |        ▼ = Marked Spike                           │
│  │                  |        | = Red stim line                           │
│  │                                                                       │
│  └─────┬────────────┬────────────┬────────────┬──────────              │
│       -5s          0s           5s          10s       ...               │
│               (Stimulation onset)                                       │
│                                                                          │
├─────────────────────────────────────────────────────────────────────────┤
│  Scroll: [░░░░░▓▓▓▓░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  ←→          │
│                                                                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Click to mark spikes • Click near existing to remove • ↑↓ gain         │
│                                        [← Previous]  [Save & Next →]    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Data Timeline

```
          Pre-Stim           During Stimulation              Post-Stim
        ═══════════════════════════════════════════════════════════════
        
        -10s              0s                           30s          41s
         │                │                             │            │
         │◄───10 sec────►│◄──────30 seconds──────────►│◄──11 sec──►│
         │                │                             │            │
         │   Baseline     │    30 pulses @ 1Hz         │  Baseline  │
         │                │                             │            │
         │                ↓                             ↓            │
         │                │ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ...        │            │
         │                │ (stimulation pulses)        │            │
         │                │                             │            │
         ▼                ▼                             ▼            ▼
    ┌────────────────────────────────────────────────────────────────┐
    │         Blue triangles (▼) mark annotated spikes               │
    │         Red lines (|) mark stimulation times                   │
    └────────────────────────────────────────────────────────────────┘
```

## Keyboard Shortcuts

```
╔══════════════════════════════════════════════════════════╗
║                    KEYBOARD CONTROLS                      ║
╠══════════════════════════════════════════════════════════╣
║                                                           ║
║   ↑  UP ARROW      →   Increase gain (bigger signals)    ║
║                                                           ║
║   ↓  DOWN ARROW    →   Decrease gain (smaller signals)   ║
║                                                           ║
║   ←  LEFT ARROW    →   Scroll left                       ║
║                                                           ║
║   →  RIGHT ARROW   →   Scroll right                      ║
║                                                           ║
║   ⏎  ENTER         →   Save annotations & next file      ║
║                                                           ║
║   🖱  CLICK         →   Mark spike (or remove if near)    ║
║                                                           ║
╚══════════════════════════════════════════════════════════╝
```

## Workflow Steps

```
┌─────────────────────────────────────────────────────────────┐
│                      DAILY WORKFLOW                          │
└─────────────────────────────────────────────────────────────┘

1. Launch GUI
   ▼
   bash run_gui.sh
   
2. GUI loads first uncompleted file
   ▼
   Display shows 15 seconds of data
   
3. Adjust viewing parameters
   ▼
   • Press ↑/↓ to adjust gain
   • Toggle interpolated signal checkbox
   • Use scrollbar to see full recording
   
4. Mark spikes
   ▼
   • Click on trace where spike occurs
   • Blue triangle (▼) appears
   • Spike count updates
   
5. Remove mistakes
   ▼
   • Click near existing marker
   • Marker disappears
   
6. Review full recording
   ▼
   • Scroll through entire 51 seconds
   • Check all channels
   • Mark all visible spikes
   
7. Save and continue
   ▼
   • Press Enter
   • Annotations saved to JSON
   • Next file loads automatically
   
8. Repeat 3-7 until complete
   ▼
   Progress tracked in annotation_progress.json
   
9. Close GUI when done
   ▼
   Next session resumes from first uncompleted file
```

## File Organization

```
spike-gui/
├── 📄 Python Scripts
│   ├── spike_annotation_gui.py       ← Main GUI application
│   ├── create_annotation_edfs.py     ← Generate EDF files
│   ├── analyze_annotations.py        ← Analyze results
│   └── validate_edfs.py              ← Check file integrity
│
├── 🚀 Launch Scripts
│   ├── run_gui.sh                    ← Quick launcher
│   └── workflow.sh                   ← Complete workflow
│
├── 📚 Documentation
│   ├── README.md                     ← Full documentation
│   ├── QUICKSTART.md                 ← Quick start guide
│   ├── INDEX.md                      ← Technical reference
│   ├── SUMMARY.md                    ← Implementation summary
│   └── VISUAL_GUIDE.md               ← This file
│
├── ⚙️  Configuration
│   └── requirements.txt              ← Python dependencies
│
└── 📁 Generated (auto-created)
    ├── annotation_progress.json      ← Progress tracking
    ├── annotations/                  ← Annotation output
    │   └── *.json                    ← Individual annotations
    └── annotation_analysis/          ← Analysis results
        ├── annotation_statistics.pdf
        ├── temporal_distribution.pdf
        └── annotation_summary.csv
```

## Signal Processing Flow

```
Raw iEEG Signal
      ↓
┌─────────────────┐
│ Notch Filter    │  Remove 60Hz, 120Hz, 180Hz
│ (60Hz + harm.)  │
└────────┬────────┘
         ↓
┌─────────────────┐
│ Bandpass Filter │  Keep 1-70 Hz
│ (1-70 Hz)       │
└────────┬────────┘
         ↓
┌─────────────────┐
│ Bipolar Montage │  Ch1-Ch2, Ch2-Ch3, ...
│ (Sequential)    │
└────────┬────────┘
         ↓
┌─────────────────┐
│ Interpolation   │  Fill 0.06s gaps
│ (Optional view) │  at stim artifacts
└────────┬────────┘
         ↓
  Display in GUI
```

## Annotation Format

```json
{
  "edf_file": "HUP213_LA1-LA2_LE3.edf",
  "timestamp": "2024-10-08T14:30:00",
  "n_channels": 10,
  "channel_names": [
    "LA1-LA2",
    "LA2-LA3",
    "LA3-LA4",
    ...
  ],
  "sampling_rate": 512.0,
  "spikes": [
    [2, 12.5],   ← Channel index 2, time 12.5 seconds
    [2, 15.3],
    [5, 18.7],
    ...
  ],
  "n_spikes": 42
}
```

## Quick Command Reference

```bash
# Setup (one-time)
pip install -r requirements.txt
python create_annotation_edfs.py --n_files 200

# Daily use
bash run_gui.sh

# Or manual
python spike_annotation_gui.py --edf_dir ../results/spike_annotation_edfs

# Validation
python validate_edfs.py --edf_dir ../results/spike_annotation_edfs

# Analysis
python analyze_annotations.py

# Complete workflow
bash workflow.sh
```

## Tips for Efficient Annotation

```
1. Start with gain adjustment
   ▼
   Find comfortable amplitude where spikes are visible

2. Scan the full recording first
   ▼
   Use scrollbar to get overview

3. Mark spikes systematically
   ▼
   Go channel by channel, left to right

4. Use interpolated view
   ▼
   See what the detector sees

5. Compare with raw view
   ▼
   Verify spikes are real, not artifacts

6. Take breaks
   ▼
   Eye strain is real with EEG review

7. Don't rush
   ▼
   Quality > Speed
```

## Legend

```
Symbol       Meaning
──────       ───────────────────────────────
   ▼         Annotated spike marker
   │         Stimulation time marker (red)
  ───        EEG trace
  ░░░        Scrollbar (inactive region)
  ▓▓▓        Scrollbar (active position)
   ↑↓        Gain adjustment arrows
   ←→        Scroll arrows
   ⏎         Enter key
   🖱         Mouse click
```

## Color Guide

```
Display Element       Color       Meaning
───────────────       ─────       ───────────────────
EEG traces           Black        Signal data
Stimulation lines    Red          Pulse timing
Spike markers        Blue         Annotated spikes
Grid lines           Gray         Time/amplitude ref
Background           White        Clean display
```

---

*This visual guide complements the other documentation files*

