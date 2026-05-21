# Seizure Annotation GUI — Standard Operating Procedure

## Overview

A PyQt5 desktop application for manual seizure annotation of intracranial iEEG (SEEG/ECoG) recordings. Clinicians can review ictal recordings, label global seizure patterns, and mark channel-level activity — all saved as structured JSON files.

---

## Installing the GUI

The GUI is a .dmg file, you can just open the .dmg, double-click the file, and it will load the GUI! Additionally, you can drag and drop the .dmg to the /Applications folder.

GUI: https://upenn.box.com/s/dggyu3i1ammwpup09t5gl17chpd4913t

Download .dmg file, open .dmg, drag and drop the sz-gui into the /Applications folder

---

## Launching the GUI

Click **"Open Data Folder…"** in the top toolbar and navigate to the data directory.

**The data folder we should be using is on the sauce server.**

The directory is "/sauce/DATA/Human_Data/CNT_iEEG_BIDS"

To get access to this directory as a CNT user you must first mount the sauce directory to you're macbook. 

1. Open Finder
2. Press CMD + K
3. connect to this server ID: smb://sauce.seas.upenn.edu/
4. You will need to put in you're username and password. This should be the same you use to access Pioneer or BOREL servers. 
5. Then you select the volume labeled "DATA"
6. You can then run sz_gui in the terminal and either put the path to the DATA folder after the --data_dir flag or you can navigate the gui to open your mounted volume.

---

## Expected Data Layout (BIDS-like)

```
data_dir/
  sub-RID####/
    ses-clinical01/
      *_scans.tsv              <- lists all EDF filenames for the session
      ieeg/
        *_ieeg.edf             <- raw iEEG recording (Volts)
        *_ieeg.json            <- sampling frequency, recording duration
        *_channels.tsv         <- channel names, types (SEEG/ECoG), status
        *_events.tsv           <- ictal onset time and duration
```

Only recordings whose filename contains `ictal` (but not `interictal`) are loaded into the Recording dropdown.

---

## Step-by-Step Workflow

1. **Select patient** from the Patient dropdown (top-left). The GUI auto-populates ictal recordings for that patient.
2. **Select recording** from the Recording dropdown. The EDF loads automatically (~5–15 s depending on file size). A status bar message confirms the number of channels, duration, and sampling frequency.
3. **Review the EEG.** On load, a solid red vertical line marks the onset time from the `_events.tsv` file.
4. **Choose a channel view** using the toggle buttons:
   - **ALL** - the last options lets you load in all channels. 
   - **Every-12th Channel** — displays every 12th SEEG/ECoG channel as a quick survey view. Additional channels can be added from the right-panel list.
   - **SOZ Electrodes** — displays only the seizure-onset zone electrodes listed in `seizure_gui/soz_electrodes.csv` for that patient. SOZ channels are highlighted in orange in both views. 
5. **Update the SOZ Electrode** by double click on the channel label on the left. Selected SOZ eletrode will be displayed in orange and has * symbol in front.
6. **Adjust gain** with the `↑` / `↓` arrow keys, or mouse scroll. Channel baselines remain fixed; only the trace amplitude grows or shrinks. The current gain multiplier is shown in the toolbar.
7. **Navigate** along the time axis with the horizontal scrollbar or `←` / `→` arrow keys.
8. **Global and Channel-Level Annotation Guidelines**

  When reviewing each recording segment, the annotator should first determine the global impression of the electrographic pattern:

   > **8a. Uniform Global Pattern**
   > If the recording demonstrates a uniform pattern throughout (e.g., entirely low-voltage fast activity [LVFA] or entirely rhythmic activity), the annotator should:
   >
   > Assign the appropriate global label (e.g., LVFA), and proceed to the next segment without additional channel-level annotation.

   > **8b. Mixed or Evolving Patterns**
   > If the recording contains multiple distinct electrographic patterns (e.g., burst discharges followed by LVFA), the annotator should:
   >
   > Assign the global label based on the predominant or overall impression of the segment, and use channel-level annotations to highlight specific regions exhibiting distinct or clinically relevant patterns (e.g., burst discharges preceding LVFA).

   > **8c. Channel-Level Overrides**
   > Channel-level annotations function as pattern-specific highlights within a recording. During model training and downstream analysis:
   >
   > Channel-level annotations will override the global label for the annotated portion of the recording. This ensures that focal or temporally restricted events (e.g., discharges occurring within predominantly LVFA activity) are accurately represented.
   >
   > To make channel-level annotations, single-click on a trace to add a start-time annotation. If you drag across a channel, you can select a time range and create both a start and stop annotation.

   > **8d. Majority Pattern with Focal Exceptions**
   > If the majority of channels demonstrate one pattern (e.g., LVFA), but a subset of channels show a different pattern (e.g., discharges), the annotator should:
   >
   > Assign the global label according to the majority pattern, and annotate the divergent channels accordingly to capture the heterogeneity of activity.


9. **Edit or delete a channel annotation** by double-clicking its entry in the Channel Annotations list (right panel), or by clicking the same trace location again to re-open the dialog.
10. **Save** with `S` or the green **Save Annotations** button. Annotations are written to `seizure_gui/annotations/{patient}_{recording}.json`. Saving is non-destructive — re-opening the same recording reloads prior annotations automatically.

---

## Annotation Output Format

```json
{
  "patient": "sub-RID0190",
  "recording": "sub-RID0190_ses-clinical01_task-ictal78680_run-02_ieeg",
  "global_types": ["Low-voltage fast activity (LVFA)"],
  "channel_annotations": [
    {
      "channel": "LAST03",
      "time_sec": 31.4,
      "types": ["Low-frequency repetitive spiking (LFRS)"]
    }
  ],
  "timestamp": "2026-02-22T10:00:00"
}
```

---

## Signal Processing Applied on Load

The raw EDF (in Volts) is converted to µV and the following filters are applied before display:

| Filter | Purpose |
|---|---|
| 0.5 Hz high-pass (4th-order Butterworth) | Removes DC offset and slow electrode drift |
| 60 Hz notch (IIR, Q=30) | Removes power-line interference |

No upper bandpass cutoff is applied, preserving high-frequency content relevant to LVFA (80–200 Hz).

---

## Keyboard Shortcuts

| Key | Action |
|---|---|
| `↑` | Increase gain (×1.5) |
| `↓` | Decrease gain (÷1.5) |
| `←` / `→` | Scroll time axis |
| `S` | Save annotations |
