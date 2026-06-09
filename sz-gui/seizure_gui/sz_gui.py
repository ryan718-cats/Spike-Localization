"""
Seizure Annotation GUI

A PyQt5 + pyqtgraph GUI for manual seizure annotation by clinicians.
Loads BIDS-format iEEG data (EDF + TSV/JSON metadata) and displays ictal
recordings for SEEG/ECOG channels. Supports three channel views (All Channels,
12 Channels with vertical scroll, SOZ Electrodes), gain-adjustable EEG traces
with fixed spacing, global seizure-type labeling, and channel-level click
annotations.

Data layout expected (BIDS-like):
    data_dir/
        sub-RID####/
            ses-*/
                {sub}_scans.tsv            # lists EDF filenames
                ieeg/
                    *_channels.tsv         # channel metadata (type, status)
                    *_events.tsv           # onset/duration/trial_type
                    *_ieeg.json            # SamplingFrequency, duration
                    *_ieeg.edf             # raw iEEG data (in Volts)

SOZ electrode list: seizure_gui/soz_electrodes.csv
    Columns: rid (e.g. "sub-RID0190"), soz_electrode (e.g. "LA01")

Usage:
    python seizure_annotation_gui.py --data_dir /path/to/data

Annotations are saved to seizure_gui/annotations/{rid}_{taskname}.json
"""

import sys
import os
import io
import re
import json
import csv
import argparse
from pathlib import Path
from datetime import datetime
from collections.abc import Iterable

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, iirnotch, sosfilt

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QComboBox, QCheckBox, QScrollBar, QGroupBox,
    QDialog, QDialogButtonBox, QListWidget, QListWidgetItem, QStatusBar,
    QShortcut, QSplitter, QLineEdit, QMessageBox, QSizePolicy, QAbstractItemView,
)
from PyQt5.QtCore import Qt, QTimer, QEvent, pyqtSignal, QRect, QRectF
from PyQt5.QtGui import (
    QKeySequence, QFont, QColor, QBrush, QMouseEvent,
    QPainter, QPen,
)

import pyqtgraph as pg
import mne

mne.set_log_level("WARNING")

# ── Constants ──────────────────────────────────────────────────────────────────

FREQUENCY_OPTIONS = ["(none)", "delta", "theta", "alpha", "beta", "gamma"]
PATTERN_OPTIONS = ["(none)", "rhythmic", "discharge"]
MODIFIER_OPTIONS = ["None", "burst", "repetitive burst"]

CHANNEL_SPACING_UV = 300.0   # fixed baseline spacing between channels in µV
VISIBLE_SECONDS = 3.0        # visible EEG window (seconds)
TIME_SCROLL_STEP_SEC = 0.25  # horizontal scroll step (seconds); ←→ and time scrollbar arrows
SCROLL_TICKS_PER_SEC = 4     # scrollbar resolution (1 tick = TIME_SCROLL_STEP_SEC)
IEEG_SPIKE_HALF_WINDOW_SEC = 7.0   # ±7 s around spike center (14 s total)
# CSV timestamp_sec is the SN2 trigger (0.46 crossing); spike center is +0.5 s.
SPIKENET_TRIGGER_TO_CENTER_SEC = 0.5
CLIP_SPIKE_CENTER_REL_SEC = (
    IEEG_SPIKE_HALF_WINDOW_SEC + SPIKENET_TRIGGER_TO_CENTER_SEC
)  # 7.5 s in preprocessed clips (7.0 s before +0.5 s export shift)
CLIP_SPIKE_REGION_START_SEC = 7.0  # spike review band in preprocessed clips
CLIP_SPIKE_REGION_END_SEC = 8.0
SPIKE_MARK_HALF_WIDTH_SEC = 0.5    # ±0.5 s band when streaming (non-clip) iEEG
SPIKE_MARK_LINE_COLOR = (220, 40, 40)
SPIKE_MARK_FILL_COLOR = (255, 180, 180, 70)
GAIN_STEP = 1.5              # multiply/divide per ↑↓ key press
GAIN_MIN = 1e-9              # minimum gain (wide ↑↓ range; step unchanged)
GAIN_MAX = 1e9               # maximum gain (wide ↑↓ range; step unchanged)
IEEG_SCALE = 1e6             # convert Volts → µV

# Colour used for the ictal background shading
ICTAL_SHADE_COLOR = (255, 200, 200, 80)   # light red, semi-transparent

SOZ_ELECTRODE_COLOR = (255, 165, 0)       # orange traces for SOZ channels

# Core EEG channel order from EDF handler workflow.
CORE_EEG_CHANNELS = [
    "C3", "C4", "Cz", "F3", "F4", "F7", "F8", "Fp1", "Fp2",
    "Fz", "O1", "O2", "P3", "P4", "Pz", "T3", "T4", "T5", "T6",
]

# 10–20 schematic: normalized (0–1) positions inside the head ellipse. No bitmap —
# drawing + hit tests scale with widget size.
TEN_TWENTY_SHAPE_NORM: dict[str, tuple[float, float]] = {
    "Fp1": (0.36, 0.07),
    "Fp2": (0.64, 0.07),
    "F7": (0.13, 0.25),
    "F3": (0.33, 0.29),
    "Fz": (0.50, 0.29),
    "F4": (0.67, 0.29),
    "F8": (0.87, 0.25),
    "T3": (0.06, 0.45),
    "C3": (0.29, 0.45),
    "Cz": (0.50, 0.45),
    "C4": (0.71, 0.45),
    "T4": (0.94, 0.45),
    "T5": (0.18, 0.63),
    "P3": (0.34, 0.61),
    "Pz": (0.50, 0.61),
    "P4": (0.66, 0.61),
    "T6": (0.82, 0.63),
    "O1": (0.39, 0.85),
    "O2": (0.61, 0.85),
}

# Map legacy 10–20 names to common alternate labels in iEEG channel names.
_TEN_TWENTY_CONTACT_ALIASES: dict[str, set[str]] = {
    "T3": {"T7"},
    "T7": {"T3"},
    "T4": {"T8"},
    "T8": {"T4"},
    "T5": {"P7"},
    "P7": {"T5"},
    "T6": {"P8"},
    "P8": {"T6"},
}


class TenTwentyShapeMontageWidget(QWidget):
    """Vector 10–20 diagram: head outline + circular electrodes; clicks always align with drawn shapes."""

    clicked_contact = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._positions = TEN_TWENTY_SHAPE_NORM
        self._highlight_names: set[str] = set()
        self.setMinimumSize(320, 380)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.MinimumExpanding)
        self.setToolTip(
            "Click an electrode to add or remove that contact only. "
            "T3/T7 etc. share one disk."
        )
        self.setStyleSheet("background: #f7f7f7; border: 1px solid #c0c0c0;")

    def set_highlight_contacts(self, schematic_keys: set[str]) -> None:
        """Highlight electrode disks (keys in TEN_TWENTY_SHAPE_NORM) for annotated contacts."""
        self._highlight_names = set(schematic_keys)
        self.update()

    def _margin_px(self) -> int:
        # Tight margin so the head oval uses most of the widget.
        return max(3, int(min(self.width(), self.height()) * 0.005))

    def _head_rect(self):
        m = self._margin_px()
        return QRect(m, m, max(1, self.width() - 2 * m), max(1, self.height() - 2 * m))

    def _electrode_radius_px(self) -> float:
        hr = self._head_rect()
        return max(14.0, float(min(hr.width(), hr.height())) * 0.060)

    def _center_for(self, name: str) -> tuple[float, float] | None:
        if name not in self._positions:
            return None
        nx, ny = self._positions[name]
        hr = self._head_rect()
        cx = float(hr.left()) + nx * float(hr.width())
        cy = float(hr.top()) + ny * float(hr.height())
        return cx, cy

    def paintEvent(self, event):
        del event
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        hr = self._head_rect()
        p.setPen(QPen(QColor(100, 100, 100), 1.5))
        p.setBrush(QBrush(QColor(245, 232, 210)))
        p.drawEllipse(hr)

        # Left/right and anterior/posterior quadrant guides (dotted cross at ellipse center).
        cx = float(hr.center().x())
        cy = float(hr.center().y())
        quad_pen = QPen(QColor(90, 90, 90))
        quad_pen.setStyle(Qt.DotLine)
        quad_pen.setWidthF(1.2)
        p.setPen(quad_pen)
        p.drawLine(int(cx), int(hr.top()), int(cx), int(hr.bottom()))
        p.drawLine(int(hr.left()), int(cy), int(hr.right()), int(cy))

        r_el = self._electrode_radius_px()
        font = QFont("Arial", max(6, int(r_el * 0.34)))
        p.setFont(font)

        for name, (nx, ny) in self._positions.items():
            cx = float(hr.left()) + nx * float(hr.width())
            cy = float(hr.top()) + ny * float(hr.height())
            is_hi = name in self._highlight_names
            p.setPen(QPen(QColor(0, 110, 40) if is_hi else QColor(40, 80, 160), 2.0 if is_hi else 1.2))
            p.setBrush(
                QBrush(QColor(190, 255, 190) if is_hi else QColor(255, 255, 255))
            )
            p.drawEllipse(QRectF(cx - r_el, cy - r_el, 2 * r_el, 2 * r_el))
            p.setPen(QPen(QColor(20, 20, 20)))
            p.drawText(
                QRectF(cx - r_el, cy - r_el, 2 * r_el, 2 * r_el),
                Qt.AlignCenter,
                name,
            )

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() != Qt.LeftButton:
            return super().mousePressEvent(event)
        px, py = float(event.x()), float(event.y())
        hit_r = self._electrode_radius_px() * 1.05
        r2 = hit_r * hit_r
        best: tuple[float, str] | None = None
        for name in self._positions:
            c = self._center_for(name)
            if c is None:
                continue
            cx, cy = c
            d2 = (px - cx) ** 2 + (py - cy) ** 2
            if d2 <= r2 and (best is None or d2 < best[0]):
                best = (d2, name)
        if best is not None:
            self.clicked_contact.emit(best[1])
            return
        super().mousePressEvent(event)

# When running as a PyInstaller .app bundle the package directory is read-only;
# save annotations to ~/Documents/SeizureAnnotations instead.
if getattr(sys, "frozen", False):
    ANNOTATIONS_DIR = Path.home() / "Documents" / "SeizureAnnotations"
else:
    ANNOTATIONS_DIR = Path(__file__).parent / "annotations"

# SOZ electrode data embedded directly — no external CSV file required.
_SOZ_CSV_DATA = """\
rid,soz_electrode
sub-RID0013,MST01
sub-RID0013,MST02
sub-RID0013,MST03
sub-RID0013,RG50
sub-RID0013,RG51
sub-RID0013,RG62
sub-RID0013,RG63
sub-RID0013,RH01
sub-RID0013,RH02
sub-RID0013,RH03
sub-RID0013,RH04
sub-RID0013,RH05
sub-RID0013,RH06
sub-RID0014,LG05
sub-RID0014,LG06
sub-RID0014,LG07
sub-RID0014,LG13
sub-RID0014,LG14
sub-RID0014,LG15
sub-RID0014,LG22
sub-RID0014,LG23
sub-RID0014,LG24
sub-RID0015,LA01
sub-RID0015,LA02
sub-RID0015,LA03
sub-RID0015,LH01
sub-RID0015,LH02
sub-RID0015,LH03
sub-RID0015,LH04
sub-RID0015,LMST01
sub-RID0015,LMST02
sub-RID0015,LMST03
sub-RID0018,LA01
sub-RID0018,LA02
sub-RID0018,LA03
sub-RID0018,LA04
sub-RID0018,LAST01
sub-RID0018,LAST02
sub-RID0018,LAST03
sub-RID0018,LAST04
sub-RID0018,LG17
sub-RID0018,LG18
sub-RID0018,LG19
sub-RID0018,LG20
sub-RID0018,LG25
sub-RID0018,LG26
sub-RID0018,LG27
sub-RID0018,LG28
sub-RID0018,LG33
sub-RID0018,LG34
sub-RID0018,LG35
sub-RID0018,LG49
sub-RID0018,LG50
sub-RID0018,LG51
sub-RID0018,LG52
sub-RID0018,LG53
sub-RID0018,LG54
sub-RID0018,LG55
sub-RID0018,LG58
sub-RID0018,LG59
sub-RID0018,LG60
sub-RID0018,LG61
sub-RID0018,LG62
sub-RID0018,LG63
sub-RID0018,LG64
sub-RID0018,LMST01
sub-RID0018,LMST02
sub-RID0018,LMST03
sub-RID0018,LMST04
sub-RID0018,LPF03
sub-RID0018,LPF04
sub-RID0020,LG29
sub-RID0020,LG36
sub-RID0020,LG38
sub-RID0021,LMST01
sub-RID0021,LMST02
sub-RID0024,AD01
sub-RID0024,AD02
sub-RID0024,AD03
sub-RID0030,LAT02
sub-RID0030,LAT03
sub-RID0030,LAT04
sub-RID0030,LPT02
sub-RID0030,LPT03
sub-RID0030,RAT01
sub-RID0030,RAT02
sub-RID0030,RAT03
sub-RID0030,RMT01
sub-RID0030,RMT02
sub-RID0030,RMT03
sub-RID0030,RMT04
sub-RID0030,RPT01
sub-RID0030,RPT02
sub-RID0030,RPT03
sub-RID0030,RPT04
sub-RID0031,RA01
sub-RID0031,RA02
sub-RID0031,RA03
sub-RID0031,RA04
sub-RID0031,RB01
sub-RID0031,RB02
sub-RID0031,RB03
sub-RID0031,RB04
sub-RID0032,DH01
sub-RID0032,DH02
sub-RID0032,DH03
sub-RID0032,DH04
sub-RID0032,DHA01
sub-RID0032,DHA02
sub-RID0032,DHA03
sub-RID0032,DHA04
sub-RID0033,AT02
sub-RID0033,AT03
sub-RID0033,DH01
sub-RID0033,DH02
sub-RID0033,LG05
sub-RID0033,LG06
sub-RID0033,LG46
sub-RID0033,LG48
sub-RID0033,LG62
sub-RID0033,LG63
sub-RID0033,PT02
sub-RID0033,PT03
sub-RID0037,LAT01
sub-RID0037,LAT02
sub-RID0037,LAT03
sub-RID0037,LMT01
sub-RID0037,LMT02
sub-RID0037,RHI01
sub-RID0037,RHI02
sub-RID0037,RHI03
sub-RID0037,RHI04
sub-RID0037,RPH01
sub-RID0037,RPH02
sub-RID0037,RPH03
sub-RID0042,RAF02
sub-RID0042,RAF04
sub-RID0042,RAF05
sub-RID0042,RPF07
sub-RID0042,RPF09
sub-RID0050,DA01
sub-RID0050,DA02
sub-RID0050,DA03
sub-RID0050,DA04
sub-RID0050,DH01
sub-RID0050,DH02
sub-RID0050,DH03
sub-RID0050,DH04
sub-RID0051,RDA01
sub-RID0051,RDA02
sub-RID0051,RDA03
sub-RID0051,RDA04
sub-RID0054,LG23
sub-RID0054,LG31
sub-RID0055,RAT01
sub-RID0055,RAT02
sub-RID0055,RAT03
sub-RID0055,RAT04
sub-RID0055,RG03
sub-RID0055,RG04
sub-RID0055,RG05
sub-RID0055,RG06
sub-RID0055,RG09
sub-RID0055,RG10
sub-RID0055,RG11
sub-RID0055,RG12
sub-RID0055,RG17
sub-RID0055,RG18
sub-RID0055,RG19
sub-RID0055,RG20
sub-RID0055,RG25
sub-RID0055,RG26
sub-RID0055,RG27
sub-RID0055,RST01
sub-RID0055,RST02
sub-RID0055,RST03
sub-RID0058,RG04
sub-RID0058,RG11
sub-RID0058,RG12
sub-RID0058,RG13
sub-RID0058,RG14
sub-RID0058,RG20
sub-RID0058,RG21
sub-RID0058,RG22
sub-RID0058,RG23
sub-RID0058,RG24
sub-RID0058,RG25
sub-RID0058,RG26
sub-RID0058,RG27
sub-RID0058,RG28
sub-RID0058,RG29
sub-RID0058,RG30
sub-RID0058,RST02
sub-RID0058,RST03
sub-RID0058,RST04
sub-RID0060,LG20
sub-RID0060,LG21
sub-RID0060,LG22
sub-RID0060,LG23
sub-RID0063,ROF01
sub-RID0063,ROF02
sub-RID0063,ROF03
sub-RID0064,G38
sub-RID0064,G39
sub-RID0064,G40
sub-RID0064,TLD03
sub-RID0064,TLD04
sub-RID0064,TLD05
sub-RID0064,TLD06
sub-RID0065,AMY01
sub-RID0065,AMY02
sub-RID0065,AMY03
sub-RID0065,AMY04
sub-RID0065,AST01
sub-RID0065,AST02
sub-RID0065,AST03
sub-RID0065,AST04
sub-RID0065,HIP01
sub-RID0065,HIP02
sub-RID0065,HIP03
sub-RID0065,HIP04
sub-RID0068,AST01
sub-RID0068,AST02
sub-RID0069,LG01
sub-RID0069,LG02
sub-RID0069,LG03
sub-RID0069,LG04
sub-RID0069,LG05
sub-RID0069,LG09
sub-RID0069,LG10
sub-RID0069,LG11
sub-RID0069,LG12
sub-RID0069,LG13
sub-RID0069,LG14
sub-RID0069,LG15
sub-RID0069,LG18
sub-RID0069,LG19
sub-RID0069,LG20
sub-RID0069,LG21
sub-RID0069,LG22
sub-RID0069,LG23
sub-RID0069,LG24
sub-RID0069,LG25
sub-RID0069,LG26
sub-RID0069,LG27
sub-RID0069,LG28
sub-RID0069,LG35
sub-RID0069,LG42
sub-RID0069,LT03
sub-RID0069,LT04
sub-RID0069,LT05
sub-RID0069,LT06
sub-RID0069,OPI01
sub-RID0069,OPI02
sub-RID0069,OPI03
sub-RID0069,OPI04
sub-RID0069,OPI05
sub-RID0069,PST05
sub-RID0069,PST06
sub-RID0070,AST01
sub-RID0070,AST02
sub-RID0070,LA02
sub-RID0070,LA03
sub-RID0070,LA04
sub-RID0101,RDNET01
sub-RID0101,RDNET02
sub-RID0101,RDNET03
sub-RID0101,RDNET04
sub-RID0102,APD01
sub-RID0102,APD02
sub-RID0102,APD03
sub-RID0102,APD04
sub-RID0102,LPF02
sub-RID0102,LPF03
sub-RID0102,PPD01
sub-RID0102,PPD02
sub-RID0102,PPD03
sub-RID0102,PPD04
sub-RID0102,ROF07
sub-RID0102,ROF08
sub-RID0112,RA01
sub-RID0112,RA02
sub-RID0112,RA08
sub-RID0112,RA09
sub-RID0117,RDH01
sub-RID0117,RDH02
sub-RID0117,RDH03
sub-RID0117,RDH04
sub-RID0117,RTO04
sub-RID0117,RTO05
sub-RID0131,LA01
sub-RID0131,LA02
sub-RID0131,LA03
sub-RID0131,LA04
sub-RID0131,LB01
sub-RID0131,LB02
sub-RID0131,LB03
sub-RID0131,LC05
sub-RID0131,LC06
sub-RID0131,LD10
sub-RID0131,LD11
sub-RID0131,LD12
sub-RID0131,LE04
sub-RID0131,LE05
sub-RID0131,LH08
sub-RID0131,LH09
sub-RID0131,LH10
sub-RID0131,LK03
sub-RID0131,LK04
sub-RID0131,LK05
sub-RID0139,LC01
sub-RID0139,LC02
sub-RID0139,LC03
sub-RID0139,LC04
sub-RID0139,LC05
sub-RID0139,LC06
sub-RID0139,LC07
sub-RID0139,LD01
sub-RID0139,LD02
sub-RID0139,LD03
sub-RID0139,LD04
sub-RID0139,LD05
sub-RID0139,LE01
sub-RID0139,LE02
sub-RID0139,LE03
sub-RID0139,LE04
sub-RID0139,LE05
sub-RID0139,LE06
sub-RID0139,LE07
sub-RID0139,LG01
sub-RID0139,LG02
sub-RID0139,LG03
sub-RID0139,LG04
sub-RID0139,LG05
sub-RID0139,LH05
sub-RID0139,LH06
sub-RID0139,LH07
sub-RID0139,LH08
sub-RID0139,LH09
sub-RID0139,LH10
sub-RID0139,LH11
sub-RID0139,LJ05
sub-RID0139,LJ06
sub-RID0139,LJ07
sub-RID0139,LJ08
sub-RID0139,LJ09
sub-RID0139,LJ10
sub-RID0139,LJ11
sub-RID0139,LL03
sub-RID0139,LL04
sub-RID0139,LL05
sub-RID0139,LL06
sub-RID0139,LL07
sub-RID0143,LAT02
sub-RID0143,LDH02
sub-RID0146,LE07
sub-RID0146,LE08
sub-RID0146,LE09
sub-RID0146,LM01
sub-RID0146,LM02
sub-RID0146,LM03
sub-RID0146,LN01
sub-RID0146,RK01
sub-RID0146,RM01
sub-RID0146,RM02
sub-RID0146,RM03
sub-RID0157,RDA01
sub-RID0157,RDA02
sub-RID0157,RDA03
sub-RID0157,RDA04
sub-RID0160,LAT02
sub-RID0160,LAT03
sub-RID0160,LAT04
sub-RID0160,LG03
sub-RID0160,LG04
sub-RID0160,LG05
sub-RID0160,LG11
sub-RID0160,LG12
sub-RID0160,LG13
sub-RID0160,LG20
sub-RID0160,LG26
sub-RID0160,LG27
sub-RID0160,LG28
sub-RID0160,LG29
sub-RID0160,LG35
sub-RID0160,LG36
sub-RID0160,LG37
sub-RID0160,LG38
sub-RID0160,LG43
sub-RID0160,LG44
sub-RID0160,LG45
sub-RID0160,LMT01
sub-RID0160,LMT02
sub-RID0160,LMT03
sub-RID0160,LMT04
sub-RID0165,LDH02
sub-RID0165,RDH02
sub-RID0167,RC05
sub-RID0167,RC06
sub-RID0167,RC07
sub-RID0167,RC08
sub-RID0167,RL04
sub-RID0167,RL05
sub-RID0167,RL06
sub-RID0171,DP02
sub-RID0171,LG23
sub-RID0171,LG34
sub-RID0171,LG42
sub-RID0175,RA01
sub-RID0175,RA02
sub-RID0175,RA03
sub-RID0175,RA04
sub-RID0179,RAST01
sub-RID0179,RAST02
sub-RID0179,RAST03
sub-RID0179,RAST04
sub-RID0179,RDH01
sub-RID0179,RDH02
sub-RID0179,RDH03
sub-RID0179,RDH04
sub-RID0179,RG20
sub-RID0179,RG21
sub-RID0179,RG22
sub-RID0179,RG28
sub-RID0179,RG29
sub-RID0179,RG30
sub-RID0179,RG34
sub-RID0179,RG35
sub-RID0179,RG36
sub-RID0179,RG37
sub-RID0179,RG38
sub-RID0179,RG40
sub-RID0179,RG41
sub-RID0179,RG42
sub-RID0179,RG43
sub-RID0179,RG44
sub-RID0179,RMST01
sub-RID0179,RMST02
sub-RID0179,RO02
sub-RID0179,RO03
sub-RID0179,RO04
sub-RID0179,RO05
sub-RID0179,RO06
sub-RID0179,RPST01
sub-RID0179,RPST02
sub-RID0179,RPST03
sub-RID0179,RPST04
sub-RID0179,RSP04
sub-RID0179,RSP05
sub-RID0179,RSP06
sub-RID0186,LB01
sub-RID0186,LB02
sub-RID0186,RB01
sub-RID0190,LDA02
sub-RID0190,LDH02
sub-RID0190,LMST02
sub-RID0192,INSP01
sub-RID0192,INSP02
sub-RID0192,INSP03
sub-RID0192,MTG01
sub-RID0192,OFAL01
sub-RID0192,OFAL02
sub-RID0192,OFAL03
sub-RID0192,PIFG01
sub-RID0192,PIFG02
sub-RID0192,PIFG03
sub-RID0193,DNET01
sub-RID0193,RDA01
sub-RID0193,RDA02
sub-RID0193,RDA03
sub-RID0193,RDA04
sub-RID0206,LA01
sub-RID0206,LA02
sub-RID0206,LA03
sub-RID0206,LA04
sub-RID0206,LA05
sub-RID0206,LB01
sub-RID0206,LB02
sub-RID0206,LB03
sub-RID0206,LB04
sub-RID0206,LB05
sub-RID0206,LB06
sub-RID0206,LB07
sub-RID0206,RA01
sub-RID0206,RA02
sub-RID0206,RD03
sub-RID0213,LAT01
sub-RID0213,LAT02
sub-RID0213,LDH01
sub-RID0213,LDH02
sub-RID0213,LDH03
sub-RID0213,LDH04
sub-RID0213,LMT01
sub-RID0213,LMT02
sub-RID0213,RDA01
sub-RID0213,RDA02
sub-RID0213,RDA03
sub-RID0213,RDA04
sub-RID0213,RDH01
sub-RID0213,RDH02
sub-RID0213,RDH03
sub-RID0213,RDH04
sub-RID0227,RIP01
sub-RID0227,RIP02
sub-RID0227,RIP03
sub-RID0227,RIP04
sub-RID0227,RSP01
sub-RID0227,RSP02
sub-RID0227,RSP03
sub-RID0227,RSP04
sub-RID0230,LA08
sub-RID0230,LA09
sub-RID0230,LA10
sub-RID0230,LA11
sub-RID0230,LJ01
sub-RID0230,LJ02
sub-RID0230,LJ03
sub-RID0230,LJ04
sub-RID0230,LJ05
sub-RID0238,LG01
sub-RID0240,R02
sub-RID0241,RE01
sub-RID0250,LA01
sub-RID0250,LA02
sub-RID0250,LO01
sub-RID0250,LO02
sub-RID0252,LDA04
sub-RID0252,LDA05
sub-RID0252,LSZ02
sub-RID0252,LSZ03
sub-RID0259,LA01
sub-RID0259,LA02
sub-RID0259,LA03
sub-RID0259,LB01
sub-RID0259,LB02
sub-RID0259,LB03
sub-RID0267,LB01
sub-RID0267,LB02
sub-RID0267,LB03
sub-RID0267,LC01
sub-RID0267,LC02
sub-RID0267,LD01
sub-RID0267,LD02
sub-RID0267,LD03
sub-RID0267,LD04
sub-RID0267,LD05
sub-RID0267,LD06
sub-RID0267,RB01
sub-RID0267,RB02
sub-RID0267,RB03
sub-RID0267,RB04
sub-RID0272,LA08
sub-RID0272,LB06
sub-RID0272,LB07
sub-RID0272,LC06
sub-RID0272,LC08
sub-RID0272,LY05
sub-RID0272,LY06
sub-RID0272,LY07
sub-RID0274,RB01
sub-RID0274,RB02
sub-RID0278,LA02
sub-RID0278,LA03
sub-RID0278,LA04
sub-RID0279,LA01
sub-RID0279,LA02
sub-RID0279,LA03
sub-RID0279,LA04
sub-RID0279,LA05
sub-RID0279,LB01
sub-RID0279,LB02
sub-RID0279,LB03
sub-RID0280,LA06
sub-RID0280,LC01
sub-RID0320,LC01
sub-RID0320,LC02
sub-RID0322,RG02
sub-RID0322,RG03
sub-RID0322,RG04
sub-RID0322,RG05
sub-RID0325,LA03
sub-RID0325,LA04
sub-RID0325,RA03
sub-RID0325,RA04
sub-RID0325,RB03
sub-RID0325,RB04
sub-RID0328,LA01
sub-RID0328,LA02
sub-RID0328,LA03
sub-RID0328,LC01
sub-RID0328,LC02
sub-RID0328,LC03
sub-RID0328,LJ01
sub-RID0328,LJ02
sub-RID0328,LJ03
sub-RID0328,LJ04
sub-RID0328,LJ05
sub-RID0328,RA03
sub-RID0328,RA05
sub-RID0328,RA07
sub-RID0328,RD01
sub-RID0328,RD02
sub-RID0328,RD03
sub-RID0328,RD04
sub-RID0328,RD05
sub-RID0329,LB03
sub-RID0329,LB04
sub-RID0329,LC02
sub-RID0329,LC03
sub-RID0329,LC04
sub-RID0329,LD02
sub-RID0329,LD03
sub-RID0329,LE02
sub-RID0329,LE03
sub-RID0329,LE04
sub-RID0330,LG05
sub-RID0330,LG06
sub-RID0330,LG07
sub-RID0330,LG08
sub-RID0330,LG09
sub-RID0330,LI05
sub-RID0330,LI06
sub-RID0330,LI07
sub-RID0330,LI08
sub-RID0330,RF03
sub-RID0330,RF04
sub-RID0330,RF05
sub-RID0330,RF06
sub-RID0330,RF07
sub-RID0330,RF08
sub-RID0330,RF09
sub-RID0332,LA01
sub-RID0332,LA02
sub-RID0332,LA03
sub-RID0332,LA04
sub-RID0332,LB01
sub-RID0332,LB02
sub-RID0332,LB03
sub-RID0332,LB04
sub-RID0334,LA01
sub-RID0334,LA02
sub-RID0334,LA03
sub-RID0334,LA04
sub-RID0334,LB01
sub-RID0334,LB02
sub-RID0334,LB03
sub-RID0334,LB04
sub-RID0334,LD01
sub-RID0334,LD02
sub-RID0334,LK01
sub-RID0334,LK02
sub-RID0334,LK03
sub-RID0334,LK04
sub-RID0334,LM01
sub-RID0334,LM02
sub-RID0334,LM03
sub-RID0334,LM04
sub-RID0337,LB05
sub-RID0337,LB06
sub-RID0337,LB07
sub-RID0337,RC06
sub-RID0338,RC01
sub-RID0338,RC02
sub-RID0338,RC03
sub-RID0338,RC04
sub-RID0356,RB03
sub-RID0356,RJ01
sub-RID0356,RJ02
sub-RID0356,RJ03
sub-RID0356,RJ04
sub-RID0356,RJ05
sub-RID0357,LXB03
sub-RID0357,LXB04
sub-RID0357,LXB05
sub-RID0357,LXB06
sub-RID0357,RH01
sub-RID0357,RH02
sub-RID0357,RH03
sub-RID0365,LA01
sub-RID0365,LA02
sub-RID0365,LB01
sub-RID0365,LB02
sub-RID0365,LB07
sub-RID0365,LB08
sub-RID0365,LB09
sub-RID0365,LC01
sub-RID0365,LC02
sub-RID0365,LC07
sub-RID0365,LC08
sub-RID0365,LC09
sub-RID0365,LJ01
sub-RID0365,RJ01
sub-RID0371,LD09
sub-RID0371,LF05
sub-RID0371,LG02
sub-RID0371,LG03
sub-RID0371,LG04
sub-RID0371,LG05
sub-RID0371,LG06
sub-RID0371,LG07
sub-RID0371,LG08
sub-RID0371,LG09
sub-RID0371,LH03
sub-RID0371,LH05
sub-RID0371,RD09
sub-RID0371,RD10
sub-RID0371,RD11
sub-RID0371,RI03
sub-RID0371,RI05
sub-RID0381,RA04
sub-RID0381,RA05
sub-RID0381,RA06
sub-RID0381,RB01
sub-RID0381,RB02
sub-RID0381,RB03
sub-RID0381,RB04
sub-RID0381,RC01
sub-RID0381,RC02
sub-RID0381,RF06
sub-RID0381,RF07
sub-RID0381,RF08
sub-RID0381,RG01
sub-RID0381,RG02
sub-RID0381,RG03
sub-RID0381,RI01
sub-RID0381,RI02
sub-RID0381,RI03
sub-RID0381,RJ02
sub-RID0381,RL08
sub-RID0381,RL09
sub-RID0381,RL10
sub-RID0382,RL01
sub-RID0382,RL02
sub-RID0382,RL03
sub-RID0382,RL04
sub-RID0385,RB02
sub-RID0385,RB03
sub-RID0385,RB04
sub-RID0385,RB05
sub-RID0385,RB09
sub-RID0385,RB10
sub-RID0385,RC04
sub-RID0385,RC05
sub-RID0385,RC06
sub-RID0385,RC07
sub-RID0386,LA01
sub-RID0386,LA02
sub-RID0386,LA03
sub-RID0386,LB01
sub-RID0386,LB02
sub-RID0386,LB03
sub-RID0386,LC01
sub-RID0386,LC02
sub-RID0386,LC03
sub-RID0386,LC04
sub-RID0392,LB07
sub-RID0405,LA01
sub-RID0405,LA02
sub-RID0405,LA03
sub-RID0405,LB01
sub-RID0405,LB02
sub-RID0405,LB03
sub-RID0405,LB04
sub-RID0405,LB05
sub-RID0405,LC01
sub-RID0405,LC02
sub-RID0405,LC03
sub-RID0405,LC04
sub-RID0412,LA01
sub-RID0412,LA02
sub-RID0412,LA03
sub-RID0412,LA04
sub-RID0412,LB01
sub-RID0412,LB02
sub-RID0412,LB03
sub-RID0412,LB04
sub-RID0420,LA01
sub-RID0420,LA02
sub-RID0420,LB02
sub-RID0420,LC01
sub-RID0420,LC02
sub-RID0420,LD01
sub-RID0420,LD02
sub-RID0420,LD03
sub-RID0420,LE01
sub-RID0420,LE02
sub-RID0420,LF01
sub-RID0420,LF02
sub-RID0420,LF03
sub-RID0424,RB01
sub-RID0424,RB02
sub-RID0424,RB03
sub-RID0424,RC01
sub-RID0424,RC02
sub-RID0424,RC03
sub-RID0424,RH01
sub-RID0424,RH02
sub-RID0424,RH03
sub-RID0424,RH04
sub-RID0424,RH05
sub-RID0424,RH06
sub-RID0424,RH07
sub-RID0440,LE10
sub-RID0440,LE11
sub-RID0440,LH01
sub-RID0440,LH02
sub-RID0440,LH03
sub-RID0440,LH04
sub-RID0440,LH05
sub-RID0440,LH06
sub-RID0440,LH07
sub-RID0440,LH08
sub-RID0442,RD03
sub-RID0442,RD04
sub-RID0442,RD05
sub-RID0442,RD06
sub-RID0442,RD07
sub-RID0452,LD02
sub-RID0452,LD03
sub-RID0452,LE07
sub-RID0452,LF03
sub-RID0452,LF04
sub-RID0452,LH06
sub-RID0452,LH07
sub-RID0452,LI03
sub-RID0452,LI04
sub-RID0452,RD02
sub-RID0452,RD03
sub-RID0452,RG03
sub-RID0452,RG04
sub-RID0452,RH05
sub-RID0452,RH06
sub-RID0452,RK04
sub-RID0452,RK05
sub-RID0454,LB01
sub-RID0454,RA01
sub-RID0454,RA02
sub-RID0454,RA03
sub-RID0454,RA04
sub-RID0454,RA05
sub-RID0454,RB01
sub-RID0454,RB02
sub-RID0454,RB03
sub-RID0454,RB04
sub-RID0454,RB05
sub-RID0454,RC01
sub-RID0454,RC02
sub-RID0454,RC03
sub-RID0454,RC04
sub-RID0459,LA01
sub-RID0459,LA02
sub-RID0459,LA09
sub-RID0459,LA10
sub-RID0459,LB01
sub-RID0459,LB02
sub-RID0459,LB03
sub-RID0459,LB04
sub-RID0459,LB09
sub-RID0459,LB10
sub-RID0459,LB11
sub-RID0459,LD01
sub-RID0459,LD02
sub-RID0459,LD03
sub-RID0459,LD04
sub-RID0459,LE01
sub-RID0459,LE02
sub-RID0459,LE10
sub-RID0459,LE11
sub-RID0459,LE12
sub-RID0459,LF01
sub-RID0459,LF02
sub-RID0459,LG01
sub-RID0459,LG02
sub-RID0459,LG03
sub-RID0459,LK01
sub-RID0459,LK02
sub-RID0459,LK03
sub-RID0459,LK04
sub-RID0459,LK05
sub-RID0459,LK06
sub-RID0459,LK07
sub-RID0459,LK08
sub-RID0459,LK09
sub-RID0459,LM07
sub-RID0459,LM08
sub-RID0459,LM09
sub-RID0459,LN01
sub-RID0459,LN02
sub-RID0459,LN03
sub-RID0459,LN04
sub-RID0459,LN05
sub-RID0459,LN06
sub-RID0459,LN07
sub-RID0459,LN08
sub-RID0459,LN09
sub-RID0459,LN10
sub-RID0459,LN11
sub-RID0459,LN12
sub-RID0472,LD07
sub-RID0472,LF05
sub-RID0472,LG09
sub-RID0472,LH07
sub-RID0472,LI05
sub-RID0472,LK05
sub-RID0472,LL05
sub-RID0490,LA01
sub-RID0490,LA02
sub-RID0490,LA03
sub-RID0502,LB09
sub-RID0502,LB10
sub-RID0502,LB11
sub-RID0502,RB01
sub-RID0502,RB02
sub-RID0502,RB03
sub-RID0502,RB04
sub-RID0502,RB05
sub-RID0508,LA01
sub-RID0508,LA02
sub-RID0508,LA03
sub-RID0508,LA04
sub-RID0508,LA11
sub-RID0508,LA12
sub-RID0508,LB01
sub-RID0508,LB02
sub-RID0508,LB03
sub-RID0508,LB04
sub-RID0520,LA06
sub-RID0520,LA07
sub-RID0520,LA08
sub-RID0520,LA09
sub-RID0520,LA10
sub-RID0520,LA11
sub-RID0520,LA12
sub-RID0520,LB10
sub-RID0520,LB11
sub-RID0520,LB12
sub-RID0520,LC08
sub-RID0520,LC09
sub-RID0520,LC10
sub-RID0520,LC11
sub-RID0520,LC12
sub-RID0520,LGr38
sub-RID0520,LGr39
sub-RID0520,LGr40
sub-RID0520,LGr46
sub-RID0520,LGr47
sub-RID0520,LGr48
sub-RID0520,LGr55
sub-RID0520,LGr56
sub-RID0522,RA01
sub-RID0522,RA02
sub-RID0522,RA03
sub-RID0522,RA04
sub-RID0522,RB01
sub-RID0522,RB02
sub-RID0522,RB03
sub-RID0522,RB04
sub-RID0522,RB05
sub-RID0529,RA04
sub-RID0529,RA05
sub-RID0529,RF03
sub-RID0529,RF04
sub-RID0530,LA01
sub-RID0530,LA02
sub-RID0530,LA03
sub-RID0530,LA04
sub-RID0530,LA05
sub-RID0530,LB01
sub-RID0530,LB02
sub-RID0530,LB03
sub-RID0530,LB04
sub-RID0530,LB05
sub-RID0530,RA01
sub-RID0530,RA02
sub-RID0530,RA03
sub-RID0530,RA04
sub-RID0530,RA05
sub-RID0530,RB01
sub-RID0530,RB02
sub-RID0530,RB03
sub-RID0530,RB04
sub-RID0530,RB05
sub-RID0536,LO01
sub-RID0536,LO02
sub-RID0536,LO03
sub-RID0536,LO04
sub-RID0536,LO05
sub-RID0536,RG03
sub-RID0536,RG04
sub-RID0536,RG05
sub-RID0536,RG06
sub-RID0536,RG08
sub-RID0536,RJ06
sub-RID0536,RJ07
sub-RID0560,LA01
sub-RID0560,LA02
sub-RID0560,LA03
sub-RID0560,LA04
sub-RID0560,LB05
sub-RID0560,LB06
sub-RID0560,RB01
sub-RID0560,RB02
sub-RID0560,RB09
sub-RID0560,RB10
sub-RID0560,RB11
sub-RID0560,RC01
sub-RID0560,RC03
sub-RID0560,RE08
sub-RID0560,RF07
sub-RID0560,RG08
sub-RID0562,RD06
sub-RID0562,RD07
sub-RID0562,RD08
sub-RID0566,LC02
sub-RID0566,LC03
sub-RID0566,LC04
sub-RID0566,LC05
sub-RID0566,LC06
sub-RID0572,RB01
sub-RID0572,RB02
sub-RID0572,RB03
sub-RID0572,RG05
sub-RID0572,RG06
sub-RID0582,LA01
sub-RID0582,LA02
sub-RID0582,LA03
sub-RID0582,RB03
sub-RID0582,RB04
sub-RID0582,RB05
sub-RID0583,LC01
sub-RID0583,LC02
sub-RID0583,LC03
sub-RID0583,LD01
sub-RID0583,LD02
sub-RID0583,LD03
sub-RID0583,LG05
sub-RID0583,LG06
sub-RID0583,LL03
sub-RID0583,LL04
sub-RID0588,LA01
sub-RID0588,LA02
sub-RID0588,LB02
sub-RID0588,LB03
sub-RID0588,RA01
sub-RID0588,RA02
sub-RID0588,RA03
sub-RID0588,RA04
sub-RID0588,RA05
sub-RID0588,RB01
sub-RID0588,RB02
sub-RID0595,LB01
sub-RID0595,LC01
sub-RID0595,LI01
sub-RID0595,LJ02
sub-RID0595,LK02
sub-RID0595,LL01
sub-RID0596,LA01
sub-RID0596,LB02
sub-RID0596,LC02
sub-RID0617,LB01
sub-RID0617,LB02
sub-RID0617,LB03
sub-RID0617,LB04
sub-RID0617,LB05
sub-RID0617,LC01
sub-RID0617,LC02
sub-RID0617,LC03
sub-RID0617,LC04
sub-RID0617,RB02
sub-RID0617,RB03
sub-RID0617,RC02
sub-RID0621,RA02
sub-RID0621,RA07
sub-RID0621,RB01
sub-RID0621,RB02
sub-RID0621,RB03
sub-RID0621,RC02
sub-RID0646,LB02
sub-RID0647,LB05
sub-RID0647,LB06
sub-RID0647,LC05
sub-RID0647,LC06
sub-RID0648,LA01
sub-RID0648,LB01
sub-RID0648,LC01
sub-RID0648,LJ01
sub-RID0648,LK05
sub-RID0648,LK06
sub-RID0648,LK07
sub-RID0649,LA01
sub-RID0649,LA02
sub-RID0649,LA03
sub-RID0649,LA04
sub-RID0649,LA05
sub-RID0649,LH03
sub-RID0649,LH04
sub-RID0649,LH05
sub-RID0649,LH06
sub-RID0649,LJ01
sub-RID0649,LJ02
sub-RID0649,LJ03
sub-RID0649,LJ04
sub-RID0650,LA03
sub-RID0650,LA04
sub-RID0650,LB01
sub-RID0650,LB02
sub-RID0650,LC01
sub-RID0650,LC02
sub-RID0650,LI01
sub-RID0650,LI02
sub-RID0650,LI03
sub-RID0650,LI04
sub-RID0650,RA01
sub-RID0650,RA02
sub-RID0650,RA03
sub-RID0650,RA04
sub-RID0650,RB01
sub-RID0650,RB02
sub-RID0650,RB03
sub-RID0650,RB04
sub-RID0650,RC01
sub-RID0650,RC03
sub-RID0650,RH05
sub-RID0650,RH06
sub-RID0650,RH07
sub-RID0650,RH08
sub-RID0650,RH09
sub-RID0652,LB01
sub-RID0652,LB02
sub-RID0652,RB01
sub-RID0652,RB02
sub-RID0658,LB01
sub-RID0658,LB02
sub-RID0658,LC01
sub-RID0658,LC02
sub-RID0677,RA01
sub-RID0677,RA02
sub-RID0677,RB02
sub-RID0677,RB03
sub-RID0677,RB04
sub-RID0677,RD01
sub-RID0677,RD02
sub-RID0677,RD04
sub-RID0679,RA01
sub-RID0679,RA02
sub-RID0679,RB01
sub-RID0679,RB02
sub-RID0679,RC01
sub-RID0679,RC02
sub-RID0695,RB02
sub-RID0695,RB03
sub-RID0695,RC02
sub-RID0695,RC03
sub-RID0700,RB01
sub-RID0700,RB02
sub-RID0700,RB03
sub-RID0700,RB04
sub-RID0700,RT04
sub-RID0700,RT05
sub-RID0700,RT06
sub-RID0785,LB03
sub-RID0785,LC03
"""


# ── Helpers ────────────────────────────────────────────────────────────────────

def normalize_channel_name(name: str) -> str:
    """Strip leading zeros from numeric parts so LA01 == LA1."""
    parts = re.findall(r"[A-Za-z]+|\d+", name)
    return "".join(str(int(p)) if p.isdigit() else p for p in parts)


def match_channel(target: str, candidates: list[str]) -> str | None:
    """Return the first candidate that matches *target* after normalisation."""
    t = normalize_channel_name(target.upper())
    for c in candidates:
        if normalize_channel_name(c.upper()) == t:
            return c
    return None


# Shared chain definitions (alias-aware left/right option lists).
# Subtemporal (inferior) chains — only appear when F9/T9/P9 (or F10/T10/P10) exist.
_CHAIN_SUBTEMPORAL_L: list[tuple[list[str], list[str]]] = [
    (["Fp1"], ["F9"]),
    (["F9"], ["T9"]),
    (["T9"], ["P9"]),
]
_CHAIN_SUBTEMPORAL_R: list[tuple[list[str], list[str]]] = [
    (["Fp2"], ["F10"]),
    (["F10"], ["T10"]),
    (["T10"], ["P10"]),
]
_CHAIN_TEMPORAL_L: list[tuple[list[str], list[str]]] = [
    (["Fp1"], ["F7"]),
    (["F7"], ["T3", "T7"]),
    (["T3", "T7"], ["T5", "P7"]),
    (["T5", "P7"], ["O1"]),
]
_CHAIN_TEMPORAL_R: list[tuple[list[str], list[str]]] = [
    (["Fp2"], ["F8"]),
    (["F8"], ["T4", "T8"]),
    (["T4", "T8"], ["T6", "P8"]),
    (["T6", "P8"], ["O2"]),
]
_CHAIN_PARASAG_L: list[tuple[list[str], list[str]]] = [
    (["Fp1"], ["F3"]),
    (["F3"], ["C3"]),
    (["C3"], ["P3"]),
    (["P3"], ["O1"]),
]
_CHAIN_PARASAG_R: list[tuple[list[str], list[str]]] = [
    (["Fp2"], ["F4"]),
    (["F4"], ["C4"]),
    (["C4"], ["P4"]),
    (["P4"], ["O2"]),
]

# Central chain below T6–O2 (Cz–Pz only when Pz is present in the recording).
_CHAIN_CENTRAL: list[tuple[list[str], list[str]]] = [
    (["Fz"], ["Cz"]),
    (["Cz"], ["Pz"]),
]

# Longitudinal bipolar montage: subtemporal L/R → temporal L/R → parasag L/R → central.
BANANA_CHAIN_GROUPS: list[tuple[str, list[tuple[list[str], list[str]]]]] = [
    ("subtemporal_left", _CHAIN_SUBTEMPORAL_L),
    ("subtemporal_right", _CHAIN_SUBTEMPORAL_R),
    ("temporal_left", _CHAIN_TEMPORAL_L),
    ("temporal_right", _CHAIN_TEMPORAL_R),
    ("parasag_left", _CHAIN_PARASAG_L),
    ("parasag_right", _CHAIN_PARASAG_R),
    ("central", _CHAIN_CENTRAL),
]

# AP bipolar: L temp → L central → R temp → R central (legacy).
_CHAIN_MIDLINE: list[tuple[list[str], list[str]]] = [
    (["Fz"], ["Cz"]),
]

# AP bipolar: same chains, clinical ordering L temp → L central → R temp → R central → midline.
AP_BIPOLAR_CHAIN_GROUPS: list[tuple[str, list[tuple[list[str], list[str]]]]] = [
    ("temporal_left", _CHAIN_TEMPORAL_L),
    ("parasag_left", _CHAIN_PARASAG_L),
    ("temporal_right", _CHAIN_TEMPORAL_R),
    ("parasag_right", _CHAIN_PARASAG_R),
    ("midline", _CHAIN_MIDLINE),
]


def _compute_chain_groups_montage(
    data: np.ndarray,
    channel_names: list[str],
    groups: list[tuple[str, list[tuple[list[str], list[str]]]]],
    *,
    include_spacers: bool = True,
) -> tuple[np.ndarray, list[str], set[str]]:
    """Build longitudinal bipolar chains; optional spacers between chain groups."""
    norm_to_actual: dict[str, str] = {
        normalize_channel_name(ch.upper()): ch for ch in channel_names
    }
    name_to_idx: dict[str, int] = {ch: i for i, ch in enumerate(channel_names)}

    def resolve(options: list[str]) -> str | None:
        for opt in options:
            key = normalize_channel_name(opt.upper())
            if key in norm_to_actual:
                return norm_to_actual[key]
        return None

    montage_cols: list[np.ndarray] = []
    montage_names: list[str] = []
    midline_pairs: set[str] = set()
    spacer_count = 0
    for gidx, (group_name, chains) in enumerate(groups):
        if include_spacers and gidx > 0:
            spacer_count += 1
            montage_cols.append(np.zeros(data.shape[0], dtype=data.dtype))
            montage_names.append(f"__SPACER_{spacer_count}__")
        for left_opts, right_opts in chains:
            left = resolve(left_opts)
            right = resolve(right_opts)
            if left is None or right is None:
                continue
            pair_name = f"{left}-{right}"
            montage_cols.append(
                data[:, name_to_idx[left]] - data[:, name_to_idx[right]]
            )
            montage_names.append(pair_name)
            if group_name == "midline":
                midline_pairs.add(pair_name)

    if not montage_cols:
        return data, channel_names, set()

    return np.column_stack(montage_cols).astype(data.dtype), montage_names, midline_pairs


def load_soz_csv() -> pd.DataFrame:
    """Parse the SOZ electrode table from the embedded string constant."""
    return pd.read_csv(io.StringIO(_SOZ_CSV_DATA))


def _format_clip_timestamp_sec(timestamp_sec: float) -> str:
    """Format timestamp like spike clip EDF names (e.g. 32621.8438)."""
    return f"{float(timestamp_sec):.4f}".rstrip("0").rstrip(".")


def clip_display_stem(ieeg_file_name: str, timestamp_sec: float) -> str:
    """Human-readable clip id (e.g. EMU1049_Day08_1_32621.8438)."""
    return f"{ieeg_file_name}_{_format_clip_timestamp_sec(timestamp_sec)}"


def clip_annotation_stem(ieeg_file_name: str, timestamp_sec: float) -> str:
    """Recording id for annotation JSON (e.g. EMU1049_Day08_1_32621.8438.edf)."""
    return f"{clip_display_stem(ieeg_file_name, timestamp_sec)}.edf"


def load_clip_shuffle_key(folder: Path) -> dict[str, dict]:
    """Map clip file stem (clip_001 or EMU…_timestamp) → shuffle-key row."""
    key_path = folder / "clip_shuffle_key.csv"
    if not key_path.is_file():
        return {}
    by_stem: dict[str, dict] = {}
    with key_path.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            out = (row.get("output_name") or "").strip()
            if out:
                by_stem[Path(out).stem] = row
            ieeg = (row.get("ieeg_file_name") or "").strip()
            ts_raw = (row.get("timestamp_sec") or "").strip()
            if ieeg and ts_raw:
                by_stem[clip_display_stem(ieeg, float(ts_raw))] = row
    return by_stem


# ── Channel Annotation Dialog ──────────────────────────────────────────────────

class ChannelAnnotationDialog(QDialog):
    """
    Modal dialog that pops up when the clinician clicks or drags on a channel
    trace.  Lets the clinician describe the seizure pattern using structured
    controls: LVFA checkbox, Frequency/Pattern/Modifier dropdowns.
    """

    def __init__(self, channel_names: str | list[str], time_sec: float,
                 existing_types: dict | list,
                 end_sec: float | None = None, parent=None):
        super().__init__(parent)

        # Normalise to list
        if isinstance(channel_names, str):
            channel_names = [channel_names]

        display_name = ", ".join(channel_names)
        plural = len(channel_names) > 1
        title = f"Annotate Channel{'s' if plural else ''}: {display_name}"
        if len(title) > 80:
            title = f"Annotate {len(channel_names)} Channels"
        self.setWindowTitle(title)
        self.setMinimumWidth(360)

        layout = QVBoxLayout(self)

        if end_sec is not None:
            time_text = f"{time_sec:.3f} s – {end_sec:.3f} s"
        else:
            time_text = f"{time_sec:.3f} s"

        ch_label = "Channels" if plural else "Channel"
        info_label = QLabel(
            f"<b>{ch_label}:</b> {display_name}<br>"
            f"<b>Time:</b> {time_text}"
        )
        info_label.setStyleSheet("font-size: 11pt; padding: 4px;")
        layout.addWidget(info_label)

        layout.addWidget(QLabel("Describe seizure pattern:"))

        # ── LVFA checkbox ─────────────────────────────────────────────────
        self.lvfa_cb = QCheckBox("LVFA (Low-voltage fast activity)")
        layout.addWidget(self.lvfa_cb)

        # ── Frequency dropdown ────────────────────────────────────────────
        freq_row = QHBoxLayout()
        freq_row.addWidget(QLabel("Frequency:"))
        self.frequency_combo = QComboBox()
        self.frequency_combo.addItems(FREQUENCY_OPTIONS)
        freq_row.addWidget(self.frequency_combo)
        freq_row.addStretch()
        layout.addLayout(freq_row)

        # ── Pattern dropdown ──────────────────────────────────────────────
        pat_row = QHBoxLayout()
        pat_row.addWidget(QLabel("Pattern:"))
        self.pattern_combo = QComboBox()
        self.pattern_combo.addItems(PATTERN_OPTIONS)
        pat_row.addWidget(self.pattern_combo)
        pat_row.addStretch()
        layout.addLayout(pat_row)

        # ── Modifier dropdown ─────────────────────────────────────────────
        mod_row = QHBoxLayout()
        mod_row.addWidget(QLabel("Modifier:"))
        self.modifier_combo = QComboBox()
        self.modifier_combo.addItems(MODIFIER_OPTIONS)
        mod_row.addWidget(self.modifier_combo)
        mod_row.addStretch()
        layout.addLayout(mod_row)

        # ── Custom type text field ────────────────────────────────────────
        type_row = QHBoxLayout()
        type_row.addWidget(QLabel("Type:"))
        self.custom_type_edit = QLineEdit()
        self.custom_type_edit.setPlaceholderText("Custom annotation type (optional)")
        type_row.addWidget(self.custom_type_edit)
        layout.addLayout(type_row)

        # ── Pre-populate from existing annotation ─────────────────────────
        if isinstance(existing_types, dict):
            self.lvfa_cb.setChecked(existing_types.get("lvfa", False))
            freq = existing_types.get("frequency", "(none)")
            idx = self.frequency_combo.findText(freq)
            if idx >= 0:
                self.frequency_combo.setCurrentIndex(idx)
            pat = existing_types.get("pattern", "(none)")
            idx = self.pattern_combo.findText(pat)
            if idx >= 0:
                self.pattern_combo.setCurrentIndex(idx)
            mod = existing_types.get("modifier", "None")
            idx = self.modifier_combo.findText(mod)
            if idx >= 0:
                self.modifier_combo.setCurrentIndex(idx)
            self.custom_type_edit.setText(existing_types.get("custom_type", ""))

        # ── Buttons ───────────────────────────────────────────────────────
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    @property
    def selected_types(self) -> dict:
        """Return structured annotation dict."""
        return {
            "lvfa": self.lvfa_cb.isChecked(),
            "frequency": self.frequency_combo.currentText(),
            "pattern": self.pattern_combo.currentText(),
            "modifier": self.modifier_combo.currentText(),
            "custom_type": self.custom_type_edit.text().strip(),
        }


class BananaChannelSelectionDialog(QDialog):
    """Simple multi-select dialog for banana montage channel annotation."""

    def __init__(self, all_contacts: list[str], preselected: list[str], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Banana Contacts")
        self.setMinimumWidth(320)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Select one or more banana contacts (e.g. Fp2, F8):"))

        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QAbstractItemView.MultiSelection)
        for contact in all_contacts:
            item = QListWidgetItem(contact)
            self.list_widget.addItem(item)
            if contact in preselected:
                item.setSelected(True)
        layout.addWidget(self.list_widget)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    @property
    def selected_contacts(self) -> list[str]:
        return [i.text() for i in self.list_widget.selectedItems()]


# ── Main GUI ───────────────────────────────────────────────────────────────────

class SeizureAnnotationGUI(QMainWindow):
    """Main window for seizure annotation."""

    def __init__(
        self,
        data_dir: str | None = None,
        ieeg_dataset_id: str | None = None,
        ieeg_username: str | None = None,
        ieeg_password: str | None = None,
        patient_id: str | None = None,
        ictal_onset_sec: float | None = None,
        ictal_duration_sec: float | None = None,
        ieeg_patient_datasets_csv: str | None = None,
        ieeg_patient_col: str = "patient_id",
        ieeg_dataset_col: str = "ieeg_dataset_id",
        ieeg_edf_options_csv: str | None = None,
        ieeg_edf_filename_col: str = "Filename",
        ieeg_edf_split_token: str = "_spike_at_",
        ieeg_edf_trim_ext: str = ".edf",
        clips_csv: str | None = None,
        clip_edf_dir: str | None = None,
        clip_scan_dir: str | None = None,
    ):
        super().__init__()

        # ── paths ──────────────────────────────────────────────────────────────
        self.data_dir: Path | None = Path(data_dir) if data_dir else None
        self.ieeg_dataset_id: str | None = ieeg_dataset_id
        self.ieeg_username: str | None = ieeg_username
        self.ieeg_password: str | None = ieeg_password
        # iEEG mode can be entered via:
        # - explicit dataset id
        # - CSV mapping (patient -> dataset ids)
        # - CSV listing (selections.csv -> dataset ids)
        # - clips CSV (final_selected_200_events.csv)
        # - clip folder scan (blinded clip_*.edf review, no CSV)
        self.clips_csv: Path | None = Path(clips_csv) if clips_csv else None
        self.clips_folder_dir: Path | None = (
            Path(clip_scan_dir) if clip_scan_dir else None
        )
        self._clips_review_ui_configured = False
        # Folder-scan reviews load their EDFs straight from the scanned folder.
        if clip_edf_dir:
            self.clip_edf_dir: Path | None = Path(clip_edf_dir)
        elif self.clips_folder_dir is not None:
            self.clip_edf_dir = self.clips_folder_dir
        else:
            self.clip_edf_dir = None
        self.clip_metadata_by_label: dict[str, dict] = {}
        self.ieeg_mode: bool = (
            self.ieeg_dataset_id is not None
            or ieeg_patient_datasets_csv is not None
            or ieeg_edf_options_csv is not None
            or self.clips_csv is not None
            or self.clips_folder_dir is not None
        )
        self.patient_id_for_annotations: str = patient_id or "ieeg"
        self.ieeg_ictal_onset_sec: float | None = ictal_onset_sec
        self.ieeg_ictal_duration_sec: float | None = ictal_duration_sec
        self.ieeg_patient_datasets_csv: Path | None = (
            Path(ieeg_patient_datasets_csv) if ieeg_patient_datasets_csv else None
        )
        self.ieeg_patient_col: str = ieeg_patient_col
        self.ieeg_dataset_col: str = ieeg_dataset_col
        self.ieeg_patient_datasets: dict[str, list[str]] = {}
        self.ieeg_edf_options_csv: Path | None = (
            Path(ieeg_edf_options_csv) if ieeg_edf_options_csv else None
        )
        self.ieeg_edf_filename_col: str = ieeg_edf_filename_col
        self.ieeg_edf_split_token: str = ieeg_edf_split_token
        self.ieeg_edf_trim_ext: str = ieeg_edf_trim_ext
        self.ieeg_spike_time_by_dataset: dict[str, float] = {}
        self.ieeg_selection_options: dict[str, dict] = {}
        ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)

        # ── patient / recording state ──────────────────────────────────────────
        self.patients: list[str] = []          # sorted list of sub-RID* folder names
        self.current_patient: str | None = None
        self.recordings: list[str] = []        # ictal EDF basenames for current patient
        self.current_recording: str | None = None
        self.patient_dir: Path | None = None   # resolved patient session dir

        # ── raw data ──────────────────────────────────────────────────────────
        self.eeg_data: np.ndarray | None = None     # (n_samples, n_channels) µV
        self.channel_names_all: list[str] = []      # all SEEG/ECOG channels
        self.fs: float = 1000.0
        self.total_duration: float = 0.0
        self.time_offset_sec: float = 0.0           # absolute start time of loaded window
        self.clip_spike_abs_sec: float | None = None   # spike center in recording time (x-axis)
        self.clip_spike_rel_sec: float | None = None   # spike center relative to clip start
        self.ictal_onset: float | None = None
        self.ictal_duration: float | None = None

        # ── SOZ lookup ────────────────────────────────────────────────────────
        self.soz_df: pd.DataFrame = load_soz_csv()
        self.soz_channels: set[str] = set()

        # ── view state ────────────────────────────────────────────────────────
        self.view_mode: str = "all"               # "all" | "12ch" | "soz"
        self.v_scroll_offset: int = 0            # 12ch mode: bottom-most visible channel index
        self.displayed_channels: list[str] = []  # channels currently shown (plot rows)
        # Channels checked in the right-hand list (subset of current montage); unchecked = hidden.
        self._visible_channel_names: set[str] = set()
        self.gain: float = 1.0
        self.scroll_pos: float = 0.0             # seconds from start

        # ── annotations ───────────────────────────────────────────────────────
        # global seizure type for this recording (structured dict)
        self.global_types: dict = {
            "lvfa": False, "frequency": "(none)",
            "pattern": "(none)", "modifier": "None", "custom_type": "",
        }
        # channel-level: list of {"channel": str, "time_sec": float, "types": {dict}}
        self.channel_annotations: list[dict] = []

        # ── plot items ────────────────────────────────────────────────────────
        self.trace_items: list[pg.PlotDataItem] = []
        self.annotation_markers: list = []           # ScatterPlot, LinearRegion, TextItem
        self.ictal_region: pg.LinearRegionItem | None = None

        # ── montage ───────────────────────────────────────────────────────────
        self.montage: str = "car"                     # car | banana | bipolar | ap_bipolar
        self._ref_eeg_data_base: np.ndarray | None = None  # referential µV (as loaded)
        self._ref_eeg_data: np.ndarray | None = None
        self._ref_channel_names: list[str] = []
        self.eeg_data_car: np.ndarray | None = None
        self.channel_names_car: list[str] = []
        self.eeg_data_bipolar: np.ndarray | None = None
        self.channel_names_bipolar: list[str] = []
        self.eeg_data_banana: np.ndarray | None = None
        self.channel_names_banana: list[str] = []
        self.banana_midline_pairs: set[str] = set()
        self.eeg_data_ap_bipolar: np.ndarray | None = None
        self.channel_names_ap_bipolar: list[str] = []
        self.ap_bipolar_midline_pairs: set[str] = set()

        # ── drag-to-annotate state ────────────────────────────────────────
        self._drag_start_data: tuple | None = None   # (time,) in data coords
        self._drag_channel_idx: int | None = None     # channel index at press
        self._drag_region: pg.PlotCurveItem | None = None   # bounded rect (fill + border)
        self._drag_start_line: pg.InfiniteLine | None = None  # visual feedback

        # ── cursor line ───────────────────────────────────────────────────
        self._cursor_line: pg.InfiniteLine | None = None
        # auto-scroll during drag
        self._drag_scroll_timer: QTimer = QTimer(self)
        self._drag_scroll_timer.setInterval(50)       # 50 ms ticks
        self._drag_scroll_timer.timeout.connect(self._on_drag_auto_scroll)
        self._drag_scroll_dir: int = 0                # -1 left, 0 none, +1 right
        self._drag_last_scene_pos = None              # latest scene pos for timer

        self._build_ui()
        self._setup_shortcuts()

        self.montage = "banana"

        if self.data_dir:
            self._discover_patients()
        elif self.ieeg_mode:
            self._init_ieeg_recording()

    # ── UI Construction ────────────────────────────────────────────────────────

    def _build_ui(self):
        self.setWindowTitle("Seizure Annotation GUI")
        self.setGeometry(80, 80, 1700, 950)

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setSpacing(4)

        # ── Row 1: patient / recording / view selectors ────────────────────────
        row1 = QHBoxLayout()

        row1.addWidget(QLabel("Patient:"))
        self.patient_combo = QComboBox()
        self.patient_combo.setMinimumWidth(320)
        self.patient_combo.currentTextChanged.connect(self._on_patient_changed)
        row1.addWidget(self.patient_combo)

        self.btn_prev_patient = QPushButton("◀ Prev")
        self.btn_next_patient = QPushButton("Next ▶")
        self.btn_prev_patient.setToolTip("Previous entry in patient list")
        self.btn_next_patient.setToolTip("Next entry in patient list")
        self.btn_prev_patient.clicked.connect(self._on_prev_patient)
        self.btn_next_patient.clicked.connect(self._on_next_patient)
        row1.addWidget(self.btn_prev_patient)
        row1.addWidget(self.btn_next_patient)

        self.btn_ieeg_no_spike = QPushButton("No spike")
        self.btn_ieeg_no_spike.setToolTip(
            "Mark this clip as no visible spike (saved in annotations). "
            "In legacy selections.csv mode, also toggles loading the first 14 s of the recording."
        )
        if self.clips_mode:
            self.btn_ieeg_no_spike.clicked.connect(self._on_no_spike_clicked)
        else:
            self.btn_ieeg_no_spike.setCheckable(True)
            self.btn_ieeg_no_spike.toggled.connect(self._on_ieeg_no_spike_toggled)
        if self.ieeg_mode:
            row1.addWidget(self.btn_ieeg_no_spike)

        self.clip_info_label = QLabel("")
        self.clip_info_label.hide()  # clips mode: label is only in the dropdown

        self.recording_label = QLabel("Recording:")
        row1.addWidget(self.recording_label)
        self.recording_combo = QComboBox()
        self.recording_combo.setMinimumWidth(280)
        self.recording_combo.currentTextChanged.connect(self._on_recording_changed)
        row1.addWidget(self.recording_combo)

        # In selections/clips CSV iEEG mode, "patient" is the only selector.
        if self.ieeg_mode and (self.ieeg_edf_options_csv or self.clips_mode):
            self.recording_label.hide()
            self.recording_combo.hide()

        if not self.data_dir and not self.ieeg_mode:
            browse_btn = QPushButton("Open Data Folder…")
            browse_btn.clicked.connect(self._browse_data_dir)
            row1.addWidget(browse_btn)

        row1.addSpacing(20)

        view_group = QGroupBox("View")
        vgl = QHBoxLayout(view_group)
        self.btn_view_all = QPushButton("All Channels")
        self.btn_view_all.setEnabled(False)
        vgl.addWidget(self.btn_view_all)
        row1.addWidget(view_group)
        self.view_mode = "all"

        row1.addSpacing(10)

        row1.addWidget(QLabel("Montage:"))
        self.montage_combo = QComboBox()
        self.montage_combo.addItem("Common average", "car")
        self.montage_combo.addItem("Bipolar", "banana")
        self.montage_combo.setCurrentIndex(1)
        self.montage_combo.setToolTip(
            "Common average: each scalp channel minus the mean across all channels. "
            "Bipolar: subtemporal → temporal → parasagittal chains."
        )
        self.montage_combo.currentIndexChanged.connect(self._on_montage_combo_changed)
        row1.addWidget(self.montage_combo)

        row1.addSpacing(10)

        row1.addWidget(QLabel("Scale:"))
        self.gain_label = QLabel("— µV/mm")
        self.gain_label.setMinimumWidth(110)
        self.gain_label.setStyleSheet("font-weight: bold;")
        row1.addWidget(self.gain_label)
        row1.addWidget(QLabel("(↑↓)"))

        row1.addStretch()

        root.addLayout(row1)

        # ── Row 2: global seizure type controls ──────────────────────────────
        self.global_group = QGroupBox("Global Seizure Type")
        gl = QHBoxLayout(self.global_group)

        self.global_lvfa_cb = QCheckBox("LVFA")
        self.global_lvfa_cb.stateChanged.connect(self._on_global_type_changed)
        gl.addWidget(self.global_lvfa_cb)

        gl.addSpacing(12)
        gl.addWidget(QLabel("Frequency:"))
        self.global_frequency_combo = QComboBox()
        self.global_frequency_combo.addItems(FREQUENCY_OPTIONS)
        self.global_frequency_combo.currentIndexChanged.connect(self._on_global_type_changed)
        gl.addWidget(self.global_frequency_combo)

        gl.addSpacing(12)
        gl.addWidget(QLabel("Pattern:"))
        self.global_pattern_combo = QComboBox()
        self.global_pattern_combo.addItems(PATTERN_OPTIONS)
        self.global_pattern_combo.currentIndexChanged.connect(self._on_global_type_changed)
        gl.addWidget(self.global_pattern_combo)

        gl.addSpacing(12)
        gl.addWidget(QLabel("Modifier:"))
        self.global_modifier_combo = QComboBox()
        self.global_modifier_combo.addItems(MODIFIER_OPTIONS)
        self.global_modifier_combo.currentIndexChanged.connect(self._on_global_type_changed)
        gl.addWidget(self.global_modifier_combo)

        gl.addSpacing(12)
        gl.addWidget(QLabel("Type:"))
        self.global_custom_type_edit = QLineEdit()
        self.global_custom_type_edit.setPlaceholderText("Custom type (optional)")
        self.global_custom_type_edit.setMaximumWidth(180)
        self.global_custom_type_edit.textChanged.connect(self._on_global_type_changed)
        gl.addWidget(self.global_custom_type_edit)

        gl.addStretch()
        root.addWidget(self.global_group)
        # Annotation is now quick "mark + save"; hide legacy global-type controls.
        self.global_group.hide()

        # ── Main split: EEG plot | right panel ────────────────────────────────
        splitter = QSplitter(Qt.Horizontal)

        # EEG plot + vertical scrollbar
        plot_container = QWidget()
        plot_hbox = QHBoxLayout(plot_container)
        plot_hbox.setContentsMargins(0, 0, 0, 0)
        plot_hbox.setSpacing(0)

        self.plot_widget = pg.GraphicsLayoutWidget()
        self.plot_widget.setBackground("w")
        self.plot = self.plot_widget.addPlot()
        self.plot.showGrid(x=True, y=False, alpha=0.25)
        self.plot.setLabel("bottom", "Time (s)")
        self.plot.getAxis("left").setStyle(tickLength=0)

        # Fixed y-axis (no auto-range on y)
        self.plot.setMouseEnabled(x=False, y=False)
        self.plot.scene().installEventFilter(self)
        self.plot.vb.sigRangeChanged.connect(self._update_scale_display)
        plot_hbox.addWidget(self.plot_widget)

        # Vertical scrollbar for 12-channel mode
        self.v_scroll_bar = QScrollBar(Qt.Vertical)
        self.v_scroll_bar.setMinimum(0)
        self.v_scroll_bar.setMaximum(0)
        self.v_scroll_bar.setValue(0)
        self.v_scroll_bar.valueChanged.connect(self._on_v_scroll)
        self.v_scroll_bar.hide()  # only visible in "12ch" mode
        plot_hbox.addWidget(self.v_scroll_bar)

        splitter.addWidget(plot_container)

        # Right panel
        right_panel = QWidget()
        right_panel.setMaximumWidth(420)
        right_panel.setMinimumWidth(300)
        rpl = QVBoxLayout(right_panel)

        rpl.addWidget(QLabel("<b>Show channels</b>"))
        self.ch_search = QLineEdit()
        self.ch_search.setPlaceholderText("Search channels…")
        self.ch_search.textChanged.connect(self._filter_channel_list)
        rpl.addWidget(self.ch_search)

        self.ch_list = QListWidget()
        self.ch_list.setSelectionMode(QAbstractItemView.NoSelection)
        self.ch_list.itemChanged.connect(self._on_channel_list_item_changed)
        _chfont = QFont("Monospace", 8)
        self.ch_list.setFont(_chfont)
        rpl.addWidget(self.ch_list)
        self.clips_channel_help_label = QLabel(
            "<i>Uncheck to hide from montage. 10–20 map: click to mark. "
            "Trace row click: remove marks on that row.</i>"
        )
        rpl.addWidget(self.clips_channel_help_label)

        rpl.addSpacing(10)
        rpl.addWidget(QLabel("<b>Channel Annotations</b>"))
        self.annot_list = QListWidget()
        self.annot_list.setMaximumHeight(200)
        self.annot_list.itemDoubleClicked.connect(self._remove_channel_annotation)
        rpl.addWidget(self.annot_list)
        rpl.addWidget(QLabel("<i>Double-click annotation to delete</i>"))
        rpl.addStretch(1)

        self.montage_1020_title_label = QLabel(
            "<b>10–20 schematic</b> <small>(vector)</small>"
        )
        rpl.addWidget(self.montage_1020_title_label)
        self.montage_1020_label = TenTwentyShapeMontageWidget()
        self.montage_1020_label.setMinimumHeight(420)
        self.montage_1020_label.setMaximumHeight(560)
        self.montage_1020_label.clicked_contact.connect(self._on_montage_1020_contact_clicked)
        rpl.addWidget(self.montage_1020_label)

        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        root.addWidget(splitter, stretch=1)

        # ── Scrollbar ─────────────────────────────────────────────────────────
        scroll_row = QHBoxLayout()
        scroll_row.addWidget(QLabel("Time:"))
        self.scroll_bar = QScrollBar(Qt.Horizontal)
        self.scroll_bar.setMinimum(0)
        self.scroll_bar.setMaximum(1000)
        self.scroll_bar.setValue(0)
        self.scroll_bar.valueChanged.connect(self._on_scroll)
        scroll_row.addWidget(self.scroll_bar)
        root.addLayout(scroll_row)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        if self.ieeg_mode:
            self.status_bar.showMessage(
                "10–20 map: mark channels. Trace row click: clear marks (auto-saved).  "
                "↑↓ amplitude.  ←→ scroll time (0.25 s, 3 s window)."
            )
        else:
            self.status_bar.showMessage(
                "Open a data folder or pass --data_dir.  "
                "10–20 map: mark channels. Trace row click: clear marks (auto-saved).  "
                "↑↓ amplitude.  ←→ scroll time (0.25 s, 3 s window)."
            )

    def _setup_shortcuts(self):
        QShortcut(QKeySequence(Qt.Key_Up), self, self._increase_gain)
        QShortcut(QKeySequence(Qt.Key_Down), self, self._decrease_gain)
        QShortcut(QKeySequence(Qt.Key_Left), self, lambda: self._nudge_time_scroll(-1))
        QShortcut(QKeySequence(Qt.Key_Right), self, lambda: self._nudge_time_scroll(1))

    @property
    def clips_mode(self) -> bool:
        """True for both CSV-driven and folder-scan clip review."""
        return self.clips_csv is not None or self.clips_folder_dir is not None

    def _init_ieeg_recording(self):
        """Populate UI controls for a single iEEG.org dataset."""
        if self.clips_csv:
            self._init_from_clips_csv()
            return
        if self.clips_folder_dir:
            self._init_from_clip_folder()
            return

        if self.ieeg_patient_datasets_csv:
            self._init_ieeg_from_patient_dataset_csv()
            return

        if self.ieeg_edf_options_csv:
            self._init_ieeg_from_edf_options_csv()
            return

        # Backwards-compatible single-dataset mode.
        self.current_patient = self.patient_id_for_annotations
        self.current_recording = self.ieeg_dataset_id
        self.patients = [self.current_patient]
        self.recordings = [self.ieeg_dataset_id] if self.ieeg_dataset_id else []

        # The combos already exist after _build_ui().
        self.patient_combo.blockSignals(True)
        self.patient_combo.clear()
        self.patient_combo.addItems(self.patients)
        self.patient_combo.setCurrentIndex(0)
        self.patient_combo.blockSignals(False)

        self.recording_combo.blockSignals(True)
        self.recording_combo.clear()
        self.recording_combo.addItems(self.recordings)
        self.recording_combo.setCurrentIndex(0)
        self.recording_combo.blockSignals(False)

        if self.ieeg_dataset_id:
            self._load_ieeg_recording(self.ieeg_dataset_id)

    def _init_ieeg_from_patient_dataset_csv(self):
        """Populate Patient dropdown from a CSV mapping.

        Expected columns:
          - patient column (default: `patient_id`)
          - dataset column (default: `ieeg_dataset_id`)
        """
        import pandas as _pd

        if not self.ieeg_patient_datasets_csv or not self.ieeg_patient_datasets_csv.exists():
            QMessageBox.critical(
                self,
                "iEEG Config Error",
                f"Patient dataset CSV not found: {self.ieeg_patient_datasets_csv}",
            )
            return

        df = _pd.read_csv(self.ieeg_patient_datasets_csv)
        if self.ieeg_patient_col not in df.columns or self.ieeg_dataset_col not in df.columns:
            QMessageBox.critical(
                self,
                "iEEG Config Error",
                f"CSV must include columns `{self.ieeg_patient_col}` and `{self.ieeg_dataset_col}`. "
                f"Found columns: {list(df.columns)}",
            )
            return

        # Build mapping in CSV order, keeping first-seen dataset order per patient.
        patient_to_datasets: dict[str, list[str]] = {}
        for _, row in df.iterrows():
            patient = row[self.ieeg_patient_col]
            ds_id = row[self.ieeg_dataset_col]
            if not isinstance(patient, str) or not isinstance(ds_id, str):
                continue
            patient = patient.strip()
            ds_id = ds_id.strip()
            if not patient or not ds_id:
                continue
            patient_to_datasets.setdefault(patient, [])
            if ds_id not in patient_to_datasets[patient]:
                patient_to_datasets[patient].append(ds_id)

        if not patient_to_datasets:
            QMessageBox.critical(self, "iEEG Config Error", "No valid (patient, dataset) rows found in CSV.")
            return

        # Probe connectivity by opening each patient's first dataset.
        if not self.ieeg_username or not self.ieeg_password:
            QMessageBox.critical(
                self,
                "iEEG Connection Error",
                "Missing iEEG credentials. Provide --ieeg_username and a password via env/--ieeg_password.",
            )
            return

        failed_patients: list[tuple[str, str]] = []  # (patient, dataset_id)
        valid_patients: list[str] = []

        from ieeg.auth import Session

        with Session(self.ieeg_username, self.ieeg_password) as s:
            for patient, datasets in patient_to_datasets.items():
                probe_id = datasets[0]
                try:
                    ds = s.open_dataset(probe_id)
                    # Force basic metadata reads
                    channel_labels = list(ds.get_channel_labels())
                    if not channel_labels:
                        raise ValueError("No channel labels")
                    first_details = ds.get_time_series_details(channel_labels[0])
                    fs_probe = getattr(first_details, "sample_frequency", None)
                    _ = float(
                        fs_probe if fs_probe is not None else first_details.sample_rate
                    )
                    s.close_dataset(ds)
                    valid_patients.append(patient)
                except Exception:
                    failed_patients.append((patient, probe_id))

        if not valid_patients:
            msg = "GUI could not connect to any patient datasets."
            if failed_patients:
                msg += "\n\nFailed connections (first dataset per patient):\n" + "\n".join(
                    f"  - {p}: {d}" for p, d in failed_patients[:50]
                )
            QMessageBox.critical(self, "iEEG Connection Error", msg)
            return

        # Save mapping for runtime patient switching.
        self.ieeg_patient_datasets = {p: patient_to_datasets[p] for p in valid_patients}

        self.current_patient = valid_patients[0]
        self.patient_id_for_annotations = self.current_patient
        self.patients = valid_patients
        self.recordings = self.ieeg_patient_datasets[self.current_patient]

        # Populate combos
        self.patient_combo.blockSignals(True)
        self.patient_combo.clear()
        self.patient_combo.addItems(self.patients)
        self.patient_combo.setCurrentIndex(0)
        self.patient_combo.blockSignals(False)

        self.recording_combo.blockSignals(True)
        self.recording_combo.clear()
        self.recording_combo.addItems(self.recordings)
        self.recording_combo.setCurrentIndex(0)
        self.recording_combo.blockSignals(False)

        first_dataset = self.recordings[0] if self.recordings else None
        if first_dataset:
            # Load first dataset; if it fails, the load method will show the error.
            self.current_recording = first_dataset
            self._load_ieeg_recording(first_dataset)

        # Inform user of skipped patients.
        if failed_patients:
            failed_list = "\n".join(f"  - {p}: {d}" for p, d in failed_patients)
            self.status_bar.showMessage(
                f"Skipped {len(failed_patients)} patients due to iEEG connection issues."
            )
            QMessageBox.information(
                self,
                "iEEG Connection Skips",
                "The GUI couldn't connect to some patients. These patients were skipped:\n" + failed_list,
            )

    def _init_ieeg_from_edf_options_csv(self):
        """Populate dropdown from `sz-gui/selections.csv`.

        Assumes your CSV has a `Filename` column with values that contain
        something like: `<edf_or_dataset_id>_spike_at_<seconds>.edf`.
        We extract:
          - dataset id: `<edf_or_dataset_id>.edf`
          - spike time: `<seconds>`
        """
        import pandas as _pd

        if not self.ieeg_edf_options_csv or not self.ieeg_edf_options_csv.exists():
            QMessageBox.critical(
                self,
                "iEEG Config Error",
                f"iEEG options CSV not found: {self.ieeg_edf_options_csv}",
            )
            return

        df = _pd.read_csv(self.ieeg_edf_options_csv)
        if self.ieeg_edf_filename_col not in df.columns:
            QMessageBox.critical(
                self,
                "iEEG Config Error",
                f"CSV missing column `{self.ieeg_edf_filename_col}`. Found columns: {list(df.columns)}",
            )
            return

        option_labels: list[str] = []
        parsed_spike_times: dict[str, float] = {}
        selection_options: dict[str, dict] = {}
        trial_counts: dict[str, int] = {}
        for _, row in df.iterrows():
            filename_full = row[self.ieeg_edf_filename_col]
            if not isinstance(filename_full, str):
                continue
            if self.ieeg_edf_split_token not in filename_full:
                continue

            base = filename_full
            if self.ieeg_edf_trim_ext and base.endswith(self.ieeg_edf_trim_ext):
                base = base[: -len(self.ieeg_edf_trim_ext)]
            else:
                base = base.replace(self.ieeg_edf_trim_ext, "")

            # Expect: "<id><token><time>"
            try:
                file_part, time_part = base.split(self.ieeg_edf_split_token, 1)
            except ValueError:
                continue

            file_part = str(file_part).strip()
            # Your iEEG dataset ids include ".edf" in the name.
            dataset_id = (
                f"{file_part}.edf"
                if not str(file_part).lower().endswith(".edf")
                else str(file_part)
            )
            if not dataset_id:
                continue

            try:
                spike_time_sec = float(time_part)
            except Exception:
                continue

            trial_counts[file_part] = trial_counts.get(file_part, 0) + 1
            option_label = f"{file_part}_trial_{trial_counts[file_part]}"
            option_labels.append(option_label)
            parsed_spike_times[option_label] = spike_time_sec
            selection_options[option_label] = {
                "dataset_id": dataset_id,
                "spike_time_sec": spike_time_sec,
                "recording_name": filename_full.strip(),
            }

        if not option_labels:
            QMessageBox.critical(
                self,
                "iEEG Config Error",
                "No dataset ids could be parsed from selections.csv (check Filename values contain the split token).",
            )
            return

        # Lazy mode: do not probe connectivity up-front.
        # Populate options immediately; actual iEEG open/download happens when
        # the user selects a dataset in the dropdown.
        self.ieeg_selection_options = selection_options
        self.ieeg_patient_datasets = {
            label: [selection_options[label]["dataset_id"]] for label in option_labels
        }
        self.ieeg_spike_time_by_dataset = parsed_spike_times.copy()
        self.patients = option_labels
        self.recordings = []
        self.current_patient = None
        self.current_recording = None

        self.patient_combo.blockSignals(True)
        self.patient_combo.clear()
        self.patient_combo.addItem("-- Select Patient --")
        self.patient_combo.addItems(self.patients)
        self.patient_combo.setCurrentIndex(0)
        self.patient_combo.blockSignals(False)
        self._update_patient_dropdown_colors()

        # Recording selector is hidden in this mode.
        self.recording_combo.blockSignals(True)
        self.recording_combo.clear()
        self.recording_combo.blockSignals(False)

        self.status_bar.showMessage(
            f"Loaded {len(option_labels)} patient options from selections.csv. "
            "Select a patient to load iEEG data."
        )

    def _init_from_clips_csv(self):
        """Populate dropdown from final_selected_200_events.csv (200-clip review)."""
        if not self.clips_csv or not self.clips_csv.exists():
            QMessageBox.critical(
                self,
                "Clips CSV Error",
                f"Clips CSV not found: {self.clips_csv}",
            )
            return

        try:
            from ieeg_clip_io import (
                clip_gui_label,
                clip_timestamp_display,
                load_clips_csv,
                row_to_option,
            )
        except ImportError as exc:
            QMessageBox.critical(
                self,
                "Clips CSV Error",
                f"Could not import ieeg_clip_io (add Carlos_GUI to PYTHONPATH): {exc}",
            )
            return

        df = load_clips_csv(self.clips_csv)
        clip_ids: list[str] = []
        selection_options: dict[str, dict] = {}
        parsed_spike_times: dict[str, float] = {}
        metadata_by_label: dict[str, dict] = {}

        for i, row in df.iterrows():
            opt = row_to_option(row, int(i))
            clip_id = opt["annotation_stem"]
            clip_ids.append(clip_id)
            parsed_spike_times[clip_id] = opt["spike_time_sec"]
            selection_options[clip_id] = opt
            metadata_by_label[clip_id] = {
                "row_index": int(i),
                "patient_id": opt.get("patient_id"),
                "ieeg_file_name": opt.get("ieeg_file_name"),
                "timestamp_sec": float(row["timestamp_sec"]),
            }

        self.ieeg_selection_options = selection_options
        self.ieeg_spike_time_by_dataset = parsed_spike_times.copy()
        self.clip_metadata_by_label = metadata_by_label
        self.ieeg_patient_datasets = {
            clip_id: [selection_options[clip_id]["dataset_id"]] for clip_id in clip_ids
        }
        self.patients = clip_ids
        self.recordings = []
        self.current_patient = None
        self.current_recording = None

        self.patient_combo.blockSignals(True)
        self.patient_combo.clear()
        self.patient_combo.addItem("-- Select clip --")
        for clip_id in clip_ids:
            opt = selection_options[clip_id]
            row_i = metadata_by_label[clip_id]["row_index"]
            display = clip_gui_label(
                df.iloc[row_i], row_i
            )
            self.patient_combo.addItem(display, clip_id)
        self.patient_combo.setCurrentIndex(0)
        self.patient_combo.blockSignals(False)
        self._update_patient_dropdown_colors()

        self.recording_combo.blockSignals(True)
        self.recording_combo.clear()
        self.recording_combo.blockSignals(False)

        src = self.clips_csv.name
        edf_hint = (
            f" Local EDF folder: {self.clip_edf_dir}" if self.clip_edf_dir else " (stream from iEEG.org)"
        )
        self._configure_clips_review_ui()
        self.status_bar.showMessage(
            f"Loaded {len(clip_ids)} clips from {src}.{edf_hint} Select a clip."
        )
        self._apply_active_montage()

    def _init_from_clip_folder(self):
        """Populate dropdown by scanning blinded clip EDFs in a folder."""
        folder = self.clips_folder_dir
        if not folder or not folder.is_dir():
            QMessageBox.critical(
                self, "Clip Folder Error", f"Clip folder not found: {folder}"
            )
            return

        shuffle = load_clip_shuffle_key(folder)

        def _review_order(path: Path) -> int:
            row = shuffle.get(path.stem)
            if row and str(row.get("review_order", "")).strip().isdigit():
                return int(row["review_order"])
            return 10**9

        edf_files = sorted(
            (p for p in folder.glob("*.edf") if p.is_file()),
            key=_review_order,
        )
        if not edf_files:
            QMessageBox.critical(
                self, "Clip Folder Error", f"No .edf files found in {folder}"
            )
            return

        clip_ids: list[str] = []
        selection_options: dict[str, dict] = {}
        parsed_spike_times: dict[str, float | None] = {}
        metadata_by_label: dict[str, dict] = {}

        for i, path in enumerate(edf_files):
            file_stem = path.stem
            row = shuffle.get(file_stem)
            ieeg_name = file_stem
            ts: float | None = None
            clip_type = ""
            original_name = path.name
            review_order = i + 1
            if row:
                ieeg_name = (row.get("ieeg_file_name") or file_stem).strip()
                ts_raw = (row.get("timestamp_sec") or "").strip()
                if ts_raw:
                    ts = float(ts_raw)
                clip_type = (row.get("clip_type") or row.get("selection_type") or "").strip()
                original_name = (row.get("original_name") or path.name).strip()
                ro_raw = str(row.get("review_order", "")).strip()
                if ro_raw.isdigit():
                    review_order = int(ro_raw)

            if ts is not None and ieeg_name:
                display_id = clip_display_stem(ieeg_name, ts)
                ann_stem = clip_annotation_stem(ieeg_name, ts)
                ann_relpath = f"{ann_stem}.json"
            else:
                display_id = file_stem
                ann_stem = file_stem
                ann_relpath = f"{file_stem}.json"

            opt = {
                "dataset_id": file_stem,
                "display_id": display_id,
                "spike_time_sec": ts,
                "annotation_stem": ann_stem,
                "annotation_relpath": ann_relpath,
                "recording_name": path.name,
                "original_name": original_name,
                "clip_type": clip_type,
                "patient_id": ieeg_name,
                "ieeg_file_name": ieeg_name,
                "timestamp_sec": ts,
                "review_order": review_order,
                "row_index": i,
            }
            clip_ids.append(display_id)
            selection_options[display_id] = opt
            parsed_spike_times[display_id] = ts
            metadata_by_label[display_id] = {
                "row_index": i,
                "patient_id": ieeg_name,
                "ieeg_file_name": ieeg_name,
                "timestamp_sec": ts,
                "review_order": review_order,
            }

        self.ieeg_selection_options = selection_options
        self.ieeg_spike_time_by_dataset = parsed_spike_times.copy()
        self.clip_metadata_by_label = metadata_by_label
        self.ieeg_patient_datasets = {
            clip_id: [selection_options[clip_id]["dataset_id"]] for clip_id in clip_ids
        }
        self.patients = clip_ids
        self.recordings = []
        self.current_patient = None
        self.current_recording = None

        self.patient_combo.blockSignals(True)
        self.patient_combo.clear()
        self.patient_combo.addItem("-- Select clip --")
        for clip_id in clip_ids:
            ro = metadata_by_label[clip_id].get("review_order", 0)
            self.patient_combo.addItem(f"{ro:03d}  {clip_id}", clip_id)
        self.patient_combo.setCurrentIndex(0)
        self.patient_combo.blockSignals(False)
        self._update_patient_dropdown_colors()

        self.recording_combo.blockSignals(True)
        self.recording_combo.clear()
        self.recording_combo.blockSignals(False)

        self._configure_clips_review_ui()
        self.status_bar.showMessage(
            f"Loaded {len(clip_ids)} clips from folder {folder}. "
            "10–20 map: click contacts to mark. Spike region 7–8 s (red band)."
        )
        self._apply_active_montage()

    def _configure_clips_review_ui(self) -> None:
        """Clips review: 10–20 contact marks (list only); spike band 7–8 s."""
        if not self.clips_mode:
            return
        if hasattr(self, "montage_1020_title_label"):
            self.montage_1020_title_label.show()
        if hasattr(self, "montage_1020_label"):
            self.montage_1020_label.show()
        if self._clips_review_ui_configured:
            return
        self._clips_review_ui_configured = True
        if hasattr(self, "clips_channel_help_label"):
            self.clips_channel_help_label.setText(
                "<i>Uncheck to hide traces. 10–20 map: click contacts to mark "
                "(saved in list only). Spike window 7–8 s shown in red.</i>"
            )
        if self.ieeg_mode:
            self.status_bar.showMessage(
                "10–20 map: click contacts to mark (auto-saved). "
                "Spike region 7–8 s.  ↑↓ amplitude.  ←→ scroll time."
            )

    def _update_clip_info_label(self, label: str | None):
        """Clips mode uses dropdown text only (number | patient | timestamp)."""
        del label

    # ── Data discovery ─────────────────────────────────────────────────────────

    def _browse_data_dir(self):
        from PyQt5.QtWidgets import QFileDialog
        start = str(self.data_dir) if self.data_dir and self.data_dir.exists() else ""
        d = QFileDialog.getExistingDirectory(self, "Select data directory", start)
        if d:
            self.data_dir = Path(d)
            self._discover_patients()

    def _discover_patients(self):
        """Populate patient dropdown from sub-RID* folders."""
        if not self.data_dir or not self.data_dir.exists():
            return
        self.patients = sorted(
            p.name for p in self.data_dir.iterdir()
            if p.is_dir() and p.name.startswith("sub-")
        )
        self.patient_combo.blockSignals(True)
        self.patient_combo.clear()
        self.patient_combo.addItems(self.patients)
        self.patient_combo.blockSignals(False)
        if self.patients:
            self.patient_combo.setCurrentIndex(0)
            self._on_patient_changed(self.patients[0])

    def _on_next_patient(self):
        cb = self.patient_combo
        n = cb.count()
        if n < 2:
            return
        start = 1 if cb.itemText(0).startswith("-- Select") else 0
        i = cb.currentIndex()
        if i < start:
            i = start - 1
        new_i = i + 1
        if new_i >= n:
            new_i = start
        cb.setCurrentIndex(new_i)

    def _on_prev_patient(self):
        cb = self.patient_combo
        n = cb.count()
        if n < 2:
            return
        start = 1 if cb.itemText(0).startswith("-- Select") else 0
        i = cb.currentIndex()
        if i < start:
            i = start
        new_i = i - 1
        if new_i < start:
            new_i = n - 1
        cb.setCurrentIndex(new_i)

    def _selected_clip_id(self) -> str | None:
        """Unique clip key (annotation stem); display text may repeat ieeg_file_name."""
        idx = self.patient_combo.currentIndex()
        if idx < 0:
            return None
        data = self.patient_combo.itemData(idx)
        if data is not None and str(data).strip():
            return str(data)
        return None

    def _reload_current_ieeg_selection(self):
        """Re-download the current dropdown option (e.g. after toggling no-spike view mode)."""
        if not self.ieeg_mode or not self.ieeg_edf_options_csv:
            return
        patient = (
            self._selected_clip_id()
            if self.clips_csv
            else self.patient_combo.currentText()
        )
        placeholder = "-- Select clip --" if self.clips_csv else "-- Select Patient --"
        if not patient or patient == placeholder:
            return
        option = self.ieeg_selection_options.get(patient)
        if option is None:
            return
        self.current_patient = patient
        self.patient_id_for_annotations = str(option.get("patient_id", patient))
        self._pending_recording_name = option.get("recording_name")
        self._update_clip_info_label(patient)
        if getattr(self, "btn_ieeg_no_spike", None) and self.btn_ieeg_no_spike.isChecked():
            self._pending_spike_time_sec = None
        else:
            self._pending_spike_time_sec = option.get("spike_time_sec")
        self._load_current_clip(option)

    def _apply_clip_type_defaults_after_load(self) -> None:
        """When no saved review file exists, sync UI from CSV clip_type."""
        if not self.clips_csv or not self.current_patient:
            return
        path = self._annotation_path()
        if path is not None and path.exists():
            return
        opt = self.ieeg_selection_options.get(self.current_patient)
        if not opt:
            return
        if opt.get("clip_type") == "no_spike" and not self._has_no_spike_annotation():
            self.channel_annotations.append({
                "channel": "(clip)",
                "label": "no_spike",
            })
            self._refresh_annotation_list()

    def _has_no_spike_annotation(self) -> bool:
        for ann in self.channel_annotations:
            label = (ann.get("label") or "").strip().lower()
            if label == "no_spike":
                return True
        return False

    def _toggle_no_spike_annotation(self) -> None:
        if self._has_no_spike_annotation():
            self.channel_annotations = [
                ann for ann in self.channel_annotations
                if (ann.get("label") or "").strip().lower() != "no_spike"
            ]
            self.status_bar.showMessage("Removed no spike mark.")
        else:
            self.channel_annotations.append({
                "channel": "(clip)",
                "label": "no_spike",
            })
            self.status_bar.showMessage("Marked no spike.")
        self._refresh_annotation_list()
        self._save_annotations()

    def _on_no_spike_clicked(self) -> None:
        if not self.clips_mode or self.eeg_data is None:
            self.status_bar.showMessage("Load a clip first.")
            return
        self._toggle_no_spike_annotation()

    def _on_ieeg_no_spike_toggled(self, checked: bool):
        if not self.ieeg_mode or not self.ieeg_edf_options_csv:
            return
        self.global_types["custom_type"] = "no_spike" if checked else ""
        self._reset_global_widgets()
        self.global_custom_type_edit.setText(self.global_types.get("custom_type", ""))
        self._reload_current_ieeg_selection()

    def _load_current_clip(self, option: dict):
        """Load pre-exported EDF if available, otherwise stream from iEEG.org."""
        rec = option.get("recording_name", "")
        if self.clip_edf_dir and rec:
            edf_path = self.clip_edf_dir / rec
            if edf_path.exists():
                self._load_local_clip_edf(edf_path, option)
                return
        if not self.ieeg_username or not self.ieeg_password:
            self.status_bar.showMessage(
                f"Missing local EDF for clip: {rec}. "
                f"No iEEG credentials available to stream as fallback."
            )
            return
        self._load_ieeg_recording(option["dataset_id"])

    def _on_patient_changed(self, patient: str):
        if self.ieeg_mode:
            if not patient:
                return
            placeholder = "-- Select clip --" if self.clips_mode else "-- Select Patient --"
            if patient == placeholder:
                self.current_patient = None
                self.current_recording = None
                self._clear_recording_state()
                self._update_clip_info_label(None)
                self.status_bar.showMessage(
                    "Select a clip to load." if self.clips_mode
                    else "Select a patient to load iEEG data."
                )
                return
            if self.clips_mode:
                clip_id = self._selected_clip_id()
                if not clip_id:
                    return
                patient = clip_id
            self.current_patient = patient

            # selections.csv / clips.csv / folder scan: one option per dropdown row.
            if self.ieeg_edf_options_csv or self.clips_mode:
                option = self.ieeg_selection_options.get(patient)
                if option is None:
                    self.status_bar.showMessage(f"No option mapping found for {patient}.")
                    return
                self.patient_id_for_annotations = str(
                    option.get("patient_id", patient)
                )
                self._update_clip_info_label(patient)
                if getattr(self, "btn_ieeg_no_spike", None) and self.btn_ieeg_no_spike.isChecked():
                    self._pending_spike_time_sec = None
                else:
                    self._pending_spike_time_sec = option.get("spike_time_sec")
                self._pending_recording_name = option.get("recording_name")
                self._load_current_clip(option)
                return

            self.patient_id_for_annotations = patient

            # In CSV-driven iEEG mode, switching patient swaps the dataset list.
            if self.ieeg_patient_datasets:
                datasets = self.ieeg_patient_datasets.get(patient, [])
                self.recording_combo.blockSignals(True)
                self.recording_combo.clear()
                self.recording_combo.addItems(datasets)
                self.recording_combo.blockSignals(False)

                if datasets:
                    self.current_recording = datasets[0]
                    self._load_ieeg_recording(datasets[0])
                else:
                    self._clear_recording_state()

            # If not using the mapping CSV, patient switching is cosmetic
            # (single dataset mode), so nothing else is required.
            return
        if not patient:
            return
        self.current_patient = patient
        patient_path = self.data_dir / patient

        # Find session dir – prefer ses-* directories, skip derivatives etc.
        session_dirs = sorted(
            d for d in patient_path.iterdir()
            if d.is_dir() and d.name.startswith("ses-")
        )
        if not session_dirs:
            # Fallback: accept any subdirectory that is not 'derivatives'
            session_dirs = sorted(
                d for d in patient_path.iterdir()
                if d.is_dir() and d.name.lower() != "derivatives"
            )
        if not session_dirs:
            return
        self.patient_dir = session_dirs[0]   # take first session

        # ── Discover recordings ──────────────────────────────────────────
        # Strategy 1: use scans.tsv if available
        scans_files = list(self.patient_dir.glob("*_scans.tsv"))
        ictal_files: list[str] = []
        if scans_files:
            scans_df = pd.read_csv(scans_files[0], sep="\t")
            if "filename" in scans_df.columns:
                ictal_files = [
                    f for f in scans_df["filename"].tolist()
                    if "ictal" in Path(f).name
                    and "interictal" not in Path(f).name
                    and f.endswith(".edf")
                ]

        # Strategy 2: fallback – list all EDF files in ieeg/ directly
        if not ictal_files:
            ieeg_dir = self.patient_dir / "ieeg"
            if ieeg_dir.is_dir():
                all_edfs = sorted(ieeg_dir.glob("*.edf"))
                # Prefer ictal EDFs if any, otherwise show all
                ictal_edfs = [
                    f for f in all_edfs
                    if "ictal" in f.name and "interictal" not in f.name
                ]
                chosen = ictal_edfs if ictal_edfs else all_edfs
                ictal_files = [str(f) for f in chosen]

        self.recordings = [Path(f).stem for f in ictal_files]
        self.recording_combo.blockSignals(True)
        self.recording_combo.clear()
        self.recording_combo.addItems(self.recordings)
        self.recording_combo.blockSignals(False)

        if self.recordings:
            self.recording_combo.setCurrentIndex(0)
            self._on_recording_changed(self.recordings[0])
        else:
            # No recordings found — clear stale state from previous patient
            self._clear_recording_state()

    def _clear_recording_state(self):
        """Reset all recording-related state and clear the plot."""
        self.current_recording = None
        self.eeg_data = None
        self.channel_names_all = []
        self.displayed_channels = []
        self.total_duration = 0.0
        self.time_offset_sec = 0.0
        self.clip_spike_abs_sec = None
        self.clip_spike_rel_sec = None
        self.ictal_onset = None
        self.ictal_duration = None
        self._ref_eeg_data_base = None
        self._ref_eeg_data = None
        self._ref_channel_names = []
        self.eeg_data_bipolar = None
        self.channel_names_bipolar = []
        self.eeg_data_banana = None
        self.channel_names_banana = []
        self.banana_midline_pairs = set()
        self.eeg_data_ap_bipolar = None
        self.channel_names_ap_bipolar = []
        self.ap_bipolar_midline_pairs = set()
        self.global_types = {
            "lvfa": False, "frequency": "(none)",
            "pattern": "(none)", "modifier": "None", "custom_type": "",
        }
        self.channel_annotations = []
        self._reset_global_widgets()
        self._refresh_annotation_list()
        if hasattr(self, "montage_1020_label"):
            self.montage_1020_label.set_highlight_contacts(set())
        self.ch_list.clear()
        self.plot.clear()
        self.trace_items = []
        self.annotation_markers = []
        self.ictal_region = None
        self.status_bar.showMessage(
            f"No recordings found for {self.current_patient}."
        )

    def _on_recording_changed(self, recording_stem: str):
        if not recording_stem:
            return

        self.current_recording = recording_stem

        if self.ieeg_mode:
            self._load_ieeg_recording(recording_stem)
            return

        if not self.patient_dir:
            return

        self._load_recording(recording_stem)

    # ── Recording loader ───────────────────────────────────────────────────────

    def _load_ieeg_recording(self, dataset_id: str):
        """Load iEEG.org data directly (no local EDF/BIDS folder needed).

        This implementation downloads the whole dataset into memory so the
        existing plotting + annotation logic can stay unchanged.
        """
        selected_recording_name = getattr(self, "_pending_recording_name", None)
        selected_spike_time = getattr(self, "_pending_spike_time_sec", None)
        self._pending_recording_name = None
        self._pending_spike_time_sec = None

        self._clear_recording_state()
        if not dataset_id:
            return

        if not self.ieeg_username or not self.ieeg_password:
            QMessageBox.critical(
                self,
                "iEEG Connection Error",
                "Missing iEEG credentials. Provide --ieeg_username and set the password via --ieeg_password_env (or --ieeg_password).",
            )
            self.status_bar.showMessage("iEEG load failed.")
            return

        self.current_recording = selected_recording_name or dataset_id
        # In iEEG "spike window" mode we do not show ictal guide lines.
        self.ictal_onset = None
        self.ictal_duration = None

        self.status_bar.showMessage(f"Loading iEEG dataset {dataset_id} …")
        QApplication.setOverrideCursor(Qt.WaitCursor)
        QApplication.processEvents()

        try:
            from ieeg_load_preprocess import CHANNELS_TO_INCLUDE, load_and_preprocess_window

            use_flat = (
                getattr(self, "btn_ieeg_no_spike", None) is not None
                and self.btn_ieeg_no_spike.isChecked()
            )
            spike_time = None if use_flat else selected_spike_time
            if spike_time is None and not use_flat and self.current_patient:
                spike_time = self.ieeg_spike_time_by_dataset.get(self.current_patient)
            if spike_time is None and not use_flat:
                spike_time = self.ieeg_spike_time_by_dataset.get(dataset_id)
            half = float(IEEG_SPIKE_HALF_WINDOW_SEC)
            if spike_time is not None and not use_flat:
                trigger_sec = float(spike_time)
                spike_center_sec = trigger_sec + SPIKENET_TRIGGER_TO_CENTER_SEC
                # Match clip export: ±7 s around spike center
                # ([trigger - 6.5, trigger + 7.5] in recording time).
                window_start_sec = max(0.0, spike_center_sec - half)
                window_end_sec = spike_center_sec + half
            else:
                window_start_sec = 0.0
                window_end_sec = 2.0 * half
                spike_center_sec = None

            window_duration_sec = max(0.0, window_end_sec - window_start_sec)
            self.total_duration = window_duration_sec
            self.time_offset_sec = window_start_sec
            if spike_center_sec is not None:
                self.clip_spike_abs_sec = spike_center_sec
                self.clip_spike_rel_sec = spike_center_sec - window_start_sec
            else:
                self.clip_spike_abs_sec = None
                self.clip_spike_rel_sec = None
            if window_duration_sec <= 0:
                raise ValueError(
                    f"Invalid data window for dataset {dataset_id}. "
                    f"Spike time may be outside recording bounds."
                )

            ref_base, channel_names, self.fs = load_and_preprocess_window(
                self.ieeg_username,
                self.ieeg_password,
                dataset_id,
                int(window_start_sec * 1e6),
                int(window_end_sec * 1e6),
                CHANNELS_TO_INCLUDE,
            )
            self._commit_referential_data(ref_base, channel_names)

        except Exception as exc:
            QMessageBox.critical(self, "iEEG Load Error", str(exc))
            self.status_bar.showMessage("Load failed.")
            QApplication.restoreOverrideCursor()
            return
        finally:
            if QApplication.overrideCursor() is not None:
                QApplication.restoreOverrideCursor()

        if self._ref_eeg_data_base is None:
            return

        # Reset scroll + gain now that data exists.
        self._finish_clip_load_status(
            f"Loaded iEEG {dataset_id}  |  {len(self.channel_names_all)} ch  |  "
            f"{self.total_duration:.1f}s window  |  fs={self.fs:.0f}Hz"
        )

    def _load_local_clip_edf(self, edf_path: Path, option: dict | None = None):
        """Load a pre-exported ±7 s clip EDF (full file = display window)."""
        self._clear_recording_state()
        if not edf_path.exists():
            self.status_bar.showMessage(f"EDF not found: {edf_path}")
            return

        self.status_bar.showMessage(f"Loading {edf_path.name} …")
        QApplication.processEvents()
        try:
            raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
            self.fs = float(raw.info["sfreq"])
            self.channel_names_all = list(raw.ch_names)
            raw_uv = raw.get_data().T.astype(np.float64)
            data_v = self._preprocess(raw_uv, self.fs).astype(np.float32)
            self.total_duration = float(data_v.shape[0] / self.fs)
            self.time_offset_sec = 0.0
            self.ictal_onset = None
            self.ictal_duration = None
            self.current_recording = (
                (option or {}).get("display_id")
                or edf_path.stem
            )

            self._commit_referential_data(data_v, self.channel_names_all)
            # Preprocessed clips: +0.5 s export shift places spike center at 7.5 s (7–8 s band).
            self.clip_spike_rel_sec = CLIP_SPIKE_CENTER_REL_SEC
            spike_sec = option.get("spike_time_sec") if option is not None else None
            if spike_sec is not None:
                self.clip_spike_abs_sec = (
                    float(spike_sec) + SPIKENET_TRIGGER_TO_CENTER_SEC
                )
            else:
                self.clip_spike_abs_sec = (
                    self.time_offset_sec + self.clip_spike_rel_sec
                )
        except Exception as exc:
            QMessageBox.critical(self, "EDF Load Error", str(exc))
            self.status_bar.showMessage("Load failed.")
            return

        if self._ref_eeg_data_base is None:
            return

        self._finish_clip_load_status(
            f"Loaded {edf_path.name}  |  {len(self.channel_names_all)} ch  |  "
            f"{self.total_duration:.1f}s  |  fs={self.fs:.0f}Hz"
        )

    def _configure_time_scroll(self) -> None:
        """3 s visible window; scroll in TIME_SCROLL_STEP_SEC steps across the clip."""
        max_scroll = max(0.0, self.total_duration - VISIBLE_SECONDS)
        if self.clip_spike_rel_sec is not None:
            initial_scroll = self.clip_spike_rel_sec - (VISIBLE_SECONDS / 2.0)
            initial_scroll = max(0.0, min(max_scroll, initial_scroll))
        elif self.ictal_onset is not None:
            initial_scroll = max(0.0, min(self.ictal_onset - 5.0, max_scroll))
        else:
            initial_scroll = 0.0
        self.scroll_pos = initial_scroll
        step_ticks = max(1, int(round(TIME_SCROLL_STEP_SEC * SCROLL_TICKS_PER_SEC)))
        max_ticks = int(round(max_scroll * SCROLL_TICKS_PER_SEC))
        self.scroll_bar.blockSignals(True)
        self.scroll_bar.setMinimum(0)
        self.scroll_bar.setMaximum(max_ticks)
        self.scroll_bar.setSingleStep(step_ticks)
        self.scroll_bar.setPageStep(step_ticks)
        self.scroll_bar.setValue(int(round(initial_scroll * SCROLL_TICKS_PER_SEC)))
        self.scroll_bar.blockSignals(False)

    def _nudge_time_scroll(self, direction: int) -> None:
        step_ticks = max(1, int(round(TIME_SCROLL_STEP_SEC * SCROLL_TICKS_PER_SEC)))
        new_val = max(
            self.scroll_bar.minimum(),
            min(self.scroll_bar.maximum(), self.scroll_bar.value() + direction * step_ticks),
        )
        self.scroll_bar.setValue(new_val)

    def _finish_clip_load_status(self, message: str) -> None:
        """After a clip loads: auto-gain, sync slider, reset scroll/annotations."""
        self._set_gain(self._auto_gain())
        self.global_types = {
            "lvfa": False, "frequency": "(none)",
            "pattern": "(none)", "modifier": "None", "custom_type": "",
        }
        self.channel_annotations = []
        self._reset_global_widgets()
        self._refresh_annotation_list()
        self._configure_time_scroll()

        self._populate_channel_list()
        self._load_annotations()
        self._apply_clip_type_defaults_after_load()
        self._load_soz_channels()
        self._rebuild_displayed_channels()
        ann_hint = ""
        ann_path = self._annotation_path()
        if ann_path is not None:
            ann_hint = f"  Annotations: {ann_path.resolve()}"
        self.status_bar.showMessage(message + ann_hint)

    def _load_recording(self, stem: str):
        """Load EDF + metadata for a given recording stem."""
        ieeg_dir = self.patient_dir / "ieeg"
        edf_path = ieeg_dir / (stem + ".edf")

        if not edf_path.exists():
            self.status_bar.showMessage(f"EDF not found: {edf_path}")
            return

        self.status_bar.showMessage(f"Loading {edf_path.name} …")
        QApplication.processEvents()

        try:
            # ── metadata ─────────────────────────────────────────────────────
            json_path = ieeg_dir / (stem + ".json")
            if json_path.exists():
                with open(json_path) as f:
                    meta = json.load(f)
                self.fs = float(meta.get("SamplingFrequency", 1000.0))
                self.total_duration = float(meta.get("RecordingDuration", 0.0))
            else:
                self.fs = 1000.0
                self.total_duration = 0.0

            events_path = ieeg_dir / (stem + "_events.tsv")
            if events_path.exists():
                ev_df = pd.read_csv(events_path, sep="\t", skip_blank_lines=True)
                ev_df = ev_df.dropna(subset=["trial_type"])
                ictal_rows = ev_df[ev_df["trial_type"].str.strip().str.lower() == "ictal"]
                if not ictal_rows.empty:
                    self.ictal_onset = float(ictal_rows.iloc[0]["onset"])
                    self.ictal_duration = float(ictal_rows.iloc[0]["duration"])
                    print(f"  Ictal onset: {self.ictal_onset}s  duration: {self.ictal_duration}s")
                else:
                    self.ictal_onset = None
                    self.ictal_duration = None
                    print("  No ictal row found in events.tsv")
            else:
                self.ictal_onset = None
                self.ictal_duration = None

            channels_path = ieeg_dir / (stem + "_channels.tsv")
            good_seeg_ecog: list[str] = []
            if channels_path.exists():
                ch_df = pd.read_csv(channels_path, sep="\t")
                mask = (
                    ch_df["type"].str.upper().isin(["SEEG", "ECOG"])
                    & (ch_df.get("status", pd.Series(["good"] * len(ch_df))).str.lower() != "bad")
                )
                good_seeg_ecog = ch_df.loc[mask, "name"].tolist()

            # ── EDF ──────────────────────────────────────────────────────────
            raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
            raw_ch_names = raw.ch_names

            # Intersect with good SEEG/ECOG channels (preserving EDF order)
            if good_seeg_ecog:
                keep = [c for c in raw_ch_names if c in good_seeg_ecog]
                if not keep:
                    # try normalised matching
                    keep = [
                        c for c in raw_ch_names
                        if any(
                            normalize_channel_name(c.upper()) == normalize_channel_name(g.upper())
                            for g in good_seeg_ecog
                        )
                    ]
            else:
                keep = raw_ch_names

            if keep:
                raw.pick(keep)

            data = raw.get_data(return_times=False, units="V")
            # data shape: (n_channels, n_samples) → transpose → (n_samples, n_channels)
            raw_uv = (data.T * IEEG_SCALE).astype(np.float64)
            self.eeg_data = self._preprocess(raw_uv, self.fs).astype(np.float32)
            self.channel_names_all = list(raw.ch_names)
            self.total_duration = self.eeg_data.shape[0] / self.fs
            self.time_offset_sec = 0.0

        except Exception as exc:
            QMessageBox.critical(self, "Load Error", str(exc))
            self.status_bar.showMessage("Load failed.")
            return

        self._ref_eeg_data_base = self.eeg_data.astype(np.float32)
        self._ref_channel_names = list(self.channel_names_all)
        self._rebuild_montage_caches()
        self._apply_active_montage()

        # ── reset state ───────────────────────────────────────────────────────
        self.global_types = {
            "lvfa": False, "frequency": "(none)",
            "pattern": "(none)", "modifier": "None", "custom_type": "",
        }
        self.channel_annotations = []
        self._reset_global_widgets()
        self._refresh_annotation_list()
        self._set_gain(self._auto_gain())
        self._configure_time_scroll()

        # Populate channel list in right panel
        self._populate_channel_list()

        # Load any existing annotations (will re-populate the list if file exists)
        self._load_annotations()

        # Load SOZ channels (override CSV → hardcoded fallback)
        self._load_soz_channels()

        # Build channel subset and draw
        self._rebuild_displayed_channels()
        self.status_bar.showMessage(
            f"Loaded {stem}  |  {len(self.channel_names_all)} ch  |  "
            f"{self.total_duration:.1f}s  |  fs={self.fs:.0f}Hz"
        )

    # ── Preprocessing ─────────────────────────────────────────────────────────

    @staticmethod
    def _preprocess(data: np.ndarray, fs: float) -> np.ndarray:
        """
        Minimal preprocessing to make raw iEEG displayable without destroying
        seizure-relevant content.

        Steps applied per channel:
          1. 1 Hz 4th-order Butterworth high-pass  →  removes DC offset and
             slow electrode drift that shift each channel's baseline arbitrarily
             far from zero, which is the main reason raw traces look like
             vertical spikes with no visible waveform.
          2. 60 Hz IIR notch (Q=30)  →  removes power-line interference that
             adds large-amplitude sinusoidal noise common in clinical recordings.

        A hard bandpass is intentionally NOT applied so that high-frequency
        seizure patterns (e.g. LVFA at 80-200 Hz) remain visible.
        """
        out = data.copy()

        nyq = fs / 2.0
        hp_sos = butter(4, 1.0 / nyq, btype="high", output="sos")

        notch_sos = None
        if nyq > 61:
            b_n, a_n = iirnotch(60.0 / nyq, Q=30)
            from scipy.signal import tf2sos
            notch_sos = tf2sos(b_n, a_n)

        for ch in range(out.shape[1]):
            sig = out[:, ch]
            sig = sosfiltfilt(hp_sos, sig)
            if notch_sos is not None:
                sig = sosfiltfilt(notch_sos, sig)
            out[:, ch] = sig

        return out

    # ── Common average montage ─────────────────────────────────────────────────

    @staticmethod
    def _compute_car_montage(
        data: np.ndarray, channel_names: list[str]
    ) -> tuple[np.ndarray, list[str]]:
        """Common average reference: each channel minus the per-sample mean
        across all channels. Channel set/order is unchanged."""
        if data.size == 0 or data.shape[1] == 0:
            return data, list(channel_names)
        avg_signal = data.mean(axis=1)
        result = data - avg_signal[:, np.newaxis]
        if result.shape != data.shape:
            raise ValueError("The shape of the resulting data doesn't match the input data.")
        return result.astype(data.dtype), list(channel_names)

    # ── Bipolar montage ───────────────────────────────────────────────────────

    @staticmethod
    def _compute_bipolar_montage(
        data: np.ndarray, channel_names: list[str]
    ) -> tuple[np.ndarray, list[str]]:
        """Compute an adjacent bipolar montage from referential data.

        For every channel whose immediate successor on the same electrode shaft
        exists in the recording, produces ch[n] - ch[n+1].  Channels with no
        successor are silently dropped.  Electrode shafts are identified by the
        leading letter(s) in the channel name (e.g. "LA" in "LA01"); pairing is
        by consecutive integer suffixes (01→02, 02→03, …).

        Parameters
        ----------
        data:
            (n_samples, n_channels) µV array in referential montage.
        channel_names:
            Ordered list of channel names matching the columns of *data*.

        Returns
        -------
        bipolar_data:
            (n_samples, n_bipolar_channels) array.
        bipolar_names:
            List of names like "LA01-LA02" in the same electrode order.
        """
        name_to_idx: dict[str, int] = {n: i for i, n in enumerate(channel_names)}

        bipolar_cols: list[np.ndarray] = []
        bipolar_names: list[str] = []

        for i, ch_name in enumerate(channel_names):
            m = re.match(r'^([A-Za-z]+)(\d+)$', ch_name.strip())
            if not m:
                continue
            prefix = m.group(1)
            num = int(m.group(2))
            pad = len(m.group(2))   # number of digits (for zero-padded names)

            # Look for the next channel: try zero-padded then bare
            next_num = num + 1
            candidates = [f"{prefix}{next_num:0{pad}d}", f"{prefix}{next_num}"]
            next_ch = next((c for c in candidates if c in name_to_idx), None)
            if next_ch is None:
                continue

            # Verify same prefix (guard against cross-shaft pairing)
            m2 = re.match(r'^([A-Za-z]+)(\d+)$', next_ch.strip())
            if not m2 or m2.group(1) != prefix:
                continue

            bipolar_cols.append(data[:, i] - data[:, name_to_idx[next_ch]])
            bipolar_names.append(f"{ch_name}-{next_ch}")

        if not bipolar_cols:
            return data, channel_names

        return np.column_stack(bipolar_cols).astype(data.dtype), bipolar_names

    @staticmethod
    def _compute_banana_montage(
        data: np.ndarray, channel_names: list[str]
    ) -> tuple[np.ndarray, list[str], set[str]]:
        """Longitudinal bipolar: subtemporal → temporal → parasagittal → central."""
        return _compute_chain_groups_montage(
            data, channel_names, BANANA_CHAIN_GROUPS, include_spacers=False
        )

    @staticmethod
    def _compute_ap_bipolar_montage(
        data: np.ndarray, channel_names: list[str]
    ) -> tuple[np.ndarray, list[str], set[str]]:
        """AP bipolar: same chains as banana; order L temp → L central → R temp → R central → midline."""
        return _compute_chain_groups_montage(data, channel_names, AP_BIPOLAR_CHAIN_GROUPS)

    # ── Auto-gain ─────────────────────────────────────────────────────────────

    def _auto_gain(self) -> float:
        """
        Compute an initial gain so the 90th-percentile signal amplitude fills
        ~40 % of the fixed channel spacing, giving clearly readable traces
        that don't bleed into adjacent channels.
        """
        if self.eeg_data is None or self.eeg_data.shape[0] == 0:
            return 1.0

        # Prefer the ictal window; fall back to the first 30 s
        if self.ictal_onset is not None:
            s0 = max(0, int(self.ictal_onset * self.fs))
            s1 = min(self.eeg_data.shape[0],
                     int((self.ictal_onset + min(self.ictal_duration or 30, 30)) * self.fs))
        else:
            s0 = 0
            s1 = min(self.eeg_data.shape[0], int(30 * self.fs))

        sample = self.eeg_data[s0:s1, :]
        p90 = float(np.percentile(np.abs(sample), 90))
        if p90 == 0:
            return 1.0

        # Target: 90th-percentile amplitude = 40 % of spacing
        target_uv = CHANNEL_SPACING_UV * 0.40
        return target_uv / p90

    # ── Montage caches ────────────────────────────────────────────────────────

    def _rebuild_montage_caches(self) -> None:
        if self._ref_eeg_data_base is None:
            return
        ref = self._ref_eeg_data_base.astype(np.float32)
        self._ref_eeg_data = ref
        names = self._ref_channel_names
        self.eeg_data_car, self.channel_names_car = self._compute_car_montage(ref, names)
        self.eeg_data_bipolar, self.channel_names_bipolar = self._compute_bipolar_montage(
            ref, names
        )
        self.eeg_data_banana, self.channel_names_banana, self.banana_midline_pairs = (
            self._compute_banana_montage(ref, names)
        )
        self.eeg_data_ap_bipolar, self.channel_names_ap_bipolar, self.ap_bipolar_midline_pairs = (
            self._compute_ap_bipolar_montage(ref, names)
        )

    def _selected_montage_mode(self) -> str:
        if hasattr(self, "montage_combo"):
            mode = self.montage_combo.currentData()
            if mode:
                return str(mode)
        return self.montage

    def _apply_active_montage(self) -> None:
        if self._ref_eeg_data_base is None:
            return
        mode = self._selected_montage_mode()
        if mode == "car":
            self.montage = "car"
            if self.eeg_data_car is not None:
                self.eeg_data = self.eeg_data_car
                self.channel_names_all = list(self.channel_names_car)
            else:
                self.eeg_data = self._ref_eeg_data
                self.channel_names_all = list(self._ref_channel_names)
        else:
            self.montage = "banana"
            if self.eeg_data_banana is not None:
                self.eeg_data = self.eeg_data_banana
                self.channel_names_all = list(self.channel_names_banana)
            else:
                self.eeg_data = self._ref_eeg_data
                self.channel_names_all = list(self._ref_channel_names)

    def _commit_referential_data(
        self,
        ref_base: np.ndarray,
        channel_names: list[str],
    ) -> None:
        self._ref_eeg_data_base = ref_base.astype(np.float32)
        self._ref_channel_names = list(channel_names)
        self._rebuild_montage_caches()
        self._apply_active_montage()

    def _on_montage_combo_changed(self, _index: int) -> None:
        self._set_montage(self._selected_montage_mode())

    def _set_view(self, mode: str):
        self.view_mode = "all"
        self.v_scroll_bar.hide()
        self._rebuild_displayed_channels()

    def _set_montage(self, mode: str | None = None):
        if self._ref_eeg_data_base is None:
            return
        if mode is not None and hasattr(self, "montage_combo"):
            idx = self.montage_combo.findData(mode)
            if idx >= 0 and self.montage_combo.currentIndex() != idx:
                self.montage_combo.blockSignals(True)
                self.montage_combo.setCurrentIndex(idx)
                self.montage_combo.blockSignals(False)
        self._apply_active_montage()
        self._populate_channel_list()
        self._rebuild_displayed_channels()
        self._sync_channel_list_annotation_style()
        self._update_montage_1020_highlights()

    def _rebuild_displayed_channels(self):
        """All visible channels; top of screen = first chain (e.g. Fp1–F7)."""
        if not self.channel_names_all:
            self.displayed_channels = []
            self._sync_channel_list_annotation_style()
            self._draw_traces()
            return

        def real_channel_visible(name: str) -> bool:
            if name.startswith("__SPACER_"):
                return False
            return name in self._visible_channel_names

        out: list[str] = []
        prev_real_shown = False
        i = 0
        n = len(self.channel_names_all)
        while i < n:
            ch = self.channel_names_all[i]
            if ch.startswith("__SPACER_"):
                future_shown = False
                for k in range(i + 1, n):
                    c2 = self.channel_names_all[k]
                    if c2.startswith("__SPACER_"):
                        continue
                    future_shown = real_channel_visible(c2)
                    break
                if prev_real_shown and future_shown:
                    out.append(ch)
                i += 1
                continue
            if real_channel_visible(ch):
                out.append(ch)
                prev_real_shown = True
            i += 1

        if self.montage == "banana":
            # Frontal chains (e.g. Fp1–F7) at top of the plot.
            self.displayed_channels = list(reversed(out))
        else:
            self.displayed_channels = out
        if not any(not c.startswith("__SPACER_") for c in self.displayed_channels):
            self.status_bar.showMessage("No channels visible — check channels in the list.")

        self._sync_channel_list_annotation_style()
        self._draw_traces()

    def _populate_channel_list(self):
        """Fill the right-panel channel list (top-to-bottom matches plot order)."""
        self.ch_list.blockSignals(True)
        self.ch_list.clear()
        self._visible_channel_names.clear()
        real_chs = [
            ch for ch in self.channel_names_all if not ch.startswith("__SPACER_")
        ]
        for ch in reversed(real_chs):
            item = QListWidgetItem(ch)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            item.setCheckState(Qt.Checked)
            self._visible_channel_names.add(ch)
            self.ch_list.addItem(item)
        self.ch_list.blockSignals(False)
        self._sync_channel_list_annotation_style()

    def _filter_channel_list(self, text: str):
        for i in range(self.ch_list.count()):
            item = self.ch_list.item(i)
            item.setHidden(text.lower() not in item.text().lower())

    def _on_channel_list_item_changed(self, item: QListWidgetItem):
        """Checkbox toggles visibility in the montage (not annotation)."""
        if self.ch_list.signalsBlocked():
            return
        name = item.text()
        if item.checkState() == Qt.Checked:
            self._visible_channel_names.add(name)
        else:
            self._visible_channel_names.discard(name)
        self._rebuild_displayed_channels()

    def _contacts_from_row_name(self, row_name: str) -> list[str]:
        """Split a bipolar trace label (Fp1-F7) into individual 10–20 contacts."""
        if not row_name or row_name.startswith("__SPACER_"):
            return []
        if "-" in row_name:
            left, _, right = row_name.partition("-")
            out = []
            if left.strip():
                out.append(left.strip())
            if right.strip():
                out.append(right.strip())
            return out
        return [row_name.strip()]

    def _is_contact_marked(self, contact: str) -> bool:
        for ann in self.channel_annotations:
            ch = ann.get("channel") or ""
            if self._ten20_contact_equivalent(str(ch), contact):
                return True
        return False

    def _remove_contact_annotation(self, contact: str) -> None:
        self.channel_annotations = [
            ann for ann in self.channel_annotations
            if not self._ten20_contact_equivalent(ann.get("channel") or "", contact)
        ]

    def _add_contact_annotation(self, contact: str) -> None:
        if self._is_contact_marked(contact):
            return
        self.channel_annotations.append({
            "channel": contact.strip(),
            "label": "marked",
        })

    def _toggle_contact_annotation(self, contact: str) -> None:
        """Toggle one scalp contact on the 10–20 map (add/remove individually)."""
        if self.eeg_data is None:
            self.status_bar.showMessage("Load a recording first.")
            return
        contact = contact.strip()
        if self._is_contact_marked(contact):
            self._remove_contact_annotation(contact)
            self.status_bar.showMessage(f"Removed {contact}.")
        else:
            self._add_contact_annotation(contact)
            self.status_bar.showMessage(f"Marked {contact}.")
        self._refresh_annotation_list()
        if self.clips_mode:
            self._update_montage_1020_highlights()
        else:
            self._redraw_annotation_markers()
            self._sync_channel_list_annotation_style()
        self._save_annotations()

    def _is_bipolar_marked(self, pair_name: str) -> bool:
        pair = pair_name.strip()
        return any((ann.get("channel") or "").strip() == pair for ann in self.channel_annotations)

    def _remove_bipolar_annotation(self, pair_name: str) -> None:
        pair = pair_name.strip()
        self.channel_annotations = [
            ann for ann in self.channel_annotations
            if (ann.get("channel") or "").strip() != pair
        ]

    def _add_bipolar_annotation(self, pair_name: str) -> None:
        pair = pair_name.strip()
        if not pair or self._is_bipolar_marked(pair):
            return
        clip_start = float(self.time_offset_sec)
        clip_end = float(self.time_offset_sec + self.total_duration)
        self.channel_annotations.append({
            "channel": pair,
            "label": "marked",
            "time_sec": clip_start,
            "end_sec": clip_end,
        })

    def _toggle_bipolar_annotation(self, pair_name: str) -> None:
        """Toggle one longitudinal bipolar row (e.g. Fp1-F3)."""
        if self.eeg_data is None:
            self.status_bar.showMessage("Load a recording first.")
            return
        pair = pair_name.strip()
        if not pair or pair.startswith("__SPACER_"):
            return
        if self._is_bipolar_marked(pair):
            self._remove_bipolar_annotation(pair)
            self.status_bar.showMessage(f"Removed {pair}.")
        else:
            self._add_bipolar_annotation(pair)
            self.status_bar.showMessage(f"Marked {pair}.")
        self._refresh_annotation_list()
        self._redraw_annotation_markers()
        self._sync_channel_list_annotation_style()
        self._save_annotations()

    def _clear_row_annotations_by_channel(self, channel_name: str) -> None:
        """Trace row click: remove annotations for contacts on that row (e.g. Fp1, F7)."""
        if not channel_name or channel_name.startswith("__SPACER_"):
            return
        if self.eeg_data is None:
            return

        contacts = self._contacts_from_row_name(channel_name)
        if not contacts:
            return

        removed = [c for c in contacts if self._is_contact_marked(c)]
        if not removed:
            self.status_bar.showMessage(f"No marks on {channel_name}.")
            return

        for c in removed:
            self._remove_contact_annotation(c)
        self.status_bar.showMessage(f"Cleared {', '.join(removed)} from {channel_name}.")
        self._refresh_annotation_list()
        self._redraw_annotation_markers()
        self._sync_channel_list_annotation_style()
        self._save_annotations()

    @staticmethod
    def _ten20_contact_equivalent(a: str, b: str) -> bool:
        """True if two contact tokens match, including T3/T7-style aliases."""
        A, B = a.strip().upper(), b.strip().upper()
        if A == B:
            return True
        if B in _TEN_TWENTY_CONTACT_ALIASES.get(A, set()):
            return True
        if A in _TEN_TWENTY_CONTACT_ALIASES.get(B, set()):
            return True
        return normalize_channel_name(A) == normalize_channel_name(B)

    def _montage_row_uses_contact(self, row_name: str, contact: str) -> bool:
        if row_name.startswith("__SPACER_"):
            return False
        c = contact.strip()
        if "-" in row_name:
            parts = [p.strip() for p in row_name.split("-", 1)]
            return any(self._ten20_contact_equivalent(p, c) for p in parts)
        return self._ten20_contact_equivalent(row_name, c)

    def _normalize_channel_annotations(self) -> None:
        """One annotation per scalp contact (split legacy Fp1-F7 rows)."""
        expanded: list[dict] = []
        for ann in self.channel_annotations:
            ch = (ann.get("channel") or "").strip()
            label = ann.get("label", "marked")
            if "-" in ch:
                for part in self._contacts_from_row_name(ch):
                    expanded.append({"channel": part, "label": label})
            elif ch:
                expanded.append({"channel": ch, "label": label})

        seen_keys: set[str] = set()
        unique: list[dict] = []
        for ann in expanded:
            contact = ann["channel"]
            disk_key = contact
            for key in TEN_TWENTY_SHAPE_NORM:
                if self._ten20_contact_equivalent(key, contact):
                    disk_key = key
                    break
            if disk_key in seen_keys:
                continue
            seen_keys.add(disk_key)
            unique.append(ann)
        self.channel_annotations = unique

    def _on_montage_1020_contact_clicked(self, contact: str):
        self._toggle_contact_annotation(contact)

    def _row_has_marked_contact(self, row_name: str) -> bool:
        return any(self._is_contact_marked(c) for c in self._contacts_from_row_name(row_name))

    def _sync_channel_list_annotation_style(self):
        """Orange text = marked contact (clips: exact list label only)."""
        marked_brush = QBrush(QColor(200, 90, 0))
        default_brush = QBrush()
        self.ch_list.blockSignals(True)
        for i in range(self.ch_list.count()):
            item = self.ch_list.item(i)
            name = item.text()
            if self.clips_mode:
                marked = self._is_contact_marked(name)
            else:
                marked = self._row_has_marked_contact(name)
            item.setForeground(marked_brush if marked else default_brush)
        self.ch_list.blockSignals(False)
        self._update_montage_1020_highlights()

    def _contacts_for_annotation_highlight(self, ann: dict) -> set[str]:
        ch = (ann.get("channel") or "").strip()
        return {ch} if ch else set()

    def _displayed_rows_for_contact(self, contact: str) -> list[str]:
        if self.clips_mode:
            return []
        return [
            ch for ch in self.displayed_channels
            if not ch.startswith("__SPACER_")
            and self._montage_row_uses_contact(ch, contact)
        ]

    def _all_contacts_from_annotations(self) -> set[str]:
        out: set[str] = set()
        for ann in self.channel_annotations:
            out |= self._contacts_for_annotation_highlight(ann)
        return out

    def _schematic_disk_keys_from_contacts(self, contacts: Iterable[str]) -> set[str]:
        lit: set[str] = set()
        clist = list(contacts)
        for key in TEN_TWENTY_SHAPE_NORM:
            for c in clist:
                if self._ten20_contact_equivalent(key, c):
                    lit.add(key)
                    break
        return lit

    def _update_montage_1020_highlights(self) -> None:
        if not hasattr(self, "montage_1020_label"):
            return
        contacts = self._all_contacts_from_annotations()
        self.montage_1020_label.set_highlight_contacts(
            self._schematic_disk_keys_from_contacts(contacts)
        )

    def _on_v_scroll(self, value: int):
        """Handle vertical scrollbar change in 12-channel mode."""
        self.v_scroll_offset = value
        if self.view_mode == "12ch" and self.displayed_channels:
            n_ch = len(self.displayed_channels)
            visible_bottom = self.v_scroll_offset
            visible_top = min(n_ch - 1, self.v_scroll_offset + 11)
            y_min = visible_bottom * CHANNEL_SPACING_UV - CHANNEL_SPACING_UV * 0.5
            y_max = visible_top * CHANNEL_SPACING_UV + CHANNEL_SPACING_UV * 0.5
            self.plot.vb.setRange(yRange=(y_min, y_max), padding=0)

    # ── EEG drawing ───────────────────────────────────────────────────────────

    def _draw_traces(self):
        """Clear and redraw all channel traces for the current time window."""
        # Disable auto-range BEFORE clearing and adding items so pyqtgraph
        # never auto-fits the y-axis around the newly added curves.  If this
        # is called after clear() the view box silently re-enables auto-range
        # and a single high-amplitude channel can expand the y window by
        # orders of magnitude, squeezing all other channels out of view.
        self.plot.vb.disableAutoRange()

        self.plot.clear()
        self.trace_items = []
        self.annotation_markers = []
        self.ictal_region = None

        if self.eeg_data is None or not self.displayed_channels:
            return

        n_ch = len(self.displayed_channels)
        t_start_rel = self.scroll_pos
        t_end_rel = min(self.total_duration, t_start_rel + VISIBLE_SECONDS)
        s_start = int(t_start_rel * self.fs)
        s_end = int(t_end_rel * self.fs)

        # Show absolute time on x-axis in iEEG window mode.
        t_start = self.time_offset_sec + t_start_rel
        t_end = self.time_offset_sec + t_end_rel
        times = self.time_offset_sec + (np.arange(s_start, s_end) / self.fs)

        y_min = -CHANNEL_SPACING_UV * 0.5
        y_max = (n_ch - 1) * CHANNEL_SPACING_UV + CHANNEL_SPACING_UV * 0.5
        self._add_spike_window_markers(t_start, t_end)

        soz_set = self.soz_channels
        midline_green = self.banana_midline_pairs if self.montage == "banana" else set()

        # Y ticks: one per channel
        tick_positions = []
        tick_labels = []

        for i, ch_name in enumerate(self.displayed_channels):
            baseline = i * CHANNEL_SPACING_UV

            ch_idx = self.channel_names_all.index(ch_name) if ch_name in self.channel_names_all else None
            is_soz = ch_name in soz_set
            if ch_name.startswith("__SPACER_"):
                label = ""
            else:
                label = f"\u2605 {ch_name}" if is_soz else ch_name

            if ch_idx is None:
                tick_positions.append(baseline)
                tick_labels.append(label)
                continue

            segment = self.eeg_data[s_start:s_end, ch_idx]
            y = baseline + segment * self.gain

            if ch_name.startswith("__SPACER_"):
                colour = (220, 220, 220)
            elif ch_name in midline_green:
                colour = (0, 140, 0)  # midline chains
            else:
                colour = SOZ_ELECTRODE_COLOR if is_soz else (30, 30, 200)
            pen = pg.mkPen(color=colour, width=0.8)
            curve = self.plot.plot(times, y, pen=pen)
            self.trace_items.append(curve)

            tick_positions.append(baseline)
            tick_labels.append(label)

        # Set Y axis ticks
        axis = self.plot.getAxis("left")
        axis.setTicks([list(zip(tick_positions, tick_labels))])
        axis.setStyle(tickFont=QFont("Monospace", 7))

        # Lock the view range — must come AFTER all items are added so that
        # pyqtgraph does not auto-range over the newly added curves.
        self.v_scroll_bar.hide()
        self.plot.vb.disableAutoRange()
        self.plot.vb.setRange(xRange=(t_start, t_end), yRange=(y_min, y_max), padding=0)

        # ── Update toolbar µV/mm readout ──────────────────────────────────
        self._update_scale_display()

        # ── Cursor crosshair line ─────────────────────────────────────────
        # Re-created every redraw so it always sits on top of all other items.
        self._cursor_line = pg.InfiniteLine(
            pos=0, angle=90,
            pen=pg.mkPen(color=(120, 120, 120), width=1, style=Qt.DashLine),
            movable=False,
        )
        self._cursor_line.setVisible(False)
        self.plot.addItem(self._cursor_line)

        # Redraw channel annotation markers
        self._redraw_annotation_markers()

    def _spike_highlight_bounds(self) -> tuple[float, float] | None:
        """Plot x-axis interval (s) for the red spike band."""
        if self.clips_mode:
            return (
                self.time_offset_sec + CLIP_SPIKE_REGION_START_SEC,
                self.time_offset_sec + CLIP_SPIKE_REGION_END_SEC,
            )
        if self.clip_spike_rel_sec is not None:
            t0 = self.time_offset_sec + float(self.clip_spike_rel_sec)
            return (t0 - SPIKE_MARK_HALF_WIDTH_SEC, t0 + SPIKE_MARK_HALF_WIDTH_SEC)
        if self.clip_spike_abs_sec is not None:
            spike_t = float(self.clip_spike_abs_sec)
            return (
                spike_t - SPIKE_MARK_HALF_WIDTH_SEC,
                spike_t + SPIKE_MARK_HALF_WIDTH_SEC,
            )
        return None

    def _add_spike_window_markers(self, t_view_start: float, t_view_end: float) -> None:
        """Red shaded band for the spike review window."""
        bounds = self._spike_highlight_bounds()
        if bounds is None:
            return
        t_lo, t_hi = bounds
        if t_hi < t_view_start or t_lo > t_view_end:
            return

        region = pg.LinearRegionItem(
            values=(t_lo, t_hi),
            orientation=pg.LinearRegionItem.Vertical,
            brush=pg.mkBrush(*SPIKE_MARK_FILL_COLOR),
            pen=pg.mkPen(color=SPIKE_MARK_LINE_COLOR, width=1.5),
            movable=False,
        )
        region.setZValue(-10)
        self.plot.addItem(region)
        self.trace_items.append(region)

    def _update_scale_display(self):
        """Compute and display µV/mm in the toolbar label."""
        vb = self.plot.vb
        pixel_h = vb.height()
        if pixel_h > 0 and self.eeg_data is not None:
            y_range = vb.viewRange()[1]
            data_span = y_range[1] - y_range[0]
            dpi = self.screen().logicalDotsPerInch() if self.screen() else 96
            px_per_mm = dpi / 25.4
            uv_per_mm = (data_span / pixel_h) * px_per_mm / self.gain
            self.gain_label.setText(f"{uv_per_mm:.1f} µV/mm")
        else:
            self.gain_label.setText("— µV/mm")

    def _redraw_annotation_markers(self):
        """Draw markers / shaded regions and text labels for annotations."""
        if self.clips_mode:
            return
        for item in self.annotation_markers:
            self.plot.removeItem(item)
        self.annotation_markers = []

        t_view_start = self.time_offset_sec + self.scroll_pos
        t_view_end = t_view_start + VISIBLE_SECONDS
        clip_start = float(self.time_offset_sec)
        clip_end = float(self.time_offset_sec + self.total_duration)

        for ann in self.channel_annotations:
            contact = (ann.get("channel") or "").strip()
            if not contact:
                continue
            t = float(ann.get("time_sec", clip_start))
            t2 = float(ann.get("end_sec", clip_end))

            if t2 < t_view_start or t > t_view_end:
                continue

            target_channels = self._displayed_rows_for_contact(contact)
            if not target_channels:
                continue

            types_text = ann.get("label") or self._format_types(ann.get("types"))
            for target_ch in target_channels:
                i = self.displayed_channels.index(target_ch)
                baseline = i * CHANNEL_SPACING_UV
                ch_idx = (self.channel_names_all.index(target_ch)
                          if target_ch in self.channel_names_all else None)
                if ch_idx is None:
                    continue

                # ── Full-clip row annotation (legacy range still supported) ─────
                y_bot = baseline - 0.5 * CHANNEL_SPACING_UV
                y_top = baseline + 0.5 * CHANNEL_SPACING_UV

                rect = pg.PlotCurveItem(
                    [t, t2, t2, t, t],
                    [y_bot, y_bot, y_top, y_top, y_bot],
                    pen=pg.mkPen(color=(255, 100, 0), width=1, style=Qt.DotLine),
                    fillLevel=y_bot,
                    fillBrush=pg.mkBrush(255, 165, 0, 50),
                )
                self.plot.addItem(rect)
                self.annotation_markers.append(rect)

                label_x = (t + t2) / 2.0
                label_y = baseline + CHANNEL_SPACING_UV * 0.35
                text_item = pg.TextItem(
                    text=f"{types_text} ({contact})", color=(255, 100, 0), anchor=(0.5, 1),
                )
                text_item.setFont(QFont("Sans", 7))
                text_item.setPos(label_x, label_y)
                self.plot.addItem(text_item)
                self.annotation_markers.append(text_item)

    def _update_view(self):
        """Efficiently update only the x/y range and trace data without full redraw."""
        self._draw_traces()

    # ── Gain control ──────────────────────────────────────────────────────────

    def _set_gain(self, value: float) -> None:
        self.gain = max(GAIN_MIN, min(GAIN_MAX, float(value)))
        self._draw_traces()

    def _increase_gain(self):
        self._set_gain(self.gain * GAIN_STEP)

    def _decrease_gain(self):
        self._set_gain(self.gain / GAIN_STEP)

    # ── Scroll ────────────────────────────────────────────────────────────────

    def _on_scroll(self, value: int):
        self.scroll_pos = float(value) / SCROLL_TICKS_PER_SEC
        self._draw_traces()


    # ── Global type controls ──────────────────────────────────────────────────

    def _on_global_type_changed(self):
        self.global_types = {
            "lvfa": self.global_lvfa_cb.isChecked(),
            "frequency": self.global_frequency_combo.currentText(),
            "pattern": self.global_pattern_combo.currentText(),
            "modifier": self.global_modifier_combo.currentText(),
            "custom_type": self.global_custom_type_edit.text().strip(),
        }

    def _reset_global_widgets(self):
        """Reset global annotation widgets to their default (empty) state."""
        for w in (self.global_lvfa_cb, self.global_frequency_combo,
                  self.global_pattern_combo, self.global_modifier_combo,
                  self.global_custom_type_edit):
            w.blockSignals(True)
        self.global_lvfa_cb.setChecked(False)
        self.global_frequency_combo.setCurrentIndex(0)
        self.global_pattern_combo.setCurrentIndex(0)
        self.global_modifier_combo.setCurrentIndex(0)
        self.global_custom_type_edit.clear()
        for w in (self.global_lvfa_cb, self.global_frequency_combo,
                  self.global_pattern_combo, self.global_modifier_combo,
                  self.global_custom_type_edit):
            w.blockSignals(False)

    # ── Click / drag to annotate ──────────────────────────────────────────────

    def eventFilter(self, obj, event):
        """Intercept mouse press/move/release on the plot scene to detect
        click (point annotation) vs. drag (range annotation)."""
        if obj is not self.plot.scene():
            return super().eventFilter(obj, event)

        etype = event.type()
        if etype == QEvent.GraphicsSceneMousePress and event.button() == Qt.LeftButton:
            self._on_mouse_press(event)
        elif etype == QEvent.GraphicsSceneMouseRelease and event.button() == Qt.LeftButton:
            self._on_mouse_release(event)
        elif etype == QEvent.GraphicsSceneMouseDoubleClick and event.button() == Qt.LeftButton:
            if self.clips_mode:
                return super().eventFilter(obj, event)
            scene_pos = event.scenePos()
            left_axis = self.plot.getAxis("left")
            if left_axis.sceneBoundingRect().contains(scene_pos):
                data_pos = self.plot.vb.mapSceneToView(scene_pos)
                ch_idx = self._y_to_channel_idx(data_pos.y())
                if ch_idx is not None:
                    ch_name = self.displayed_channels[ch_idx]
                    if ch_name in self.soz_channels:
                        self.soz_channels.discard(ch_name)
                        if self.view_mode == "soz":
                            self._rebuild_displayed_channels()
                            return True
                    else:
                        self.soz_channels.add(ch_name)
                    self._draw_traces()
                    return True
        elif etype == QEvent.GraphicsSceneMouseMove:
            self._update_cursor_line(event.scenePos())
        elif etype == QEvent.GraphicsSceneHoverMove:
            self._update_cursor_line(event.scenePos())
        elif etype == QEvent.GraphicsSceneWheel:
            delta = event.delta()  # positive = scroll up
            # Ctrl/Cmd + wheel → vertical channel scroll in 12ch mode
            if (QApplication.keyboardModifiers() & Qt.ControlModifier) and self.view_mode == "12ch":
                step = 1 if delta > 0 else -1
                new_val = max(0, min(self.v_scroll_bar.maximum(),
                                     self.v_scroll_offset + step))
                self.v_scroll_bar.setValue(new_val)
            return True  # consume so pyqtgraph doesn't zoom

        return super().eventFilter(obj, event)

    # -- low-level mouse helpers --

    def _scene_to_channel(self, scene_pos):
        """Convert a scene position to (time, channel_index) or None."""
        if not self.plot.vb.sceneBoundingRect().contains(scene_pos):
            return None
        vb = self.plot.vb
        data_pos = vb.mapSceneToView(scene_pos)
        t = data_pos.x()
        y = data_pos.y()

        n_ch = len(self.displayed_channels)
        if n_ch == 0:
            return None
        baselines = np.array([i * CHANNEL_SPACING_UV for i in range(n_ch)])
        distances = np.abs(baselines - y)
        ch_idx = int(np.argmin(distances))
        if distances[ch_idx] > CHANNEL_SPACING_UV * 0.6:
            return None
        return t, ch_idx

    def _y_to_channel_idx(self, y: float) -> int | None:
        """Map a y coordinate to the nearest channel index using midpoint
        boundaries.  Each channel *i* owns the vertical band from
        ``(i - 0.5) * CHANNEL_SPACING_UV`` to ``(i + 0.5) * CHANNEL_SPACING_UV``.
        Returns *None* when no channels are displayed."""
        n_ch = len(self.displayed_channels)
        if n_ch == 0:
            return None
        idx = round(y / CHANNEL_SPACING_UV)
        return max(0, min(n_ch - 1, idx))

    def _on_mouse_press(self, event):
        """Record drag start position on left mouse press."""
        if self.eeg_data is None or not self.displayed_channels:
            return
        result = self._scene_to_channel(event.scenePos())
        if result is None:
            return
        press_time, ch_idx = result
        self._drag_start_data = (press_time,)
        self._drag_channel_idx = ch_idx

    def _on_mouse_move(self, event):
        """Show visual feedback (bounded orange rectangle) while dragging,
        without moving the current time window."""
        if self._drag_start_data is None:
            return
        vb = self.plot.vb
        scene_pos = event.scenePos()
        self._drag_last_scene_pos = scene_pos  # store for auto-scroll callback
        if not self.plot.sceneBoundingRect().contains(scene_pos):
            return
        data_pos = vb.mapSceneToView(scene_pos)
        current_time = data_pos.x()
        current_y = data_pos.y()
        start_time = self._drag_start_data[0]

        # Keep window fixed while annotating.
        self._drag_scroll_dir = 0
        self._drag_scroll_timer.stop()

        # ── Visual feedback rectangle ─────────────────────────────────────
        # Only show feedback if drag distance is meaningful (>0.2 s)
        if abs(current_time - start_time) < 0.2:
            self._clear_drag_feedback()
            return

        self._update_drag_rect(current_time, current_y)

    def _update_drag_rect(self, current_time: float, current_y: float):
        """Create or update the bounded drag-feedback rectangle."""
        if self._drag_start_data is None:
            return
        start_time = self._drag_start_data[0]
        t_min = min(start_time, current_time)
        t_max = max(start_time, current_time)

        # Determine channel span using midpoint boundaries
        end_ch = self._y_to_channel_idx(current_y)
        if end_ch is None:
            return
        start_ch = self._drag_channel_idx
        ch_lo = min(start_ch, end_ch)
        ch_hi = max(start_ch, end_ch)
        y_bot = ch_lo * CHANNEL_SPACING_UV - 0.5 * CHANNEL_SPACING_UV
        y_top = ch_hi * CHANNEL_SPACING_UV + 0.5 * CHANNEL_SPACING_UV

        # Closed polygon rectangle: 5 points (pen = border, fillLevel = fill)
        xs = [t_min, t_max, t_max, t_min, t_min]
        ys = [y_bot, y_bot, y_top, y_top, y_bot]

        if self._drag_region is None:
            # Vertical dashed line at drag start
            self._drag_start_line = pg.InfiniteLine(
                pos=start_time, angle=90,
                pen=pg.mkPen(color=(255, 100, 0), width=2, style=Qt.DashLine),
            )
            self.plot.addItem(self._drag_start_line)
            # Single PlotCurveItem: closed polygon with fill + border
            self._drag_region = pg.PlotCurveItem(
                xs, ys,
                pen=pg.mkPen(color=(255, 100, 0), width=1, style=Qt.DashLine),
                fillLevel=y_bot,
                fillBrush=pg.mkBrush(255, 165, 0, 60),
            )
            self.plot.addItem(self._drag_region)
        else:
            self._drag_region.setData(xs, ys)
            self._drag_region.setFillLevel(y_bot)

    def _on_mouse_release(self, event):
        """Trace row click (no drag): clear row marks (non-clip mode only)."""
        if self._drag_start_data is None:
            return

        self._clear_drag_feedback()

        start_time = self._drag_start_data[0]
        start_ch_idx = self._drag_channel_idx
        self._drag_start_data = None
        self._drag_channel_idx = None

        if start_ch_idx is None:
            return

        scene_pos = event.scenePos()
        if not self.plot.sceneBoundingRect().contains(scene_pos):
            return
        data_pos = self.plot.vb.mapSceneToView(scene_pos)
        release_time = data_pos.x()

        if abs(release_time - start_time) > 0.2:
            return

        if self.clips_mode:
            return
        ch_name = self.displayed_channels[start_ch_idx]
        self._clear_row_annotations_by_channel(ch_name)

    def _clear_drag_feedback(self):
        """Remove temporary drag visual-feedback items and stop auto-scroll."""
        self._drag_scroll_timer.stop()
        self._drag_scroll_dir = 0
        self._drag_last_scene_pos = None
        if self._drag_region is not None:
            self.plot.removeItem(self._drag_region)
            self._drag_region = None
        if self._drag_start_line is not None:
            self.plot.removeItem(self._drag_start_line)
            self._drag_start_line = None

    def _update_cursor_line(self, scene_pos):
        """Move the dashed cursor line to the current mouse x position, or hide
        it when the pointer is outside the plot viewbox."""
        if self._cursor_line is None:
            return
        vb = self.plot.vb
        if vb.sceneBoundingRect().contains(scene_pos):
            data_x = vb.mapSceneToView(scene_pos).x()
            self._cursor_line.setPos(data_x)
            self._cursor_line.setVisible(True)
        else:
            self._cursor_line.setVisible(False)

    def _on_drag_auto_scroll(self):
        """Timer callback: shift the visible window while dragging near an edge."""
        if self._drag_start_data is None or self._drag_scroll_dir == 0:
            self._drag_scroll_timer.stop()
            return

        # Scroll by ~1 second per tick (50 ms interval → 20 s/s at full speed)
        step = 1.0 * self._drag_scroll_dir
        max_scroll = max(0.0, self.total_duration - VISIBLE_SECONDS)
        new_pos = max(0.0, min(max_scroll, self.scroll_pos + step))
        if new_pos == self.scroll_pos:
            return  # already at the limit

        # Update scroll position (triggers _draw_traces via signal)
        self.scroll_bar.blockSignals(True)
        self.scroll_bar.setValue(int(round(new_pos * SCROLL_TICKS_PER_SEC)))
        self.scroll_bar.blockSignals(False)
        self.scroll_pos = new_pos

        # Redraw traces (clears all items including drag feedback)
        self._drag_region = None
        self._drag_start_line = None
        self._draw_traces()

        # Re-create drag feedback at the latest mouse position
        if self._drag_last_scene_pos is not None:
            vb = self.plot.vb
            data_pos = vb.mapSceneToView(self._drag_last_scene_pos)
            self._update_drag_rect(data_pos.x(), data_pos.y())

    # ── Annotation list (right panel) ─────────────────────────────────────────

    @staticmethod
    def _format_types(types) -> str:
        """Convert a types value (dict or legacy list) to a human-readable string."""
        if types is None:
            return "marked"
        if isinstance(types, dict):
            parts = []
            if types.get("lvfa"):
                parts.append("LVFA")
            freq = types.get("frequency", "(none)")
            if freq != "(none)":
                parts.append(freq)
            pat = types.get("pattern", "(none)")
            if pat != "(none)":
                parts.append(pat)
            mod = types.get("modifier", "None")
            if mod != "None":
                parts.append(mod)
            ct = types.get("custom_type", "").strip()
            if ct:
                parts.append(ct)
            return ", ".join(parts) if parts else "(empty)"
        # Legacy list[str] fallback
        if isinstance(types, list):
            return ", ".join(types) if types else "(empty)"
        return str(types)

    def _refresh_annotation_list(self):
        self.annot_list.clear()
        for ann in self.channel_annotations:
            types_text = ann.get("label") or self._format_types(ann.get("types"))
            label = f"{ann['channel']}: {types_text}"
            self.annot_list.addItem(label)

    def _remove_channel_annotation(self, item: QListWidgetItem):
        row = self.annot_list.row(item)
        if 0 <= row < len(self.channel_annotations):
            self.channel_annotations.pop(row)
            self._refresh_annotation_list()
            if self.clips_mode:
                self._update_montage_1020_highlights()
            else:
                self._redraw_annotation_markers()
                self._sync_channel_list_annotation_style()
            self._save_annotations()

    # ── Save / load ───────────────────────────────────────────────────────────

    def _current_clip_option(self) -> dict | None:
        if not self.clips_mode or not self.current_patient:
            return None
        return self.ieeg_selection_options.get(self.current_patient)

    def _annotation_path(self) -> Path | None:
        """Clips CSV: annotations/<patient_id>/<ieeg_file_name>_spike_at_<t>.json"""
        if self.clips_mode:
            opt = self._current_clip_option()
            if opt and opt.get("annotation_relpath"):
                return ANNOTATIONS_DIR / str(opt["annotation_relpath"])
        if not self.current_patient or not self.current_recording:
            return None
        fname = f"{self.current_patient}_{self.current_recording}.json"
        return ANNOTATIONS_DIR / fname

    def _legacy_annotation_paths(self) -> list[Path]:
        """Load paths from older GUI saves (label + recording, or flat ieeg stem)."""
        if self.clips_mode and self.current_patient:
            return self._legacy_annotation_paths_for_clip(self.current_patient)
        paths: list[Path] = []
        if not self.current_patient or not self.current_recording:
            return paths
        paths.append(
            ANNOTATIONS_DIR / f"{self.current_patient}_{self.current_recording}.json"
        )
        return paths

    def _annotation_paths_to_try(self) -> list[Path]:
        primary = self._annotation_path()
        tried: set[Path] = set()
        out: list[Path] = []
        if primary is not None:
            out.append(primary)
            tried.add(primary)
        for legacy in self._legacy_annotation_paths():
            if legacy not in tried:
                out.append(legacy)
                tried.add(legacy)
        return out

    def _clip_has_saved_annotations(self, clip_id: str) -> bool:
        opt = self.ieeg_selection_options.get(clip_id)
        if not opt:
            return False
        rel = opt.get("annotation_relpath")
        if rel and (ANNOTATIONS_DIR / str(rel)).exists():
            return True
        stem = opt.get("annotation_stem")
        if stem and (ANNOTATIONS_DIR / f"{stem}.json").exists():
            return True
        for legacy in self._legacy_annotation_paths_for_clip(clip_id):
            if legacy.exists():
                return True
        return False

    def _legacy_annotation_paths_for_clip(self, clip_id: str) -> list[Path]:
        """Older saves keyed by GUI label text or flat stem."""
        paths: list[Path] = []
        opt = self.ieeg_selection_options.get(clip_id)
        if not opt:
            return paths
        stem = opt.get("annotation_stem")
        if stem:
            paths.append(ANNOTATIONS_DIR / f"{stem}.json")
        blinded = str(opt.get("dataset_id", "")).strip()
        if blinded:
            paths.append(ANNOTATIONS_DIR / "blinded" / f"{blinded}.json")
            paths.append(ANNOTATIONS_DIR / f"{blinded}.json")
        paths.append(ANNOTATIONS_DIR / "blinded" / f"{clip_id}.json")
        orig = str(opt.get("original_name", "")).strip()
        if orig:
            paths.append(ANNOTATIONS_DIR / f"{Path(orig).stem}.json")
        rec = str(opt.get("recording_name", "")).replace(".edf", "")
        display = str(opt.get("ieeg_file_name", ""))
        for label in (
            clip_id,
            display,
            f"{display} @ {opt.get('timestamp_hhmmss', '')}".strip(),
        ):
            if label:
                paths.append(ANNOTATIONS_DIR / f"{label}_{rec}.json")
        return paths

    def _patient_has_saved_annotations(self, patient_name: str) -> bool:
        if self.clips_mode:
            return self._clip_has_saved_annotations(patient_name)
        return any(ANNOTATIONS_DIR.glob(f"{patient_name}_*.json"))

    def _update_patient_dropdown_colors(self):
        if not self.patient_combo.count():
            return
        model = self.patient_combo.model()
        for i in range(self.patient_combo.count()):
            text = self.patient_combo.itemText(i)
            item = model.item(i)
            if item is None:
                continue
            if text.startswith("-- Select"):
                item.setForeground(QBrush(QColor(90, 90, 90)))
                continue
            clip_id = self.patient_combo.itemData(i)
            key = str(clip_id) if clip_id else text
            if self.clips_mode and self._clip_has_saved_annotations(key):
                item.setForeground(QBrush(QColor(0, 90, 200)))
            elif not self.clips_mode and self._patient_has_saved_annotations(text):
                item.setForeground(QBrush(QColor(0, 90, 200)))
            else:
                item.setForeground(QBrush(QColor(0, 0, 0)))

    def _save_annotations(self):
        path = self._annotation_path()
        if path is None:
            QMessageBox.warning(self, "Save", "No recording loaded.")
            return

        # Quick-mark workflow: global seizure-type metadata is intentionally disabled.
        global_types_out = None

        clean_channel_annotations = []
        for ann in self.channel_annotations:
            clean_channel_annotations.append({
                "channel": ann.get("channel"),
                "label": ann.get("label", "marked"),
            })

        opt = self._current_clip_option()
        payload = {
            "ieeg_file_name": opt.get("ieeg_file_name") if opt else None,
            "patient_id": opt.get("patient_id") if opt else self.patient_id_for_annotations,
            "timestamp_sec": (
                opt.get("timestamp_sec") if opt else None
            ) or (opt.get("spike_time_sec") if opt else None),
            "clip_gui_label": (
                opt.get("ieeg_file_name") if opt else self.patient_combo.currentText()
            ),
            "recording": (
                opt.get("display_id") if opt else self.current_recording
            ),
            "global_types": global_types_out,
            "channel_annotations": clean_channel_annotations,
            "soz_channels": sorted(self.soz_channels),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2))
        self._update_patient_dropdown_colors()
        self.status_bar.showMessage(f"Saved annotations → {path.resolve()}")

    def _save_and_remove_current_patient(self):
        """Save current annotations and remove this option from dropdown list."""
        self._save_annotations()
        self._remove_current_patient_option()

    def _remove_current_patient_option(self):
        if not (self.ieeg_mode and (self.ieeg_edf_options_csv or self.clips_mode)):
            return
        current = self.current_patient
        if not current:
            return
        if current not in self.patients:
            return

        self.patients = [p for p in self.patients if p != current]
        self.ieeg_patient_datasets.pop(current, None)
        self.ieeg_spike_time_by_dataset.pop(current, None)

        self.patient_combo.blockSignals(True)
        self.patient_combo.clear()
        placeholder = "-- Select clip --" if self.clips_mode else "-- Select Patient --"
        self.patient_combo.addItem(placeholder)
        self.patient_combo.addItems(self.patients)
        self.patient_combo.setCurrentIndex(0)
        self.patient_combo.blockSignals(False)

        self.current_patient = None
        self.current_recording = None
        self._clear_recording_state()
        self._update_clip_info_label(None)

        if self.patients:
            self.status_bar.showMessage(
                f"Saved and removed {current}. Select next clip."
                if self.clips_mode else
                f"Saved and removed {current}. Select next patient."
            )
        else:
            self.status_bar.showMessage(
                f"Saved and removed {current}. No clips remaining."
                if self.clips_mode else
                f"Saved and removed {current}. No patient options remaining."
            )

    def _load_soz_channels(self):
        """Populate self.soz_channels from hardcoded data (fallback when no
        saved annotation exists yet)."""
        self.soz_channels = set()
        patient_rid = self.current_patient
        if not patient_rid or not self.channel_names_all:
            return

        soz_rows = self.soz_df[self.soz_df["rid"] == patient_rid]["soz_electrode"].tolist()
        for name in soz_rows:
            m = match_channel(name, self.channel_names_all)
            if m:
                self.soz_channels.add(m)

    def _load_annotations(self):
        path = None
        for candidate in self._annotation_paths_to_try():
            if candidate.exists():
                path = candidate
                break
        if path is None:
            return
        try:
            payload = json.loads(path.read_text())
            raw_gt = payload.get("global_types")

            # ── Backward-compat: old format was a list of strings ────────
            if isinstance(raw_gt, list):
                # Best-effort migration: set LVFA if it was in the list
                self.global_types = {
                    "lvfa": any("LVFA" in s for s in raw_gt),
                    "frequency": "(none)",
                    "pattern": "(none)",
                    "modifier": "None",
                }
            elif isinstance(raw_gt, dict):
                self.global_types = {
                    "lvfa": raw_gt.get("lvfa", False),
                    "frequency": raw_gt.get("frequency", "(none)"),
                    "pattern": raw_gt.get("pattern", "(none)"),
                    "modifier": raw_gt.get("modifier", "None"),
                    "custom_type": raw_gt.get("custom_type", ""),
                }
            else:
                self.global_types = {
                    "lvfa": False, "frequency": "(none)",
                    "pattern": "(none)", "modifier": "None", "custom_type": "",
                }

            self.channel_annotations = payload.get("channel_annotations", [])
            self._normalize_channel_annotations()
            if self.clips_mode and self.global_types.get("custom_type", "") == "no_spike":
                if not self._has_no_spike_annotation():
                    self.channel_annotations.append({
                        "channel": "(clip)",
                        "label": "no_spike",
                    })
                self.global_types["custom_type"] = ""

            saved_soz = payload.get("soz_channels")
            if saved_soz is not None and self.channel_names_all:
                self.soz_channels = set()
                for name in saved_soz:
                    m = match_channel(name, self.channel_names_all)
                    if m:
                        self.soz_channels.add(m)

            # Populate global widgets from loaded state
            self._reset_global_widgets()
            for w in (self.global_lvfa_cb, self.global_frequency_combo,
                      self.global_pattern_combo, self.global_modifier_combo,
                      self.global_custom_type_edit):
                w.blockSignals(True)
            self.global_lvfa_cb.setChecked(self.global_types["lvfa"])
            idx = self.global_frequency_combo.findText(self.global_types["frequency"])
            if idx >= 0:
                self.global_frequency_combo.setCurrentIndex(idx)
            idx = self.global_pattern_combo.findText(self.global_types["pattern"])
            if idx >= 0:
                self.global_pattern_combo.setCurrentIndex(idx)
            idx = self.global_modifier_combo.findText(self.global_types["modifier"])
            if idx >= 0:
                self.global_modifier_combo.setCurrentIndex(idx)
            self.global_custom_type_edit.setText(self.global_types.get("custom_type", ""))
            for w in (self.global_lvfa_cb, self.global_frequency_combo,
                      self.global_pattern_combo, self.global_modifier_combo,
                      self.global_custom_type_edit):
                w.blockSignals(False)

            if hasattr(self, "btn_ieeg_no_spike") and self.btn_ieeg_no_spike.isCheckable():
                self.btn_ieeg_no_spike.blockSignals(True)
                self.btn_ieeg_no_spike.setChecked(
                    self.global_types.get("custom_type", "") == "no_spike"
                )
                self.btn_ieeg_no_spike.blockSignals(False)

            self._refresh_annotation_list()
            self._sync_channel_list_annotation_style()
        except Exception as exc:
            print(f"Could not load annotations: {exc}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Seizure Annotation GUI")
    parser.add_argument(
        "--data_dir",
        default=None,
        help="Path to the BIDS-like data directory containing sub-RID* folders",
    )
    parser.add_argument(
        "--ieeg_dataset_id",
        default=None,
        help="iEEG.org dataset id/name to stream directly (e.g. I004_A0003_D001).",
    )
    parser.add_argument("--ieeg_username", default=None, help="iEEG.org username.")
    parser.add_argument(
        "--ieeg_password_env",
        default="IEEG_PASSWORD",
        help="Environment variable name containing iEEG.org password.",
    )
    parser.add_argument(
        "--ieeg_password",
        default=None,
        help="Discouraged: use IEEG_PASSWORD env var instead (avoids argv exposure).",
    )
    parser.add_argument(
        "--patient_id",
        default=None,
        help="Prefix used when saving/loading annotations (and matching SOZ list). Default: ieeg.",
    )
    parser.add_argument(
        "--ictal_onset_sec",
        type=float,
        default=None,
        help="Ictal onset seconds relative to recording start (used if iEEG events are not provided).",
    )
    parser.add_argument(
        "--ictal_duration_sec",
        type=float,
        default=None,
        help="Ictal duration seconds relative to recording start (used if iEEG events are not provided).",
    )
    parser.add_argument(
        "--ieeg_edf_options_csv",
        default=None,
        help=(
            "CSV (e.g. sz-gui/selections.csv) containing a `Filename` column with "
            "<dataset_or_edf_id>_spike_at_<time>.edf. The GUI will parse dataset_or_edf_id "
            "for dropdown options and skip ids it can't open."
        ),
    )
    parser.add_argument(
        "--ieeg_patient_datasets_csv",
        default=None,
        help=(
            "CSV mapping patients to iEEG dataset ids for iEEG streaming mode. "
            "Expected columns: `patient_id` and `ieeg_dataset_id` (customizable via --ieeg_patient_col/--ieeg_dataset_col)."
        ),
    )
    parser.add_argument(
        "--ieeg_patient_col",
        default="patient_id",
        help="Column name in --ieeg_patient_datasets_csv for the patient id.",
    )
    parser.add_argument(
        "--ieeg_dataset_col",
        default="ieeg_dataset_id",
        help="Column name in --ieeg_patient_datasets_csv for the iEEG dataset id.",
    )
    parser.add_argument(
        "--clips_csv",
        default=None,
        help=(
            "Path to final_selected_200_events.csv for 200-clip spike review."
        ),
    )
    parser.add_argument(
        "--clip_edf_dir",
        default=None,
        help=(
            "Directory of pre-built clip EDFs (from build_clip_edfs.py). "
            "When a matching file exists, the GUI loads locally instead of iEEG.org."
        ),
    )
    parser.add_argument(
        "--clip_scan_dir",
        default=None,
        help=(
            "Directory of clip EDFs to review by scanning the folder directly "
            "(blinded review, no CSV). Every *.edf becomes a clip option."
        ),
    )
    args = parser.parse_args()

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    ieeg_password = args.ieeg_password
    if not ieeg_password:
        ieeg_password = os.environ.get(args.ieeg_password_env)

    ieeg_streaming = (
        args.ieeg_dataset_id
        or args.ieeg_patient_datasets_csv
        or args.ieeg_edf_options_csv
        or args.clips_csv
    )
    # Folder-scan review reads only local clip EDFs (blinded, no CSV/creds).
    folder_scan = bool(args.clip_scan_dir) and not ieeg_streaming
    # When clips_csv is paired with a local clip_edf_dir, the GUI can run
    # purely off pre-exported EDFs and doesn't need iEEG credentials. We only
    # enforce credentials for the true streaming paths.
    needs_ieeg_creds = ieeg_streaming and not (
        args.clips_csv and args.clip_edf_dir
    )
    if needs_ieeg_creds:
        if not args.ieeg_username:
            raise SystemExit("Missing --ieeg_username for iEEG mode.")
        if not ieeg_password:
            raise SystemExit(
                "Missing iEEG password. Provide --ieeg_password or set env var via --ieeg_password_env."
            )

    if ieeg_streaming or folder_scan:
        window = SeizureAnnotationGUI(
            data_dir=None,
            ieeg_dataset_id=args.ieeg_dataset_id,
            ieeg_username=args.ieeg_username,
            ieeg_password=ieeg_password,
            patient_id=args.patient_id,
            ictal_onset_sec=args.ictal_onset_sec,
            ictal_duration_sec=args.ictal_duration_sec,
            ieeg_patient_datasets_csv=args.ieeg_patient_datasets_csv,
            ieeg_patient_col=args.ieeg_patient_col,
            ieeg_dataset_col=args.ieeg_dataset_col,
            ieeg_edf_options_csv=args.ieeg_edf_options_csv,
            clips_csv=args.clips_csv,
            clip_edf_dir=args.clip_edf_dir,
            clip_scan_dir=args.clip_scan_dir,
        )
    else:
        window = SeizureAnnotationGUI(data_dir=args.data_dir)

    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()