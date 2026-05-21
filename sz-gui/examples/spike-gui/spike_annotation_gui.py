#!/usr/bin/env python3
"""
Spike Annotation GUI

A GUI application for manual spike annotation by clinicians.
Displays iEEG traces with pre-stimulation (10s), during stimulation (30s), 
and post-stimulation (11s) periods for visual spike marking.

Features:
- Single channel display (channel of interest extracted from filename)
- Bipolar vs raw montage toggle
- Interpolated vs raw signal toggle (simulates spike detector preprocessing)
- Sham stimulations in pre/post periods with interpolation
- Gain adjustment with arrow keys
- Click to mark/unmark spikes
- Sequential EDF file processing
- Progress tracking and resumption
- Scrollable display (15s visible, 51s total)

Usage:
    python spike_annotation_gui.py --edf_dir /path/to/edf/files
"""

import sys
import os
from pathlib import Path
import numpy as np
import json
from datetime import datetime
import pandas as pd

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSlider, QCheckBox, QScrollBar, QFileDialog,
    QMessageBox, QStatusBar, QShortcut
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QKeySequence

import pyqtgraph as pg
import mne

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from tools.iEEG_helper_functions import automatic_bipolar_montage, notch_filter, bandpass_filter


class SpikeAnnotationGUI(QMainWindow):
    """Main GUI window for spike annotation."""
    
    def __init__(self, edf_dir=None):
        super().__init__()
        
        # Configuration
        self.edf_dir = edf_dir
        self.edf_files = []
        self.current_file_idx = 0
        
        # Data storage
        self.raw_data = None
        self.raw_data_single_channel = None  # Raw version of channel of interest
        self.bipolar_data_single_channel = None  # Bipolar version of channel of interest
        self.interpolated_data = None
        self.current_data = None
        self.channel_names = None
        self.channel_of_interest = None  # Name of the channel to display
        self.channel_of_interest_idx = None  # Index in the data array
        self.bipolar_channel_name = None  # Bipolar name (e.g., "LA11-LA12")
        self.raw_channel_name = None  # Raw name (e.g., "LA11")
        self.fs = None
        self.time_vector = None
        self.gap_ranges = []  # Store interpolated gap ranges
        
        # Stimulation metadata
        self.stim_times_df = None  # DataFrame with stimulation times
        self.current_patient_id = None
        self.current_stim_channel = None
        self.current_ieeg_stim_time = None  # Time on ieeg.org when stim occurred
        
        # Display parameters
        self.gain = 1.0
        self.visible_duration = 15.0  # seconds visible at once
        self.total_duration = 51.0    # total duration: 10s pre + 30s stim + 11s post
        self.scroll_position = 0.0    # current scroll position in seconds
        self.show_interpolated = True
        self.show_bipolar = True  # Toggle between bipolar and raw signal
        
        # Spike annotations
        self.spike_annotations = []  # List of (channel_idx, time_in_seconds)
        self.spike_click_threshold = 0.2  # seconds - clicking within this removes spike
        
        # Progress tracking
        self.progress_file = os.path.join(
            os.path.dirname(__file__), 'annotation_progress.json'
        )
        self.annotations_dir = os.path.join(
            os.path.dirname(__file__), 'annotations'
        )
        os.makedirs(self.annotations_dir, exist_ok=True)
        
        # Load progress
        self.completed_files = self.load_progress()
        
        # Load stimulation times metadata
        self.load_stim_times_metadata()
        
        # Initialize UI
        self.initUI()
        
        # Load EDF files
        if self.edf_dir:
            self.load_edf_list()
            self.load_next_uncompleted_file()
    
    def initUI(self):
        """Initialize the user interface."""
        self.setWindowTitle('Spike Annotation GUI')
        self.setGeometry(100, 100, 1600, 900)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)
        
        # Top control panel
        control_layout = QHBoxLayout()
        
        # File info label
        self.file_label = QLabel('No file loaded')
        self.file_label.setStyleSheet('font-weight: bold; font-size: 12pt;')
        control_layout.addWidget(self.file_label)
        
        control_layout.addStretch()
        
        # iEEG.org stimulation time label
        self.ieeg_time_label = QLabel('iEEG.org stim time: N/A')
        self.ieeg_time_label.setStyleSheet('font-weight: bold; font-size: 11pt; color: #0066cc;')
        control_layout.addWidget(self.ieeg_time_label)
        
        control_layout.addStretch()
        
        # Progress label
        self.progress_label = QLabel('Progress: 0/0')
        control_layout.addWidget(self.progress_label)
        
        # Select directory button
        self.select_dir_btn = QPushButton('Select EDF Directory')
        self.select_dir_btn.clicked.connect(self.select_edf_directory)
        control_layout.addWidget(self.select_dir_btn)
        
        main_layout.addLayout(control_layout)
        
        # Second control row
        control_layout2 = QHBoxLayout()
        
        # Interpolated signal checkbox
        self.interp_checkbox = QCheckBox('Show Interpolated Signal')
        self.interp_checkbox.setChecked(True)
        self.interp_checkbox.stateChanged.connect(self.toggle_interpolation)
        control_layout2.addWidget(self.interp_checkbox)
        
        # Bipolar signal checkbox
        self.bipolar_checkbox = QCheckBox('Show Bipolar Signal')
        self.bipolar_checkbox.setChecked(True)
        self.bipolar_checkbox.stateChanged.connect(self.toggle_bipolar)
        control_layout2.addWidget(self.bipolar_checkbox)
        
        # Gain control
        gain_label = QLabel('Gain:')
        control_layout2.addWidget(gain_label)
        
        self.gain_label = QLabel(f'{self.gain:.2f}x')
        self.gain_label.setMinimumWidth(60)
        control_layout2.addWidget(self.gain_label)
        
        gain_info = QLabel('(Use ↑↓ arrow keys)')
        gain_info.setStyleSheet('color: gray; font-size: 9pt;')
        control_layout2.addWidget(gain_info)
        
        control_layout2.addStretch()
        
        # Spike count
        self.spike_count_label = QLabel('Spikes marked: 0')
        self.spike_count_label.setStyleSheet('font-size: 11pt; color: blue;')
        control_layout2.addWidget(self.spike_count_label)
        
        main_layout.addLayout(control_layout2)
        
        # Plot widget
        self.plot_widget = pg.GraphicsLayoutWidget()
        self.plot_widget.setBackground('w')
        main_layout.addWidget(self.plot_widget)
        
        # Initialize plot
        self.plot = self.plot_widget.addPlot()
        self.plot.setLabel('left', 'Channel')
        self.plot.setLabel('bottom', 'Time (s) relative to stimulation')
        self.plot.showGrid(x=True, y=False, alpha=0.3)
        
        # Enable mouse interaction
        self.plot.scene().sigMouseClicked.connect(self.on_plot_clicked)
        
        # Scroll bar
        scroll_layout = QHBoxLayout()
        scroll_label = QLabel('Scroll:')
        scroll_layout.addWidget(scroll_label)
        
        self.scroll_bar = QScrollBar(Qt.Horizontal)
        self.scroll_bar.setMinimum(0)
        self.scroll_bar.setMaximum(int((self.total_duration - self.visible_duration) * 10))
        self.scroll_bar.setValue(0)
        self.scroll_bar.valueChanged.connect(self.on_scroll)
        scroll_layout.addWidget(self.scroll_bar)
        
        main_layout.addLayout(scroll_layout)
        
        # Bottom control panel
        bottom_layout = QHBoxLayout()
        
        # Instructions
        instructions = QLabel(
            'Click on trace to mark spikes • Click near existing spike to remove • '
            '↑↓ to adjust gain • Enter to save & next file • Red lines = real stim • Blue dots = sham stim'
        )
        instructions.setStyleSheet('color: gray; font-size: 10pt;')
        bottom_layout.addWidget(instructions)
        
        bottom_layout.addStretch()
        
        # Navigation buttons
        self.prev_btn = QPushButton('← Previous')
        self.prev_btn.clicked.connect(self.load_previous_file)
        bottom_layout.addWidget(self.prev_btn)
        
        self.next_btn = QPushButton('Save & Next (Enter) →')
        self.next_btn.clicked.connect(self.save_and_next)
        self.next_btn.setStyleSheet('font-weight: bold;')
        bottom_layout.addWidget(self.next_btn)
        
        main_layout.addLayout(bottom_layout)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage('Ready')
        
        # Keyboard shortcuts
        QShortcut(QKeySequence(Qt.Key_Return), self, self.save_and_next)
        QShortcut(QKeySequence(Qt.Key_Enter), self, self.save_and_next)
        QShortcut(QKeySequence(Qt.Key_Up), self, self.increase_gain)
        QShortcut(QKeySequence(Qt.Key_Down), self, self.decrease_gain)
        QShortcut(QKeySequence(Qt.Key_Left), self, lambda: self.scroll_bar.setValue(
            max(0, self.scroll_bar.value() - 10)))
        QShortcut(QKeySequence(Qt.Key_Right), self, lambda: self.scroll_bar.setValue(
            min(self.scroll_bar.maximum(), self.scroll_bar.value() + 10)))
    
    def load_stim_times_metadata(self):
        """Load stimulation times from CSV file."""
        # Try to find the CSV file in the expected location
        script_dir = os.path.dirname(__file__)
        csv_path = os.path.join(script_dir, '..', 'data', 'pt-metadata', 'first_stim_times_per_channel.csv')
        
        if not os.path.exists(csv_path):
            print(f"Warning: Could not find stimulation times CSV at {csv_path}")
            self.stim_times_df = None
            return
        
        try:
            # Load the CSV
            df = pd.read_csv(csv_path)
            
            # Parse patient IDs (remove 'HUP' prefix)
            df['patient_id'] = df['hup_id'].str.replace('HUP', '', regex=False).astype(int)
            
            # Parse first_stim_time (remove brackets and convert to float)
            def extract_stim_time(value):
                if pd.isna(value):
                    return None
                str_val = str(value).strip('[]')
                try:
                    return float(str_val)
                except:
                    return None
            
            df['stim_time_seconds'] = df['first_stim_time'].apply(extract_stim_time)
            
            # Keep only needed columns
            self.stim_times_df = df[['patient_id', 'stim_ch', 'stim_time_seconds']].copy()
            
            print(f"Loaded stimulation times for {len(self.stim_times_df)} channel pairs")
            
        except Exception as e:
            print(f"Error loading stimulation times: {e}")
            self.stim_times_df = None
    
    def parse_edf_filename(self, edf_file):
        """
        Parse EDF filename to extract patient ID and stim channel.
        Expected format: HUP{patient_id}_{stim_ch}_{recording_ch}.edf
        Example: HUP211_LF1-LF2_LF1.edf
        
        Returns:
            (patient_id, stim_channel) or (None, None) if parsing fails
        """
        try:
            filename = edf_file.stem  # Get filename without extension
            parts = filename.split('_')
            
            if len(parts) < 3:
                return None, None
            
            # Extract patient ID (remove 'HUP' prefix)
            patient_str = parts[0].replace('HUP', '')
            patient_id = int(patient_str)
            
            # Extract stim channel
            stim_channel = parts[1]
            
            return patient_id, stim_channel
            
        except Exception as e:
            print(f"Error parsing filename {edf_file.name}: {e}")
            return None, None
    
    def lookup_ieeg_stim_time(self, patient_id, stim_channel):
        """
        Look up the ieeg.org stimulation time for a given patient and channel.
        
        Returns:
            float: Stimulation time in seconds, or None if not found
        """
        if self.stim_times_df is None or patient_id is None or stim_channel is None:
            return None
        
        try:
            # Find matching row
            match = self.stim_times_df[
                (self.stim_times_df['patient_id'] == patient_id) &
                (self.stim_times_df['stim_ch'] == stim_channel)
            ]
            
            if len(match) > 0:
                return match.iloc[0]['stim_time_seconds']
            else:
                print(f"No stimulation time found for HUP{patient_id}, {stim_channel}")
                return None
                
        except Exception as e:
            print(f"Error looking up stimulation time: {e}")
            return None
    
    def update_ieeg_time_label(self):
        """Update the ieeg.org time label with current stimulation time."""
        if self.current_ieeg_stim_time is not None:
            # Format the time nicely
            hours = int(self.current_ieeg_stim_time // 3600)
            minutes = int((self.current_ieeg_stim_time % 3600) // 60)
            seconds = self.current_ieeg_stim_time % 60
            
            time_str = f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"
            self.ieeg_time_label.setText(
                f'iEEG.org stim time: {time_str} ({self.current_ieeg_stim_time:.3f}s)'
            )
            self.ieeg_time_label.setToolTip(
                f"First stimulation occurs at {self.current_ieeg_stim_time:.3f} seconds on ieeg.org\n"
                f"Patient: HUP{self.current_patient_id}, Channel: {self.current_stim_channel}"
            )
        else:
            self.ieeg_time_label.setText('iEEG.org stim time: N/A')
            self.ieeg_time_label.setToolTip('Stimulation time not available for this file')
    
    def select_edf_directory(self):
        """Open file dialog to select EDF directory."""
        directory = QFileDialog.getExistingDirectory(
            self, 'Select EDF Directory', ''
        )
        if directory:
            self.edf_dir = directory
            self.load_edf_list()
            self.load_next_uncompleted_file()
    
    def load_edf_list(self):
        """Load list of EDF files from directory."""
        if not self.edf_dir or not os.path.exists(self.edf_dir):
            QMessageBox.warning(self, 'Error', 'Invalid EDF directory')
            return
        
        # Find all EDF files
        edf_path = Path(self.edf_dir)
        self.edf_files = sorted(list(edf_path.glob('*.edf')))
        
        if not self.edf_files:
            QMessageBox.warning(self, 'Error', 'No EDF files found in directory')
            return
        
        self.status_bar.showMessage(f'Found {len(self.edf_files)} EDF files')
        self.update_progress_label()
    
    def load_next_uncompleted_file(self):
        """Load the first uncompleted file."""
        if not self.edf_files:
            return
        
        # Find first uncompleted file
        for idx, edf_file in enumerate(self.edf_files):
            if str(edf_file) not in self.completed_files:
                self.current_file_idx = idx
                self.load_current_file()
                return
        
        # All files completed
        QMessageBox.information(
            self, 'Complete', 
            'All files have been annotated! Great work!'
        )
        self.status_bar.showMessage('All files completed')
    
    def load_current_file(self):
        """Load the current EDF file."""
        if not self.edf_files or self.current_file_idx >= len(self.edf_files):
            return
        
        edf_file = self.edf_files[self.current_file_idx]
        self.status_bar.showMessage(f'Loading {edf_file.name}...')
        
        try:
            # Parse filename to get patient ID and stim channel
            self.current_patient_id, self.current_stim_channel = self.parse_edf_filename(edf_file)
            
            # Look up ieeg.org stimulation time
            self.current_ieeg_stim_time = self.lookup_ieeg_stim_time(
                self.current_patient_id, 
                self.current_stim_channel
            )
            
            # Update the ieeg.org time label
            self.update_ieeg_time_label()
            
            # Extract channel of interest from filename (last channel name before .edf)
            # Expected format: HUP{patient_id}_{stim_ch}_{recording_ch}.edf
            # recording_ch might be bipolar format like "LA01-LA02"
            filename_parts = edf_file.stem.split('_')
            self.channel_of_interest = filename_parts[-1] if filename_parts else None
            
            # Load EDF file using MNE
            raw = mne.io.read_raw_edf(str(edf_file), preload=True, verbose=False)
            
            # Get data
            data = raw.get_data().T  # Shape: (n_samples, n_channels)
            channel_names = raw.ch_names
            self.fs = raw.info['sfreq']
            
            # Check if channel_of_interest is a bipolar name (contains hyphen)
            if self.channel_of_interest and '-' in self.channel_of_interest:
                # Parse bipolar name to get constituent channels
                bipolar_parts = self.channel_of_interest.split('-')
                if len(bipolar_parts) == 2:
                    ch1_name, ch2_name = bipolar_parts[0], bipolar_parts[1]
                    
                    # Find both channels in the data (handle leading zeros)
                    ch1_idx = self._find_channel_index(ch1_name, channel_names)
                    ch2_idx = self._find_channel_index(ch2_name, channel_names)
                    
                    if ch1_idx is not None and ch2_idx is not None:
                        # Use first channel as the "raw" channel
                        self.raw_data_single_channel = data[:, ch1_idx:ch1_idx+1]
                        self.raw_channel_name = channel_names[ch1_idx]
                        
                        # Create bipolar signal manually from these specific channels
                        bipolar_signal = data[:, ch1_idx] - data[:, ch2_idx]
                        self.bipolar_data_single_channel = bipolar_signal.reshape(-1, 1)
                        self.bipolar_channel_name = f"{channel_names[ch1_idx]}-{channel_names[ch2_idx]}"
                        
                        print(f"Created bipolar channel from filename: {self.bipolar_channel_name}")
                        print(f"Raw channel: {self.raw_channel_name}")
                    else:
                        print(f"Could not find both channels {ch1_name} and {ch2_name} in EDF")
                        # Fall back to last channel
                        channel_idx = len(channel_names) - 1
                        self.raw_data_single_channel = data[:, channel_idx:channel_idx+1]
                        self.raw_channel_name = channel_names[channel_idx]
                        self.bipolar_data_single_channel, self.bipolar_channel_name = self._create_bipolar_channel(
                            data, channel_names, channel_idx
                        )
                else:
                    print(f"Could not parse bipolar name: {self.channel_of_interest}")
                    # Fall back to last channel
                    channel_idx = len(channel_names) - 1
                    self.raw_data_single_channel = data[:, channel_idx:channel_idx+1]
                    self.raw_channel_name = channel_names[channel_idx]
                    self.bipolar_data_single_channel, self.bipolar_channel_name = self._create_bipolar_channel(
                        data, channel_names, channel_idx
                    )
            else:
                # Not a bipolar name, use original logic
                # Find channel of interest in raw data (handle leading zeros)
                channel_idx = self._find_channel_index(self.channel_of_interest, channel_names)
                
                if channel_idx is not None:
                    self.raw_data_single_channel = data[:, channel_idx:channel_idx+1]  # Keep 2D
                    self.raw_channel_name = channel_names[channel_idx]
                    print(f"Found channel of interest: {self.raw_channel_name}")
                else:
                    # Fall back to last channel
                    self.raw_data_single_channel = data[:, -1:]
                    self.raw_channel_name = channel_names[-1]
                    self.channel_of_interest = channel_names[-1]
                    print(f"Using last channel: {self.raw_channel_name}")
                
                # Create bipolar version of the channel of interest
                self.bipolar_data_single_channel, self.bipolar_channel_name = self._create_bipolar_channel(
                    data, channel_names, channel_idx if channel_idx is not None else len(channel_names) - 1
                )
            
            # Apply filters to both raw and bipolar channels (same preprocessing as spike detector)
            print("  Applying preprocessing filters...")
            self.raw_data_single_channel = self.apply_filters(self.raw_data_single_channel)
            self.bipolar_data_single_channel = self.apply_filters(self.bipolar_data_single_channel)
            
            # Set current channel data based on bipolar checkbox
            current_channel_data = (self.bipolar_data_single_channel if self.show_bipolar 
                                   else self.raw_data_single_channel)
            
            # Create interpolated version (simulate what spike detector sees)
            self.interpolated_data, self.gap_ranges = self.create_interpolated_data(
                current_channel_data
            )
            
            # Set current data based on interpolation checkbox
            self.current_data = (self.interpolated_data if self.show_interpolated 
                               else current_channel_data)
            
            # Set channel names for display (single channel)
            current_name = (self.bipolar_channel_name if self.show_bipolar 
                          else self.raw_channel_name)
            self.channel_names = np.array([current_name])
            
            # Create time vector (relative to stimulation at t=0)
            self.time_vector = np.arange(self.current_data.shape[0]) / self.fs - 10.0
            
            # Load existing annotations if available
            self.load_existing_annotations(edf_file)
            
            # Reset scroll position
            self.scroll_position = 0.0
            self.scroll_bar.setValue(0)
            
            # Update display
            self.file_label.setText(f'File: {edf_file.name} | Channel: {current_name}')
            self.update_progress_label()
            self.update_plot()
            
            self.status_bar.showMessage(f'Loaded {edf_file.name}')
            
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Failed to load file:\n{str(e)}')
            self.status_bar.showMessage('Error loading file')
    
    def _find_channel_index(self, target_channel, channel_names):
        """
        Find channel index, handling potential leading zeros.
        E.g., 'LA1' should match 'LA01' or 'LA001'
        
        Args:
            target_channel: Channel name from filename (e.g., 'LA1')
            channel_names: List of channel names in the EDF file
            
        Returns:
            Index of the channel, or None if not found
        """
        if not target_channel:
            return None
        
        # First try exact match
        if target_channel in channel_names:
            return channel_names.index(target_channel)
        
        # Try with normalized names (remove leading zeros from numbers)
        def normalize_channel_name(name):
            """Remove leading zeros from numeric portions of channel name."""
            import re
            # Split into parts (letters and numbers)
            parts = re.findall(r'[A-Za-z]+|\d+', name)
            normalized = []
            for part in parts:
                if part.isdigit():
                    normalized.append(str(int(part)))  # Remove leading zeros
                else:
                    normalized.append(part)
            return ''.join(normalized)
        
        target_normalized = normalize_channel_name(target_channel)
        
        for idx, ch_name in enumerate(channel_names):
            if normalize_channel_name(ch_name) == target_normalized:
                return idx
        
        return None
    
    def apply_filters(self, data):
        """
        Apply the same filters that the spike detector uses.
        
        Args:
            data: Signal data (n_samples, n_channels)
            
        Returns:
            filtered_data: Filtered signal data
        """
        if data is None or self.fs is None:
            return data
        
        print(f"  Applying filters (60Hz notch + 1-70Hz bandpass)...")
        filtered_data = data.copy()
        
        # Apply 60Hz notch filter and harmonics
        for harmonic in [60, 120, 180]:
            if harmonic < self.fs / 2:
                filtered_data = notch_filter(filtered_data, harmonic, self.fs)
        
        # Apply 1-70Hz bandpass filter
        lowcut = 1.0
        highcut = min(70.0, self.fs / 2 - 1)
        filtered_data = bandpass_filter(filtered_data, lowcut, highcut, self.fs, order=4)
        
        return filtered_data
    
    def _create_bipolar_channel(self, data, channel_names, channel_idx):
        """
        Create bipolar montage for a specific channel.
        If channel is 'LA11', creates 'LA11-LA12' bipolar signal.
        
        Args:
            data: Full raw data (n_samples, n_channels)
            channel_names: List of all channel names
            channel_idx: Index of the channel of interest
            
        Returns:
            (bipolar_data, bipolar_name): Tuple of bipolar signal and its name
        """
        import re
        
        channel_name = channel_names[channel_idx]
        
        # Parse channel name to get prefix and number
        match = re.match(r'([A-Za-z]+)(\d+)', channel_name)
        
        if not match:
            # Can't parse, return raw data with same name
            print(f"Could not parse channel name {channel_name}, using raw")
            return data[:, channel_idx:channel_idx+1], channel_name
        
        prefix = match.group(1)
        num = int(match.group(2))
        
        # Look for adjacent channel (num+1)
        next_channel_patterns = [
            f"{prefix}{num+1}",           # e.g., LA12
            f"{prefix}{num+1:02d}",       # e.g., LA12 with 2-digit padding
            f"{prefix}{num+1:03d}",       # e.g., LA012 with 3-digit padding
        ]
        
        next_idx = None
        for pattern in next_channel_patterns:
            if pattern in channel_names:
                next_idx = channel_names.index(pattern)
                break
        
        if next_idx is None:
            # Adjacent channel not found, return raw data
            print(f"Could not find adjacent channel for {channel_name}, using raw")
            return data[:, channel_idx:channel_idx+1], channel_name
        
        # Create bipolar signal: channel1 - channel2
        bipolar_signal = data[:, channel_idx] - data[:, next_idx]
        bipolar_signal = bipolar_signal.reshape(-1, 1)  # Keep 2D
        
        # Create bipolar name
        bipolar_name = f"{channel_names[channel_idx]}-{channel_names[next_idx]}"
        
        print(f"Created bipolar channel: {bipolar_name}")
        return bipolar_signal, bipolar_name
    
    def create_interpolated_data(self, data):
        """
        Create interpolated version of data by detecting and interpolating gaps.
        Simulates the stitched segments from stim-spike_detection-10s.py
        
        Gaps are created at 1-second intervals throughout the entire recording:
        - Pre-stim: -10s to 0s (10 sham stims)
        - During stim: 0s to 30s (30 real stims)
        - Post-stim: 30s to 41s (11 sham stims)
        
        Each gap spans -0.01s to +0.05s around the stim time (0.06s total)
        
        Returns:
            interpolated_data: Data with gaps interpolated
            gap_ranges: List of (start_sample, end_sample) for each gap
        """
        # Gap window: -0.01s to +0.05s around each stim (0.06s total)
        gap_start_offset = -0.01  # seconds before stim
        gap_end_offset = 0.05     # seconds after stim
        
        # We'll mark gap regions and interpolate them
        interpolated_data = data.copy()
        gap_ranges = []
        
        # Pre-stim period is 10s (starts at t=-10s, which is sample 0)
        pre_stim_duration = 10.0
        
        # Generate all stim times (real and sham)
        stim_times = []
        
        # Pre-stim sham stims: at -10, -9, -8, ..., -1 seconds
        stim_times.extend([-10 + i for i in range(10)])
        
        # Real stims during stimulation: at 0, 1, 2, ..., 29 seconds
        stim_times.extend([i for i in range(30)])
        
        # Post-stim sham stims: at 30, 31, 32, ..., 40 seconds
        stim_times.extend([30 + i for i in range(11)])
        
        # Create gaps for all stim times
        for stim_time in stim_times:
            # Convert to absolute sample indices
            # (t=-10 corresponds to sample 0)
            stim_sample = int((stim_time + pre_stim_duration) * self.fs)
            gap_start = int((stim_time + pre_stim_duration + gap_start_offset) * self.fs)
            gap_end = int((stim_time + pre_stim_duration + gap_end_offset) * self.fs)
            
            # Clamp to valid range
            gap_start = max(0, gap_start)
            gap_end = min(data.shape[0], gap_end)
            
            if gap_end <= gap_start:
                continue
            
            gap_ranges.append((gap_start, gap_end))
            
            # Linear interpolation for each channel
            for ch in range(data.shape[1]):
                if gap_start > 0 and gap_end < data.shape[0]:
                    y0 = data[gap_start - 1, ch]
                    y1 = data[gap_end, ch]
                    interpolated = np.linspace(y0, y1, gap_end - gap_start + 2)[1:-1]
                    interpolated_data[gap_start:gap_end, ch] = interpolated
        
        return interpolated_data, gap_ranges
    
    def load_existing_annotations(self, edf_file):
        """Load existing annotations for this file if they exist."""
        annotation_file = os.path.join(
            self.annotations_dir,
            edf_file.stem + '_annotations.json'
        )
        
        if os.path.exists(annotation_file):
            try:
                with open(annotation_file, 'r') as f:
                    data = json.load(f)
                    self.spike_annotations = [
                        tuple(x) for x in data.get('spikes', [])
                    ]
                self.update_spike_count()
            except Exception as e:
                print(f"Could not load existing annotations: {e}")
                self.spike_annotations = []
        else:
            self.spike_annotations = []
        
        self.update_spike_count()
    
    def toggle_interpolation(self):
        """Toggle between interpolated and raw signal."""
        self.show_interpolated = self.interp_checkbox.isChecked()
        self.update_current_data()
    
    def toggle_bipolar(self):
        """Toggle between bipolar and raw signal."""
        self.show_bipolar = self.bipolar_checkbox.isChecked()
        self.update_current_data()
    
    def update_current_data(self):
        """Update current_data based on interpolation and bipolar checkboxes."""
        if self.raw_data_single_channel is None or self.bipolar_data_single_channel is None:
            return
        
        # Select bipolar or raw
        base_data = (self.bipolar_data_single_channel if self.show_bipolar 
                    else self.raw_data_single_channel)
        
        # If showing interpolated, recreate interpolation for the selected montage
        if self.show_interpolated:
            self.interpolated_data, self.gap_ranges = self.create_interpolated_data(base_data)
            self.current_data = self.interpolated_data
        else:
            self.current_data = base_data
        
        # Update channel name
        current_name = (self.bipolar_channel_name if self.show_bipolar 
                       else self.raw_channel_name)
        self.channel_names = np.array([current_name])
        
        # Update file label to show current channel
        if hasattr(self, 'edf_files') and self.edf_files and self.current_file_idx < len(self.edf_files):
            edf_file = self.edf_files[self.current_file_idx]
            self.file_label.setText(f'File: {edf_file.name} | Channel: {current_name}')
        
        self.update_plot()
    
    def increase_gain(self):
        """Increase display gain."""
        self.gain *= 1.2
        self.gain_label.setText(f'{self.gain:.2f}x')
        self.update_plot()
    
    def decrease_gain(self):
        """Decrease display gain."""
        self.gain /= 1.2
        self.gain_label.setText(f'{self.gain:.2f}x')
        self.update_plot()
    
    def on_scroll(self, value):
        """Handle scroll bar movement."""
        self.scroll_position = value / 10.0  # Convert to seconds
        self.update_plot()
    
    def update_plot(self):
        """Update the plot display."""
        if self.current_data is None:
            return
        
        self.plot.clear()
        
        # Determine visible time range
        t_start = self.scroll_position - 10.0  # Relative to stim onset
        t_end = t_start + self.visible_duration
        
        # Find sample indices
        start_idx = int((t_start + 10.0) * self.fs)
        end_idx = int((t_end + 10.0) * self.fs)
        start_idx = max(0, start_idx)
        end_idx = min(len(self.time_vector), end_idx)
        
        if start_idx >= end_idx:
            return
        
        # Get visible data
        visible_time = self.time_vector[start_idx:end_idx]
        visible_data = self.current_data[start_idx:end_idx, :]
        
        # Calculate channel spacing for display
        n_channels = visible_data.shape[1]
        channel_spacing = 100.0 / self.gain  # Adjust spacing based on gain
        
        # Plot each channel with offset
        for ch_idx in range(n_channels):
            y_offset = (n_channels - ch_idx - 1) * channel_spacing
            trace = visible_data[:, ch_idx] * self.gain + y_offset
            
            self.plot.plot(
                visible_time, trace, 
                pen=pg.mkPen('k', width=0.5)
            )
        
        # Mark stimulation times
        # Real stims (red solid lines at 0, 1, 2, ... 29 seconds)
        for stim_time in range(0, 30):
            if t_start <= stim_time <= t_end:
                line = pg.InfiniteLine(
                    pos=stim_time, angle=90, 
                    pen=pg.mkPen('r', width=1, style=Qt.DashLine)
                )
                self.plot.addItem(line)
        
        # Sham stims in pre-period (light blue dotted lines at -10, -9, ..., -1 seconds)
        for stim_time in range(-10, 0):
            if t_start <= stim_time <= t_end:
                line = pg.InfiniteLine(
                    pos=stim_time, angle=90, 
                    pen=pg.mkPen((135, 206, 250), width=1, style=Qt.DotLine)  # Light blue
                )
                self.plot.addItem(line)
        
        # Sham stims in post-period (light blue dotted lines at 30, 31, ..., 40 seconds)
        for stim_time in range(30, 41):
            if t_start <= stim_time <= t_end:
                line = pg.InfiniteLine(
                    pos=stim_time, angle=90, 
                    pen=pg.mkPen((135, 206, 250), width=1, style=Qt.DotLine)  # Light blue
                )
                self.plot.addItem(line)
        
        # Mark annotated spikes
        for ch_idx, spike_time in self.spike_annotations:
            if t_start <= spike_time <= t_end and 0 <= ch_idx < n_channels:
                y_offset = (n_channels - ch_idx - 1) * channel_spacing
                # Find closest time point
                time_idx = np.argmin(np.abs(visible_time - spike_time))
                if time_idx < len(visible_time):
                    spike_y = visible_data[time_idx, ch_idx] * self.gain + y_offset
                    
                    # Draw spike marker
                    spike_marker = pg.ScatterPlotItem(
                        [spike_time], [spike_y],
                        symbol='t', size=15, 
                        pen=pg.mkPen('b', width=2),
                        brush=pg.mkBrush(0, 0, 255, 100)
                    )
                    self.plot.addItem(spike_marker)
        
        # Set axis ranges
        self.plot.setXRange(t_start, t_end)
        self.plot.setYRange(-channel_spacing, n_channels * channel_spacing)
        
        # Set y-axis labels to show channel names
        y_ticks = []
        for ch_idx in range(n_channels):
            y_pos = (n_channels - ch_idx - 1) * channel_spacing
            if ch_idx < len(self.channel_names):
                y_ticks.append((y_pos, self.channel_names[ch_idx]))
        
        ax = self.plot.getAxis('left')
        ax.setTicks([y_ticks])
    
    def on_plot_clicked(self, event):
        """Handle mouse clicks on the plot."""
        if self.current_data is None:
            return
        
        # Get mouse position
        mouse_point = self.plot.vb.mapSceneToView(event.scenePos())
        click_time = mouse_point.x()
        click_y = mouse_point.y()
        
        # Since we only display one channel, clicked_channel is always 0
        clicked_channel = 0
        
        # Get current channel name for display
        channel_name = self.channel_names[0] if len(self.channel_names) > 0 else "CH0"
        
        # Check if clicking near an existing spike (to remove it)
        for i, (ch_idx, spike_time) in enumerate(self.spike_annotations):
            if (ch_idx == clicked_channel and 
                abs(spike_time - click_time) < self.spike_click_threshold):
                # Remove this spike
                self.spike_annotations.pop(i)
                self.update_spike_count()
                self.update_plot()
                self.status_bar.showMessage(
                    f'Removed spike at t={spike_time:.2f}s on {channel_name}'
                )
                return
        
        # Add new spike annotation
        self.spike_annotations.append((clicked_channel, click_time))
        self.update_spike_count()
        self.update_plot()
        self.status_bar.showMessage(
            f'Added spike at t={click_time:.2f}s on {channel_name}'
        )
    
    def update_spike_count(self):
        """Update the spike count label."""
        self.spike_count_label.setText(f'Spikes marked: {len(self.spike_annotations)}')
    
    def save_and_next(self):
        """Save current annotations and move to next file."""
        if not self.edf_files or self.current_file_idx >= len(self.edf_files):
            return
        
        # Save annotations
        edf_file = self.edf_files[self.current_file_idx]
        self.save_annotations(edf_file)
        
        # Mark as completed
        self.completed_files.add(str(edf_file))
        self.save_progress()
        
        # Move to next file
        self.current_file_idx += 1
        
        if self.current_file_idx < len(self.edf_files):
            self.load_current_file()
        else:
            QMessageBox.information(
                self, 'Complete', 
                'All files have been annotated! Great work!'
            )
            self.status_bar.showMessage('All files completed')
    
    def load_previous_file(self):
        """Load the previous file."""
        if self.current_file_idx > 0:
            self.current_file_idx -= 1
            self.load_current_file()
    
    def save_annotations(self, edf_file):
        """Save annotations for the current file."""
        annotation_file = os.path.join(
            self.annotations_dir,
            edf_file.stem + '_annotations.json'
        )
        
        # Prepare annotation data
        annotation_data = {
            'edf_file': str(edf_file),
            'timestamp': datetime.now().isoformat(),
            'patient_id': self.current_patient_id,
            'stim_channel': self.current_stim_channel,
            'ieeg_stim_time_seconds': self.current_ieeg_stim_time,
            'channel_of_interest': self.channel_of_interest,
            'raw_channel_name': self.raw_channel_name,
            'bipolar_channel_name': self.bipolar_channel_name,
            'montage_used': 'bipolar' if self.show_bipolar else 'raw',
            'interpolation_applied': self.show_interpolated,
            'n_channels': len(self.channel_names) if self.channel_names is not None else 0,
            'channel_names': self.channel_names.tolist() if self.channel_names is not None else [],
            'sampling_rate': self.fs,
            'spikes': self.spike_annotations,
            'n_spikes': len(self.spike_annotations)
        }
        
        try:
            with open(annotation_file, 'w') as f:
                json.dump(annotation_data, f, indent=2)
            self.status_bar.showMessage(f'Saved {len(self.spike_annotations)} annotations')
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Failed to save annotations:\n{str(e)}')
    
    def load_progress(self):
        """Load progress tracking data."""
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, 'r') as f:
                    data = json.load(f)
                    return set(data.get('completed_files', []))
            except Exception as e:
                print(f"Could not load progress: {e}")
        return set()
    
    def save_progress(self):
        """Save progress tracking data."""
        try:
            with open(self.progress_file, 'w') as f:
                json.dump({
                    'completed_files': list(self.completed_files),
                    'last_updated': datetime.now().isoformat()
                }, f, indent=2)
        except Exception as e:
            print(f"Could not save progress: {e}")
    
    def update_progress_label(self):
        """Update the progress label."""
        if self.edf_files:
            n_completed = len(self.completed_files)
            n_total = len(self.edf_files)
            self.progress_label.setText(f'Progress: {n_completed}/{n_total}')


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Spike Annotation GUI')
    parser.add_argument(
        '--edf_dir', type=str, 
        help='Directory containing EDF files to annotate'
    )
    
    args = parser.parse_args()
    
    app = QApplication(sys.argv)
    gui = SpikeAnnotationGUI(edf_dir=args.edf_dir)
    gui.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()

