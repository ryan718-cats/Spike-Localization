# iEEG imports
# !pip install git+https://github.com/ieeg-portal/ieegpy.git # Install ieegpy toolbox directly from github
# !pip install pyedflib
from ieeg.auth import Session
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal as sig
import os
import sys
import numpy as np
import scipy.io as sio

# iEEG imports
# !pip install git+https://github.com/ieeg-portal/ieegpy.git # Install ieegpy toolbox directly from github
# !pip install pyedflib
import os
import numpy as np
import pandas as pd
import pyedflib
from ieeg.auth import Session
import warnings
import sys

def get_iEEG_data(
    username,
    password_bin_file,
    iEEG_filename,
    start_time_usec,
    stop_time_usec,
    select_electrodes=None,
):

    start_time_usec = int(start_time_usec)
    stop_time_usec = int(stop_time_usec)
    duration = stop_time_usec - start_time_usec
    with open(password_bin_file, "r") as f:
        s = Session(username, f.read())
    ds = s.open_dataset(iEEG_filename)
    all_channel_labels = ds.get_channel_labels()

    # Map selected electrode names to their corresponding indices
    if select_electrodes is not None:
        if isinstance(select_electrodes[0], str):
            # Find indices of the selected electrode names in all_channel_labels
            channel_ids = [
                i for i, e in enumerate(all_channel_labels) if e in select_electrodes
            ]
            channel_names = select_electrodes
        else:
            print("Electrodes must be given as a list of strings")
            return None, None  # Return empty if the input format is incorrect

    try:
        data = ds.get_data(start_time_usec, duration, channel_ids)
    except:
        # Clip is probably too big, pull chunks and concatenate
        clip_size = 60 * 1e6
        clip_start = start_time_usec
        data = None
        while clip_start + clip_size < stop_time_usec:
            if data is None:
                data = ds.get_data(clip_start, clip_size, channel_ids)
            else:
                data = np.concatenate(
                    ([data, ds.get_data(clip_start, clip_size, channel_ids)]), axis=0
                )
            clip_start = clip_start + clip_size
        data = np.concatenate(
            ([data, ds.get_data(clip_start, stop_time_usec - clip_start, channel_ids)]),
            axis=0,
        )

    df = pd.DataFrame(data, columns=channel_names)
    fs = ds.get_time_series_details(ds.ch_labels[0]).sample_rate  # get sample rate
    return df, fs

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Define core EEG channels
core_eeg_channels = ['C3', 'C4', 'Cz', 'F3', 'F4', 'F7', 'F8', 'Fp1', 'Fp2', 'Fz', 'O1', 'O2', 'P3', 'P4', 'Pz', 'T3', 'T4', 'T5', 'T6']

# Function to load data from each electrode and ensure correct order of channels
def load_electrode_data(username, password_bin_file, iEEG_filename, start_time, end_time, electrodes):
    combined_df = pd.DataFrame()  # Empty DataFrame to store all electrode data
    electrode_labels = []  # To store the labels of loaded electrodes

    # Open the session and dataset to get all available channel labels
    with open(password_bin_file, "r") as f:
        s = Session(username, f.read())
    ds = s.open_dataset(iEEG_filename)
    all_channel_labels = ds.get_channel_labels()

    # Loop through the electrodes and load data for each
    for electrode in electrodes:
        df, fs = get_iEEG_data(username, password_bin_file, iEEG_filename, start_time * 1e6, end_time * 1e6, [electrode])
        if df is not None:
            combined_df = pd.concat([combined_df, df], axis=1)
            electrode_labels.append(electrode)
        else:
            print(f"Warning: Data for electrode {electrode} not found.")

    return combined_df, electrode_labels, fs

# Class to suppress print statements
class SuppressPrint:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout.close()
        sys.stdout = self._original_stdout

# Function to process and save an EDF file for a given time segment
def process_and_save_edf(ieeg_file_name, start_time, end_time, single_lat):
    try:
        # Define the save directory based on single_lat value
        lat_dir = 'left' if single_lat == 'left' else 'right'
        edf_save_directory = f'/mnt/sauce/littlab/users/jurikim/spikenet/Persyst/edf_flipped_Nov1224/{lat_dir}/'

        # Ensure the save directory exists
        if not os.path.exists(edf_save_directory):
            os.makedirs(edf_save_directory)

        # Use the load_electrode_data function to get the data for the electrodes
        electrodes = core_eeg_channels
        combined_df, electrode_labels, fs = load_electrode_data('jurikim', 'jurikim_ieeglogin.bin', ieeg_file_name, start_time, end_time, electrodes)

        # Skip if there are NaN values in the combined_df DataFrame
        if combined_df.isnull().values.any():
            print(f"Skipping {ieeg_file_name} due to NaN values.")
            return

        # Ensure the combined DataFrame is not empty
        if combined_df.size > 0:
            print(f"Loaded data segment shape: {combined_df.shape}")
            print(f"Sampling rate (fs): {fs} Hz")

            # Ensure the extracted segment has the expected duration
            expected_samples = int((end_time - start_time) * fs)
            if combined_df.shape[0] != expected_samples:
                print(f"Warning: Extracted segment length {combined_df.shape[0]} does not match expected {expected_samples} samples.")
                return  # Skip this row if there's a mismatch

            # Create the EDF file name and path
            edf_file_name = f"{ieeg_file_name}_{start_time}_to_{end_time}.edf"
            flipped_segment = combined_df * -1
            edf_file_path = os.path.join(edf_save_directory, edf_file_name)

            # Save the EDF file
            with pyedflib.EdfWriter(edf_file_path, len(electrode_labels), file_type=pyedflib.FILETYPE_EDFPLUS) as f:
                channel_info = [{'label': ch, 'dimension': 'uV', 'sample_rate': fs, 'physical_min': np.min(flipped_segment.values),
                                 'physical_max': np.max(flipped_segment.values), 'digital_min': -32768, 'digital_max': 32767} for ch in electrode_labels]
                f.setSignalHeaders(channel_info)
                f.writeSamples(flipped_segment.values.T)

            print(f"Saved EDF file: {edf_file_path}")

    except Exception as e:
        print(f"Error processing file {ieeg_file_name}: {e}")

# Load the CSV containing start and end times
data_directory = '/mnt/sauce/littlab/users/jurikim/spikenet/ied_yesno/'
csv_file_path = os.path.join(data_directory, 'ied_yesno_50_Nov1224.csv')
df = pd.read_csv(csv_file_path)

# Identify the starting index
start_index = df[(df['ieeg_file_name'] == 'EMU1696_Day04_1') & 
                 (df['start_sec'] == 1187.5) & 
                 (df['end_sec'] == 1414.0625)].index[0]

# Loop over each row starting from the specified row
for _, row in df.iloc[start_index:].iterrows():
    ieeg_file_name = row['ieeg_file_name']
    start_time = row['start_sec']
    end_time = row['end_sec']
    single_lat = row['single_lat']  # Get the lateralization ('left' or 'right')

    with SuppressPrint():
        process_and_save_edf(ieeg_file_name, start_time, end_time, single_lat)


# # Load the CSV containing start and end times
# data_directory = '/mnt/sauce/littlab/users/jurikim/spikenet/ied_yesno/'
# csv_file_path = os.path.join(data_directory, 'ied_yesno_50_Nov1224.csv')
# df = pd.read_csv(csv_file_path)

# # Get the unique ieeg_file_names
# unique_ieeg_file_names = df['ieeg_file_name'].unique()
# num_unique_files = len(unique_ieeg_file_names)
# print(f"Number of unique ieeg_file_name entries: {num_unique_files}")

# # Loop over each unique ieeg_file_name and process each segment
# for ieeg_file_name in unique_ieeg_file_names:
#     # Filter rows for the current ieeg_file_name
#     filtered_rows = df[df['ieeg_file_name'] == ieeg_file_name]

#     # Process and save EDF files for each time segment
#     for _, row in filtered_rows.iterrows():
#         start_time = row['start_sec']
#         end_time = row['end_sec']
#         single_lat = row['single_lat']  # Get the lateralization ('left' or 'right')

#         with SuppressPrint():
#             process_and_save_edf(ieeg_file_name, start_time, end_time, single_lat)


import os

# Directory path to check
edf_save_directory = '/mnt/sauce/littlab/users/jurikim/spikenet/Persyst/edf_flipped_Nov1224/right/'

# Count the number of EDF files in the directory
edf_file_count = len([file for file in os.listdir(edf_save_directory) if file.endswith('.edf')])

# Calculate the total size of all EDF files in the directory
total_size = sum(
    os.path.getsize(os.path.join(edf_save_directory, file))
    for file in os.listdir(edf_save_directory)
    if file.endswith('.edf')
)

# Convert total size to gigabytes for easier readability
total_size_gb = total_size / (1024 * 1024 * 1024)

print(f"There are {edf_file_count} EDF files in the directory.")
print(f"Total data size in directory: {total_size_gb:.2f} GB")


import os
import numpy as np
import pyedflib

edf_save_directory = '/mnt/sauce/littlab/users/jurikim/spikenet/Persyst/edf_flipped_Nov1224/left/' #right

# Function to check for NaN values in an EDF file
def check_nan_in_edf(edf_file):
    try:
        # Open the EDF file
        f = pyedflib.EdfReader(edf_file)
        n_channels = f.signals_in_file
        
        for ch in range(n_channels):
            signal = f.readSignal(ch)
            if np.isnan(signal).any():
                print(f"NaN values found in channel {ch} of file {edf_file}")
                return True
        
        f.close()
    except Exception as e:
        print(f"Error processing {edf_file}: {e}")
        return False
    
    print(f"No NaN values in {edf_file}")
    return False

# Loop through all EDF files in the directory and check for NaNs
for file_name in os.listdir(edf_save_directory):
    if file_name.endswith(".edf"):  # Check if the file is an EDF file
        edf_file_path = os.path.join(edf_save_directory, file_name)
        check_nan_in_edf(edf_file_path)
