#!/usr/bin/env python3
"""
Create EDF Files for Spike Annotation (Optimized Dataset Version)

This script creates EDF files for stim-recording channel pairs from the 
pre_post_optimized_data_filt.csv dataset that have non-zero spike rates.

Key differences from original script:
- Uses pre_post_optimized_data_filt.csv (with bipolar channels already applied)
- Filters for pairs with non-zero spike rates in pre, during, or post periods
- Only creates EDFs for patients NOT in CONFIG.good_pts
- Recording channels are already bipolar (e.g., "LA01-LA02"), loads constituent channels
- Samples from bottom 80% by total spike activity to avoid very high spike rate pairs

Each EDF file contains:
- 10 seconds pre-stimulation
- 30 seconds during stimulation (with 1Hz pulses)
- 11 seconds post-stimulation
Total: 51 seconds

RUN:
python spike-gui/create_annotation_edfs_optimized.py --n_files 200 --output_dir /users/aguilac/stim-responses/results/spike_annotations_edfs_2

The data is saved as RAW (unfiltered) with both constituent channels (Ch1 and Ch2).
The spike_annotation_gui.py will handle filtering and bipolar re-referencing, allowing
annotators to toggle between raw and bipolar views.
"""

import sys
import os
from os.path import join as ospj
import numpy as np
import pandas as pd
import mne
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'code')))
from config import CONFIG
sys.path.append(CONFIG.tools_dir)
from iEEG_helper_functions import *
from get_iEEG_data2 import *
from ieeg.auth import Session


def identify_pairs_with_spikes(dataset_file):
    """
    Identify stim-recording pairs that have non-zero spike rates.
    
    Parameters:
    dataset_file: Path to pre_post_optimized_data_filt.csv
    
    Returns:
    DataFrame with pairs that have spikes
    """
    print("Loading dataset...")
    df = pd.read_csv(dataset_file)
    
    print(f"Total rows in dataset: {len(df)}")
    
    # Filter for patients NOT in CONFIG.good_pts
    print(f"\nFiltering out CONFIG.good_pts: {CONFIG.good_pts}")
    df = df[~df['hup_id'].isin(CONFIG.good_pts)]
    print(f"Rows after excluding good_pts: {len(df)}")
    
    # Filter for pairs with non-zero spike rates in any condition
    pairs_with_spikes = df[df['change_pre_to_during'] != 0].copy()
    
    print(f"\nFound {len(pairs_with_spikes)} pairs with spike detections")
    
    # Filter out white matter and outside brain
    if 'roi_rec_ch' in pairs_with_spikes.columns:
        pairs_with_spikes = pairs_with_spikes[
            ~pairs_with_spikes['roi_rec_ch'].isin(['white-matter', 'outside-brain'])
        ]
        pairs_with_spikes = pairs_with_spikes.dropna(subset=['roi_rec_ch'])
        print(f"After filtering ROI: {len(pairs_with_spikes)} pairs")
    
    # Filter for pairs with non-na change_pre_to_during
    if 'change_pre_to_during' in pairs_with_spikes.columns:
        pairs_with_spikes = pairs_with_spikes.dropna(subset=['change_pre_to_during'])
        print(f"After filtering for valid change_pre_to_during: {len(pairs_with_spikes)} pairs")
    
    return pairs_with_spikes


def load_patient_metadata():
    """Load patient metadata including iEEG filenames."""
    metadata = pd.read_excel(
        ospj(CONFIG.data_dir, 'pt-metadata', 'master_pt_list_erin.xlsx')
    )
    metadata['HUPID'] = metadata['HUPID'].str.replace('HUP', '').astype(int)
    metadata = metadata[['HUPID', 'ieeg_filename']]
    return metadata


def load_stim_times():
    """Load first stimulation times per channel."""
    stim_times = pd.read_csv(
        ospj(CONFIG.data_dir, 'pt-metadata', 'first_stim_times_per_channel.csv')
    )
    stim_times['hup_id'] = stim_times['hup_id'].str.replace('HUP', '', regex=False).astype(int)
    
    def extract_first_stim_time(value):
        if pd.isna(value):
            return None
        str_val = str(value).strip('[]')
        try:
            return float(str_val)
        except:
            return None
    
    stim_times['first_stim_time'] = stim_times['first_stim_time'].apply(
        extract_first_stim_time
    )
    stim_times = stim_times.dropna(subset=['first_stim_time'])
    
    return stim_times


def get_ieeg_filename(patient_id, metadata):
    """Get iEEG filename(s) for patient."""
    patient_meta = metadata[metadata['HUPID'] == patient_id]
    if len(patient_meta) == 0:
        return None
    
    filename = patient_meta['ieeg_filename'].iloc[0]
    
    # Handle multiple filenames
    if isinstance(filename, str):
        if ',' in filename:
            if filename.startswith('[') and filename.endswith(']'):
                filename = filename.strip('[]')
            filename_list = [f.strip().strip("'\"") for f in filename.split(',')]
            filename_list = [f for f in filename_list if f]
            return filename_list
    
    return filename


def parse_bipolar_channel(bipolar_name):
    """
    Parse a bipolar channel name to get the two constituent channels.
    E.g., 'LA01-LA02' -> ['LA01', 'LA02']
    """
    if '-' not in bipolar_name:
        return None
    parts = bipolar_name.split('-')
    if len(parts) != 2:
        return None
    return parts


def try_load_ieeg_data(username, password_file, ieeg_filename, start_usec, end_usec, all_channels=True):
    """Try to load iEEG data from a specific filename."""
    try:
        ieeg_data, fs = get_iEEG_data2(
            username, password_file, ieeg_filename,
            start_usec, end_usec, all_channels=all_channels
        )
        return ieeg_data, fs
    except Exception as e:
        print(f"    Failed to load from {ieeg_filename}: {str(e)}")
        return None, None


def get_existing_pairs(output_dir):
    """
    Get set of existing pairs that already have EDF files.
    
    Parameters:
    output_dir: Directory containing EDF files
    
    Returns:
    Set of tuples (patient_id, stim_ch, recording_ch)
    """
    existing_pairs = set()
    
    if not os.path.exists(output_dir):
        return existing_pairs
    
    for filename in os.listdir(output_dir):
        if not filename.endswith('.edf'):
            continue
        
        # Parse filename: HUP{patient_id}_{stim_ch}_{recording_ch}.edf
        try:
            # Remove .edf extension
            name_parts = filename[:-4]
            
            # Split by underscore, but need to handle channel names with hyphens
            # Format: HUP123_STIM_CH_REC_CH
            if name_parts.startswith('HUP'):
                # Remove HUP prefix
                name_parts = name_parts[3:]
                
                # Split on underscore to get patient_id and channels
                parts = name_parts.split('_')
                
                if len(parts) >= 3:
                    patient_id = int(parts[0])
                    # Reconstruct stim_ch and recording_ch
                    # Last part is recording channel, everything in between is stim channel
                    recording_ch = parts[-1]
                    stim_ch = '_'.join(parts[1:-1])
                    
                    existing_pairs.add((patient_id, stim_ch, recording_ch))
        except Exception as e:
            print(f"  Warning: Could not parse filename {filename}: {e}")
            continue
    
    return existing_pairs


def create_edf_for_pair(pair_info, metadata, stim_times, ieeg_session, output_dir, 
                       pair_idx, total_pairs):
    """
    Create an EDF file for a single stim-recording pair.
    
    Returns:
    success: Boolean indicating if EDF was created successfully
    """
    patient_id = pair_info['hup_id']
    stim_ch = pair_info['stim_ch']
    recording_ch = pair_info['recording_ch']  # Already in bipolar format: "LA01-LA02"
    
    print(f"\n[{pair_idx+1}/{total_pairs}] Processing HUP{patient_id}: {stim_ch} → {recording_ch}")
    print(f"  Spike rates - Pre: {pair_info.get('spike_rate_pre', 0):.2f}, During: {pair_info.get('spike_rate_during', 0):.2f}, Post: {pair_info.get('spike_rate_post', 0):.2f}")
    
    # Parse the bipolar channel name to get constituent channels
    constituent_channels = parse_bipolar_channel(recording_ch)
    if constituent_channels is None:
        print(f"  ✗ Could not parse bipolar channel name: {recording_ch}")
        return False
    
    ch1, ch2 = constituent_channels
    print(f"  Bipolar channel: {ch1} - {ch2}")
    
    # Get stimulation time
    stim_info = stim_times[
        (stim_times['hup_id'] == patient_id) & 
        (stim_times['stim_ch'] == stim_ch)
    ]
    
    if len(stim_info) == 0:
        print(f"  ✗ No stimulation time found")
        return False
    
    stim_time = stim_info.iloc[0]['first_stim_time']
    
    # Get iEEG filename
    ieeg_filename = get_ieeg_filename(patient_id, metadata)
    if ieeg_filename is None:
        print(f"  ✗ No iEEG filename found")
        return False
    
    # Calculate time window: 10s before to 41s after stim
    start_time = stim_time - 10.0
    end_time = stim_time + 41.0  # 30s stim + 11s post
    
    start_time_usec = int(start_time * 1e6)
    end_time_usec = int(end_time * 1e6)
    
    # Handle multiple filenames
    filenames_to_try = [ieeg_filename] if isinstance(ieeg_filename, str) else ieeg_filename
    
    ieeg_data = None
    fs = None
    
    # Try each filename
    password_bin_filepath = ospj(CONFIG.tools_dir, 'agu_ieeglogin.bin')
    
    for filename in filenames_to_try:
        try:
            # Open dataset to get channel labels
            dataset = ieeg_session.open_dataset(filename)
            all_channel_labels = np.array(dataset.get_channel_labels())
            
            # Check if both constituent channels exist
            if ch1 not in all_channel_labels or ch2 not in all_channel_labels:
                print(f"    Channels {ch1} or {ch2} not found in {filename}")
                continue
            
            # Load ALL channels
            print(f"  Loading all channels from {filename}...")
            ieeg_data, fs = try_load_ieeg_data(
                "aguilac", password_bin_filepath, filename,
                start_time_usec, end_time_usec, all_channels=True
            )
            
            if ieeg_data is None or ieeg_data.empty:
                print(f"    Failed to load data from {filename}")
                continue
            
            # Filter to just the two constituent channels
            channels_to_keep = [ch1, ch2]
            channels_available = [ch for ch in channels_to_keep if ch in ieeg_data.columns]
            
            if len(channels_available) != 2:
                print(f"    Could not find both channels in data")
                continue
            
            ieeg_data = ieeg_data[channels_available]
            
            if not ieeg_data.empty:
                print(f"  ✓ Loaded from {filename}")
                break
        
        except Exception as e:
            print(f"    Error with {filename}: {str(e)}")
            continue
    
    if ieeg_data is None or ieeg_data.empty:
        print(f"  ✗ Failed to load iEEG data")
        return False
    
    # Save both raw channels WITHOUT preprocessing
    # This allows the GUI to apply filtering and bipolar re-referencing
    print(f"  Saving raw channels: {ch1}, {ch2}...")
    raw_data = ieeg_data.values.copy()
    
    # Create MNE RawArray with both constituent channels
    info = mne.create_info(
        ch_names=[ch1, ch2],
        sfreq=fs,
        ch_types=['seeg', 'seeg']
    )
    
    # Convert to microvolts for EDF format
    raw_data_uv = raw_data / 1e6
    raw = mne.io.RawArray(raw_data_uv.T, info)
    
    # Generate output filename
    output_filename = f"HUP{patient_id}_{stim_ch}_{recording_ch}.edf"
    # Clean filename (remove special characters)
    output_filename = output_filename.replace('/', '-').replace('\\', '-')
    output_path = ospj(output_dir, output_filename)
    
    # Save as EDF
    print(f"  Saving to {output_filename}...")
    try:
        raw.export(output_path, overwrite=False)
        print(f"  ✓ Successfully created EDF file")
        return True
    except Exception as e:
        print(f"  ✗ Failed to save EDF: {str(e)}")
        return False


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Create EDF files for spike annotation from optimized dataset'
    )
    parser.add_argument(
        '--n_files', type=int, default=200,
        help='Number of EDF files to create (default: 200)'
    )
    parser.add_argument(
        '--output_dir', type=str,
        default=ospj(CONFIG.results_dir, 'spike_annotation_edfs_optimized'),
        help='Output directory for EDF files'
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("CREATE EDF FILES FOR SPIKE ANNOTATION (OPTIMIZED DATASET)")
    print("="*80)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")
    
    # Load dataset
    dataset_file = ospj(CONFIG.dataset_dir, "pre_post_optimized_data.csv")
    
    # Identify pairs with spikes
    pairs_with_spikes = identify_pairs_with_spikes(dataset_file)
    
    print(f"\nTotal pairs with spikes after filtering: {len(pairs_with_spikes)}")
    
    # Check for existing EDF files and filter them out
    print(f"\nChecking for existing EDF files in {args.output_dir}...")
    existing_pairs = get_existing_pairs(args.output_dir)
    print(f"Found {len(existing_pairs)} existing EDF files")
    
    if len(existing_pairs) > 0:
        # Filter out pairs that already have EDFs
        pairs_with_spikes['pair_tuple'] = list(zip(
            pairs_with_spikes['hup_id'],
            pairs_with_spikes['stim_ch'],
            pairs_with_spikes['recording_ch']
        ))
        pairs_with_spikes = pairs_with_spikes[
            ~pairs_with_spikes['pair_tuple'].isin(existing_pairs)
        ]
        pairs_with_spikes = pairs_with_spikes.drop(columns=['pair_tuple'])
        print(f"Filtered out existing pairs, {len(pairs_with_spikes)} new pairs remaining")
    
    if len(pairs_with_spikes) == 0:
        print("\n✓ All requested pairs already have EDF files!")
        print(f"Output directory: {args.output_dir}")
        return
    
    # Limit to requested number
    if len(pairs_with_spikes) > args.n_files:
        print(f"\nRandomly sampling {args.n_files} pairs from bottom 80% by spike activity...")
        # Sort by total spike activity
        pairs_with_spikes['total_spike_activity'] = (
            pairs_with_spikes['spike_rate_pre'].fillna(0) + 
            pairs_with_spikes['spike_rate_during'].fillna(0) +
            pairs_with_spikes['spike_rate_post'].fillna(0)
        )
        pairs_with_spikes = pairs_with_spikes.sort_values(
            'total_spike_activity', ascending=False
        )
        
        # Skip top 20% and get the rest
        n_top_20 = int(len(pairs_with_spikes) * 0.2)
        remaining_80 = pairs_with_spikes.iloc[n_top_20:]
        
        # Randomly sample n_files from remaining 80% (without replacement to avoid duplicates)
        if len(remaining_80) >= args.n_files:
            pairs_with_spikes = remaining_80.sample(n=args.n_files, random_state=42)
        else:
            # If remaining 80% has fewer than n_files, just use all of them
            pairs_with_spikes = remaining_80
    
    print(f"\nWill create {len(pairs_with_spikes)} EDF files")
    
    # Load metadata
    print("\nLoading metadata...")
    metadata = load_patient_metadata()
    stim_times = load_stim_times()
    
    # Initialize iEEG session
    print("Initializing iEEG session...")
    password_bin_filepath = ospj(CONFIG.tools_dir, 'agu_ieeglogin.bin')
    with open(password_bin_filepath, "r") as f:
        ieeg_session = Session("aguilac", f.read())
    
    # Create EDF files
    print("\n" + "="*80)
    print("CREATING EDF FILES")
    print("="*80)
    
    success_count = 0
    fail_count = 0
    
    for idx, (_, pair_info) in enumerate(pairs_with_spikes.iterrows()):
        success = create_edf_for_pair(
            pair_info, metadata, stim_times, ieeg_session,
            args.output_dir, idx, len(pairs_with_spikes)
        )
        
        if success:
            success_count += 1
        else:
            fail_count += 1
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Successfully created: {success_count} EDF files")
    print(f"Failed: {fail_count} files")
    print(f"Output directory: {args.output_dir}")
    print("\nYou can now run the annotation GUI:")
    print(f"  python spike_annotation_gui.py --edf_dir {args.output_dir}")


if __name__ == "__main__":
    main()

