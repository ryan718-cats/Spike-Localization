#!/usr/bin/env python3
"""
Analyze Spike Annotations

This script analyzes the manual spike annotations created with the GUI,
comparing them with automated spike detection results.

Usage:
    python analyze_annotations.py --annotations_dir ./annotations
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_all_annotations(annotations_dir):
    """Load all annotation files from directory."""
    annotations = []
    
    annotation_files = list(Path(annotations_dir).glob('*_annotations.json'))
    
    for ann_file in annotation_files:
        try:
            with open(ann_file, 'r') as f:
                data = json.load(f)
                annotations.append({
                    'file': ann_file.stem,
                    'edf_file': data.get('edf_file', ''),
                    'n_spikes': data.get('n_spikes', 0),
                    'n_channels': data.get('n_channels', 0),
                    'spikes': data.get('spikes', []),
                    'timestamp': data.get('timestamp', '')
                })
        except Exception as e:
            print(f"Error loading {ann_file}: {e}")
    
    return pd.DataFrame(annotations)


def plot_annotation_statistics(df, output_dir):
    """Create visualization of annotation statistics."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    
    # Figure 1: Distribution of spike counts
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Spike count distribution
    axes[0, 0].hist(df['n_spikes'], bins=30, edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel('Number of Spikes per File')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of Spike Counts')
    axes[0, 0].axvline(df['n_spikes'].median(), color='red', linestyle='--',
                       label=f'Median: {df["n_spikes"].median():.0f}')
    axes[0, 0].legend()
    
    # Channel count distribution
    axes[0, 1].hist(df['n_channels'], bins=20, edgecolor='black', alpha=0.7, color='green')
    axes[0, 1].set_xlabel('Number of Channels per File')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Distribution of Channel Counts')
    
    # Spikes per channel
    df['spikes_per_channel'] = df['n_spikes'] / df['n_channels']
    axes[1, 0].hist(df['spikes_per_channel'], bins=30, edgecolor='black', alpha=0.7, color='orange')
    axes[1, 0].set_xlabel('Spikes per Channel')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Distribution of Spikes per Channel')
    
    # Cumulative spikes
    df_sorted = df.sort_values('n_spikes', ascending=False).reset_index(drop=True)
    cumulative_spikes = df_sorted['n_spikes'].cumsum()
    axes[1, 1].plot(range(len(df_sorted)), cumulative_spikes, linewidth=2)
    axes[1, 1].set_xlabel('Number of Files')
    axes[1, 1].set_ylabel('Cumulative Spike Count')
    axes[1, 1].set_title('Cumulative Spikes Across Files')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'annotation_statistics.pdf'), dpi=300)
    plt.savefig(os.path.join(output_dir, 'annotation_statistics.png'), dpi=300)
    print(f"Saved: annotation_statistics.pdf/png")
    plt.close()
    
    # Figure 2: Temporal distribution of spikes
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    
    all_spike_times = []
    for spikes in df['spikes']:
        for ch_idx, spike_time in spikes:
            all_spike_times.append(spike_time - 10.0)  # Relative to stim onset
    
    if all_spike_times:
        ax.hist(all_spike_times, bins=100, edgecolor='black', alpha=0.7)
        ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Stimulation Onset')
        ax.axvline(30, color='red', linestyle='--', linewidth=2, label='Stimulation End')
        ax.set_xlabel('Time Relative to Stimulation (s)')
        ax.set_ylabel('Number of Spikes')
        ax.set_title('Temporal Distribution of All Annotated Spikes')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'temporal_distribution.pdf'), dpi=300)
        plt.savefig(os.path.join(output_dir, 'temporal_distribution.png'), dpi=300)
        print(f"Saved: temporal_distribution.pdf/png")
        plt.close()


def print_summary_statistics(df):
    """Print summary statistics."""
    print("\n" + "="*80)
    print("ANNOTATION SUMMARY STATISTICS")
    print("="*80)
    
    print(f"\nTotal annotated files: {len(df)}")
    print(f"Total spikes annotated: {df['n_spikes'].sum()}")
    
    print(f"\nSpike count statistics:")
    print(f"  Mean: {df['n_spikes'].mean():.2f}")
    print(f"  Median: {df['n_spikes'].median():.2f}")
    print(f"  Std: {df['n_spikes'].std():.2f}")
    print(f"  Min: {df['n_spikes'].min()}")
    print(f"  Max: {df['n_spikes'].max()}")
    
    print(f"\nChannel count statistics:")
    print(f"  Mean: {df['n_channels'].mean():.2f}")
    print(f"  Median: {df['n_channels'].median():.2f}")
    print(f"  Min: {df['n_channels'].min()}")
    print(f"  Max: {df['n_channels'].max()}")
    
    print(f"\nSpikes per channel:")
    print(f"  Mean: {df['spikes_per_channel'].mean():.2f}")
    print(f"  Median: {df['spikes_per_channel'].median():.2f}")
    
    # Files with most/least spikes
    print(f"\nTop 5 files by spike count:")
    top_files = df.nlargest(5, 'n_spikes')[['file', 'n_spikes', 'n_channels']]
    for idx, row in top_files.iterrows():
        print(f"  {row['file']}: {row['n_spikes']} spikes ({row['n_channels']} channels)")
    
    print(f"\nBottom 5 files by spike count:")
    bottom_files = df.nsmallest(5, 'n_spikes')[['file', 'n_spikes', 'n_channels']]
    for idx, row in bottom_files.iterrows():
        print(f"  {row['file']}: {row['n_spikes']} spikes ({row['n_channels']} channels)")
    
    # Temporal distribution
    all_spike_times = []
    for spikes in df['spikes']:
        for ch_idx, spike_time in spikes:
            all_spike_times.append(spike_time - 10.0)  # Relative to stim onset
    
    if all_spike_times:
        spike_times_array = np.array(all_spike_times)
        pre_stim_spikes = np.sum(spike_times_array < 0)
        during_stim_spikes = np.sum((spike_times_array >= 0) & (spike_times_array < 30))
        post_stim_spikes = np.sum(spike_times_array >= 30)
        
        print(f"\nTemporal distribution:")
        print(f"  Pre-stimulation (-10 to 0s): {pre_stim_spikes} spikes "
              f"({100*pre_stim_spikes/len(spike_times_array):.1f}%)")
        print(f"  During stimulation (0 to 30s): {during_stim_spikes} spikes "
              f"({100*during_stim_spikes/len(spike_times_array):.1f}%)")
        print(f"  Post-stimulation (30 to 41s): {post_stim_spikes} spikes "
              f"({100*post_stim_spikes/len(spike_times_array):.1f}%)")


def export_summary_csv(df, output_file):
    """Export summary to CSV."""
    summary = df[['file', 'n_spikes', 'n_channels', 'spikes_per_channel', 'timestamp']].copy()
    summary.to_csv(output_file, index=False)
    print(f"\nExported summary to: {output_file}")


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Analyze spike annotations'
    )
    parser.add_argument(
        '--annotations_dir', type=str,
        default='./annotations',
        help='Directory containing annotation JSON files'
    )
    parser.add_argument(
        '--output_dir', type=str,
        default='./annotation_analysis',
        help='Output directory for analysis results'
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("SPIKE ANNOTATION ANALYSIS")
    print("="*80)
    
    # Load annotations
    print(f"\nLoading annotations from: {args.annotations_dir}")
    df = load_all_annotations(args.annotations_dir)
    
    if len(df) == 0:
        print("No annotation files found!")
        return
    
    print(f"Loaded {len(df)} annotation files")
    
    # Print summary statistics
    print_summary_statistics(df)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate plots
    print(f"\nGenerating visualizations...")
    plot_annotation_statistics(df, args.output_dir)
    
    # Export summary CSV
    summary_file = os.path.join(args.output_dir, 'annotation_summary.csv')
    export_summary_csv(df, summary_file)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

