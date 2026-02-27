import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import seaborn as sns
import os
import re
import pandas as pd 

#VOLTAGE THRESHOLD ANALYSIS FUNCTIONS
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


def parse_filename(filename):
    """Extract day, recording, and well from filename like d83_r006_wD4_thresholds.csv"""
    basename = os.path.basename(filename)
    match = re.match(r'd(\d+)_r(\d+)_w([A-Z]\d+)_thresholds\.csv', basename)
    if match:
        return {
            'day': int(match.group(1)),
            'recording': int(match.group(2)),
            'well': match.group(3)
        }
    return {'day': None, 'recording': None, 'well': None}

def changes_btwn_recordings(well_data, well, channel_cols, output_path):
    changes = []
    
    for i in range(len(well_data) - 1):
        current = well_data.iloc[i][channel_cols].astype(float)
        next_rec = well_data.iloc[i + 1][channel_cols].astype(float)
        
        diff = next_rec - current
        # Fix: avoid division by zero and handle small values
        pct_change = np.where(current != 0, (diff / np.abs(current)) * 100, 0)
        
        current_tp = f"d{well_data.iloc[i]['day']}_r{well_data.iloc[i]['recording']}"
        next_tp = f"d{well_data.iloc[i+1]['day']}_r{well_data.iloc[i+1]['recording']}"

        changes.append({
            'from': current_tp,
            'to': next_tp,
            'mean_abs_change': np.abs(diff).mean(),
            'max_abs_change': np.abs(diff).max(),
            'mean_pct_change': np.abs(pct_change).mean(),
            'channels_increased': (diff > 0).sum(),
            'channels_decreased': (diff < 0).sum(),
            'channels_unchanged': (diff == 0).sum()
        })
        
        # print(f"\n{current_tp} -> {next_tp}:")
        # print(f"  Mean absolute change: {np.abs(diff).mean():.4f}")
        # print(f"  Max absolute change: {np.abs(diff).max():.4f}")
        # print(f"  Mean % change: {np.abs(pct_change).mean():.2f}%")
        # print(f"  Channels increased: {(diff > 0).sum()}")
        # print(f"  Channels decreased: {(diff < 0).sum()}")
        
        # Per-channel changes
        # print(f"  Per-channel changes:")
        # for idx, ch in enumerate(channel_cols):
        #     print(f"    Ch {ch}: {diff[idx]:+.4f} ({pct_change[idx]:+.2f}%)")
    
    # Save summary to CSV
    changes_df = pd.DataFrame(changes)
    changes_df.to_csv(f'{output_path}/{well}_changes.csv', index=False)
    print(f"\nSaved changes summary to {output_path}/{well}_changes.csv")

    return changes

def get_well_stats(well_data, well, channel_cols):
    # Calculate overall statistics for this well
    all_thresholds = well_data[channel_cols].astype(float).values.flatten()
    
    # Calculate changes across all timepoints
    total_changes = []
    for i in range(len(well_data) - 1):
        current = well_data.iloc[i][channel_cols].astype(float).values
        next_rec = well_data.iloc[i + 1][channel_cols].astype(float).values
        diff = next_rec - current
        total_changes.extend(diff)
    
    return({
        'well': well,
        'mean_threshold': np.mean(all_thresholds),
        'std_threshold': np.std(all_thresholds),
        'min_threshold': np.min(all_thresholds),
        'max_threshold': np.max(all_thresholds),
        'mean_change': np.mean(np.abs(total_changes)) if total_changes else 0,
        'max_change': np.max(np.abs(total_changes)) if total_changes else 0,
        'stability': np.std(total_changes) if total_changes else 0  # lower = more stable
    })

def make_comparison_figs(all_well_stats, output_path):
    comparison_df = pd.DataFrame(all_well_stats)
    print("\nWell Comparison Summary:")
    print(comparison_df.to_string(index=False))
    # Save comparison
    comparison_df.to_csv(f'{output_path}/well_comparison.csv', index=False)

    # Visualize comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Mean thresholds by well
    axes[0, 0].bar(comparison_df['well'], comparison_df['mean_threshold'])
    axes[0, 0].set_title('Mean Threshold by Well')
    axes[0, 0].set_ylabel('Mean Threshold')
    axes[0, 0].set_xlabel('Well')

    # Plot 2: Threshold variability (std)
    axes[0, 1].bar(comparison_df['well'], comparison_df['std_threshold'])
    axes[0, 1].set_title('Threshold Variability by Well')
    axes[0, 1].set_ylabel('Std Dev')
    axes[0, 1].set_xlabel('Well')

    # Plot 3: Mean change magnitude
    axes[1, 0].bar(comparison_df['well'], comparison_df['mean_change'])
    axes[1, 0].set_title('Mean Change Magnitude by Well')
    axes[1, 0].set_ylabel('Mean |Change|')
    axes[1, 0].set_xlabel('Well')

    # Plot 4: Stability (lower = more stable)
    axes[1, 1].bar(comparison_df['well'], comparison_df['stability'])
    axes[1, 1].set_title('Stability by Well (lower = more stable)')
    axes[1, 1].set_ylabel('Change Std Dev')
    axes[1, 1].set_xlabel('Well')

    plt.tight_layout()
    plt.savefig(f'{output_path}/well_comparison.png')
    plt.close()

def extract_threshold_waveforms(signal, threshold, fs):
    """
    Extracts spike-aligned waveforms and timing information.
    Requires the voltage trace to cross 0 before the next crossing time is detected.

    Parameters
    ----------
    signal : 1D array
        Voltage trace.
    threshold : float
        Threshold value (positive).
    fs : float
        Sampling frequency in Hz (e.g., 12500).

    Returns
    -------
    crossing_times : (num_crossings,) array
        Crossing times in seconds.
    """
    samples = int(round(0.001 * fs))       # 1 ms before and after
    window = np.arange(-samples, samples + 1)
    num_samples = len(window)

    # Detect negative threshold crossings (downward crossings from above -threshold to below -threshold)
    neg_threshold_crossings = np.where(np.diff(np.concatenate([[0], signal < -threshold])) == 1)[0]
    
    # Detect zero crossings (signal crosses from negative to positive or positive to negative)
    # Find where consecutive samples have opposite signs (product is negative)
    zero_crossings = np.where(signal[:-1] * signal[1:] < 0)[0]
    
    # Filter threshold crossings: only keep those where a zero crossing occurred since the last threshold crossing
    valid_crossings = []
    last_threshold_idx = -1
    
    for thresh_idx in neg_threshold_crossings:
        # Check if there's a zero crossing between the last threshold crossing and this one
        if last_threshold_idx == -1:
            # First crossing is always valid
            valid_crossings.append(thresh_idx)
            last_threshold_idx = thresh_idx
        else:
            # Check if there's a zero crossing after the last threshold crossing and before this one
            zero_after_last = zero_crossings[(zero_crossings > last_threshold_idx) & (zero_crossings < thresh_idx)]
            if len(zero_after_last) > 0:
                # Found a zero crossing, so this threshold crossing is valid
                valid_crossings.append(thresh_idx)
                last_threshold_idx = thresh_idx
    
    crossings = np.array(valid_crossings)
    num_crossings = len(crossings)

    for i, t in enumerate(crossings):
        sample_idx = t + window
        valid = (sample_idx >= 0) & (sample_idx < len(signal))

    crossing_times = crossings / fs

    return crossing_times


def test_treatment_effectiveness(day, well, threshold_matrix, bin_file_folder, output_folder, num_channels=16):
    # PARAMETER
    control_spike_times_per_channel= get_spike_times(os.path.join(bin_files_folder, f"**/006/**d{day}**{well}.bin"))
    treatment_spike_times_per_channel= get_spike_times(os.path.join(bin_files_folder, f"**/009/**d{day}**{well}.bin"))

    # Create times of threshold crossing vs total number of spikes (cumulative) for each channel
    for ch in range(num_channels):
        control_spike_times = np.asarray(control_spike_times_per_channel[ch])
        treatment_spike_times = np.asarray(treatment_spike_times_per_channel[ch])
        plt.figure(figsize=(10, 8))

        if control_spike_times.size > 0:
            control_sorted = np.sort(control_spike_times)
            control_counts = np.arange(1, control_sorted.size + 1)
            plt.step(control_sorted, control_counts, where='post', label='Control')

        if treatment_spike_times.size > 0:
            treatment_sorted = np.sort(treatment_spike_times)
            treatment_counts = np.arange(1, treatment_sorted.size + 1)
            plt.step(treatment_sorted, treatment_counts, where='post', label='Treatment')

        plt.title(f'Well {well} - Channel {ch} Spike Counts Over Time')
        plt.xlabel('Time (s)')
        plt.ylabel('Cumulative spike count')
        plt.legend()
        plt.tight_layout()
        os.makedirs(f'{output_folder}/{well}', exist_ok=True)
        plt.savefig(f'{output_folder}/{well}/ch_{ch}_spike_counts.png')
        plt.close()

    # Per-channel percent reduction in spike count for this well
    control_counts = np.array([len(st) for st in control_spike_times_per_channel])
    treatment_counts = np.array([len(st) for st in treatment_spike_times_per_channel])
    with np.errstate(divide='ignore', invalid='ignore'):
        percent_reduction = ((control_counts - treatment_counts) / control_counts) * 100
        percent_reduction[~np.isfinite(percent_reduction)] = np.nan
    
    return percent_reduction

def get_spike_times(bin_pattern, num_channels=16, sampling_frequency=50000)
    bin_files = glob.glob(control_bin_pattern, recursive=True)
    if len(bin_files) == 0:
        raise FileNotFoundError(f"No control bin file found for pattern: {bin_pattern}")
    bin_file = bin_files[0]
    print(f'control: {bin_file}')
    file_stat = os.stat(bin_file)
    num_elements = file_stat.st_size // 4  # float32 has 4 bytes
    num_samples = num_elements // args.num_channels
    data = np.memmap(bin_file, dtype='float32', mode='r', shape=(num_samples, num_channels))   

    spike_times_per_channel = []
    for ch in range(num_channels):
        x = data[:, ch]
        # print(f'threshold for ch {ch}: {threshold_matrix.iloc[ch, 0]}')
        spike_times = extract_threshold_waveforms(x, threshold_matrix.iloc[ch, 0], args.sampling_frequency)
        spike_times_per_channel.append(spike_times)
    
    del data

    return spike_times_per_channel