import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import seaborn as sns
import os
import re
import pandas as pd 

# SPIKE TIME ANALYSIS FUNCTIONS
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

def create_autocorrelogram_plot(spike_times, channel_indices, dirname, well, output_path, fs=50000, min_lag_seconds=0.006, max_lag_seconds=10.0, bin_size_seconds=0.001, recording_duration_seconds=900):
    """
    Create autocorrelogram plot for given channel indices.
    
    Parameters:
    -----------
    spike_times : numpy array
        Data array with shape (num_channels,); each index has the spikes of that indexed channel
    channel_indices : list
        Channel indices to analyze
    output_path : str or Path
        Where to save the output figure
    fs : int
        Sampling rate in Hz (default: 50000)
    min_lag_seconds : float
        Minimum lag to display in seconds (default: 0.006 to exclude spike width)
    max_lag_seconds : float
        Maximum lag to display in seconds (default: 10.0)
    bin_size_seconds : float
        Bin size for spike train in seconds (default: 0.001, i.e., 1ms)
    recording_duration_seconds : float
        Total recording duration in seconds (default: 900)
    """
    from data_functions.general_functions import channel_mapping_indices_to_actual, channel_mapping_indices_to_color

    peak_lags_per_channel = {}

    for channel_idx in channel_indices:
        spike_time = spike_times[channel_idx]
        
        # Create binned spike train (count-based)
        num_bins = int(recording_duration_seconds / bin_size_seconds)
        spike_train = np.zeros(num_bins)
        
        # Convert spike times to bin indices
        spike_bins = (spike_time / bin_size_seconds).astype(int)
        # Remove any spikes beyond recording duration
        spike_bins = spike_bins[spike_bins < num_bins]
        # Count spikes in each bin
        for bin_idx in spike_bins:
            spike_train[bin_idx] += 1
        
        # Compute autocorrelation 
        autocorr = signal.correlate(spike_train, spike_train, mode='full')
        lags = signal.correlation_lags(len(spike_train), len(spike_train), mode='full')
        
        # Convert lags to seconds
        lags_seconds_full = lags * bin_size_seconds
        
        # Normalize based on min_lag_seconds onwards
        # Find the normalization window
        norm_mask = (lags_seconds_full >= min_lag_seconds) & (lags_seconds_full <= max_lag_seconds)
        autocorr_masked = autocorr[norm_mask]
        autocorr = autocorr / np.max(autocorr_masked)
        
        # Keep only positive lags from min_lag_seconds to max_lag_seconds
        positive_mask = (lags_seconds_full >= min_lag_seconds) & (lags_seconds_full <= max_lag_seconds)
        lags_seconds = lags_seconds_full[positive_mask]
        autocorr = autocorr[positive_mask]
        
        # Find peaks in autocorrelogram
        # PARAMETER - HEIGHT (:= CORRELATION SCORE FOR PEAK FINDING)
        peaks, properties = signal.find_peaks(autocorr, height=0.7) 

        # peak indices to seconds (peaks are indices in the autocorr/lags_seconds arrays)
        # lags_seconds already contains the time values, so we can directly index
        peak_lags_per_channel[channel_idx] = list(lags_seconds[peaks])
        # print(f'Channel {channel_mapping_indices_to_actual(channel_idx)} - Mean peak lag: {np.mean(peak_times_seconds):.3f} s ({np.mean(peak_times_seconds)*1000:.1f} ms)')

        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(1, 1, 1)

        ax.plot(lags_seconds, autocorr, channel_mapping_indices_to_color(channel_idx, well), linewidth=1)
        if len(peaks) > 0:
            ax.plot(lags_seconds[peaks], autocorr[peaks], 'ro', 
                    markersize=8, label=f'Peaks (n={len(peaks)})', zorder=5)
            ax.legend()
        ax.set_xlabel('Lag (s)',fontsize=14)
        ax.set_ylabel('Normalized Autocorrelation',fontsize=14)
        ax.set_title(f'Autocorrelogram for Channel {channel_mapping_indices_to_actual(channel_idx)} - {dirname}',fontsize=16)
        ax.set_xlim(min_lag_seconds, max_lag_seconds)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        plt.savefig(f'{output_path}/auto_correlograms/{channel_mapping_indices_to_actual(channel_idx)}.png')
        plt.close()
    
    return peak_lags_per_channel

#TODO: def find common features between d_r_ws to better name the saved figure

def find_common_features(d_r_ws):
    find_common = [data_point.split('_') for data_point in d_r_ws]
    

def create_peak_lags_histogram(all_peak_lags, d_r_ws, well, output_path, bins=20):
    from data_functions.general_functions import channel_mapping_indices_to_actual, channel_mapping_indices_to_color

    for channel_idx, cumulative_peak_lags in all_peak_lags.items():
        mean_lag = np.mean(cumulative_peak_lags)
        std_lag = np.std(cumulative_peak_lags)

        plt.hist(cumulative_peak_lags, bins=bins, color=channel_mapping_indices_to_color(channel_idx, well))
        plt.xlabel('Peak lag (s)', fontsize=14)
        plt.ylabel('Count', fontsize=14)
        plt.suptitle(f'Peak lags histogram for Channel {channel_mapping_indices_to_actual(channel_idx)} - Mean: {mean_lag:.3f}s, Std: {std_lag:.3f}s')
        plt.title(', '.join(d_r_ws), fontsize=8) 
        plt.savefig(f'{output_path}/cross_analysis_{well}_{channel_mapping_indices_to_actual(channel_idx)}.png')
        plt.close()

def create_spike_times_histogram(all_wells_spike_times, d_r_ws, well, output_path, bins=20):
    from data_functions.general_functions import channel_mapping_indices_to_actual, channel_mapping_indices_to_color

    for channel_idx, cumulative_spike_times in all_wells_spike_times.items():
        mean_spike_time = np.mean(cumulative_spike_times)
        std_spike_time = np.std(cumulative_spike_times)

        plt.hist(cumulative_spike_times, bins=bins, color=channel_mapping_indices_to_color(channel_idx, well))
        plt.xlabel('Spike time diff (s)', fontsize=14)
        plt.ylabel('Count', fontsize=14)
        plt.suptitle(f'Spike time difference histogram for Channel {channel_mapping_indices_to_actual(channel_idx)} - Mean: {mean_spike_time:.3f}s, Std: {std_spike_time:.3f}s')
        plt.title(', '.join(d_r_ws), fontsize=8) 
        plt.savefig(f'{output_path}/cross_analysis__{well}_{channel_mapping_indices_to_actual(channel_idx)}.png')
        plt.close()


