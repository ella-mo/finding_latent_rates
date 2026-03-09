import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import seaborn as sns
import os
import re
import pandas as pd
from itertools import combinations

# SPIKE TIME ANALYSIS FUNCTIONS
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from data_functions.general_functions import channel_mapping_indices_to_actual, channel_mapping_indices_to_color

def create_correlogram_plots(spike_times, channel_indices, dirname, well, output_path, fs=50000, min_lag_seconds=0.006, max_lag_seconds=10.0, bin_size_seconds=0.002, recording_duration_seconds=900):
    """
    Create autocorrelogram plots for given channel indices, and
    cross-correlogram plots for all pairs of the given channel indices.
    
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
    peak_lags_per_channel = {}

    # Pre-compute binned spike trains for all requested channels
    num_bins = int(recording_duration_seconds / bin_size_seconds)
    spike_trains = {}

    for channel_idx in channel_indices:
        spike_time = np.asarray(spike_times[channel_idx])

        spike_train = np.zeros(num_bins)
        if spike_time.size > 0:
            spike_bins = (spike_time / bin_size_seconds).astype(int)
            spike_bins = spike_bins[spike_bins < num_bins]
            for bin_idx in spike_bins:
                spike_train[bin_idx] += 1

        spike_trains[channel_idx] = spike_train

    # Autocorrelograms
    for channel_idx in channel_indices:
        spike_train = spike_trains[channel_idx]
        spike_train = (spike_train - np.mean(spike_train)) / np.std(spike_train)

        autocorr = signal.correlate(spike_train, spike_train, mode='full') / (len(spike_train) * bin_size_seconds)

        lags = signal.correlation_lags(len(spike_train), len(spike_train), mode='full')
        lags_seconds = lags * bin_size_seconds

        peaks, properties = signal.find_peaks(autocorr, height=1) # PARAMETER
        peak_lags_per_channel[channel_idx] = list(lags_seconds[peaks])

        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(1, 1, 1)

        ax.plot(lags_seconds, autocorr, channel_mapping_indices_to_color(channel_idx, well), linewidth=1, label='Autocorrelogram')
        if len(peaks) > 0:
            ax.plot(lags_seconds[peaks], autocorr[peaks], 'ro',
                    markersize=8, label=f'Peaks (n={len(peaks)})', zorder=5)
            ax.legend()
        ax.set_xlabel('Lag (s)', fontsize=14)
        ax.set_ylabel('Normalized Autocorrelation', fontsize=14)
        fig.suptitle(f'Autocorrelogram for Channel {channel_mapping_indices_to_actual(channel_idx)} ', fontsize=16)
        ax.set_title(f'{dirname}', fontsize=8)
        ax.set_xlim(min_lag_seconds, max_lag_seconds)
        ax.set_ylim(-2, 10)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        plt.savefig(f'{output_path}/auto_correlograms/{channel_mapping_indices_to_actual(channel_idx)}.png')
        plt.close()

    for ch1, ch2 in combinations(channel_indices, 2):
        train1 = spike_trains[ch1]
        train2 = spike_trains[ch2]
        train1 = (train1 - np.mean(train1)) / np.std(train1)
        train2 = (train2 - np.mean(train2)) / np.std(train2)

        crosscorr = signal.correlate(train1, train2, mode='full') / len(train1)

        # # smoothing function
        # def smoothed_cross_corr(data, window_size):
        #     return np.convolve(data, np.ones(window_size) / window_size, mode='same') #https://www.statology.org/how-to-perform-time-series-analysis-with-scipy/
        # # Convert time to bins
        # window_size_seconds = 0.5  # PARAMETER
        # window_size_bins = int(window_size_seconds / bin_size_seconds)
        # if window_size_bins % 2 == 0:
        #     window_size_bins += 1
        # smoothed_corr_sg = smoothed_cross_corr(crosscorr, window_size_bins)
        # smoothed_corr_sg = gaussian_filter(crosscorr, sigma=2)
        
        lags = signal.correlation_lags(len(train1), len(train2), mode='full')
        lags_seconds = lags * bin_size_seconds

        fig = plt.figure(figsize=(16, 6))
        ax = fig.add_subplot(1, 1, 1)

        ax.plot(lags_seconds, crosscorr, channel_mapping_indices_to_color(ch1, well), alpha=0.5, linewidth=1.5, label='Cross-correlogram')
        # ax.plot(lags_seconds, smoothed_corr_sg, 'black', linewidth=1, label='Smoothed cross-correlogram')
        ax.legend()
        ax.axvline(0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.set_xlabel('Lag (s)', fontsize=14)
        ax.set_ylabel('Normalized Cross-correlation', fontsize=14)
        fig.suptitle(f'Cross-correlogram for Channels {channel_mapping_indices_to_actual(ch1)} & {channel_mapping_indices_to_actual(ch2)}', fontsize=16)
        ax.set_title(f'{dirname}', fontsize=8)
        ax.set_xlim(-max_lag_seconds, max_lag_seconds)
        ax.set_ylim(-0.05, 0.05)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        plt.savefig(f'{output_path}/cross_correlograms/{channel_mapping_indices_to_actual(ch1)}_vs_{channel_mapping_indices_to_actual(ch2)}.png')
        plt.close()
    
    return peak_lags_per_channel

def create_peak_lags_histogram(all_peak_lags, days, recordings, well, output_path, bins=20):
    for channel_idx, cumulative_peak_lags in all_peak_lags.items():
        if len(cumulative_peak_lags) == 0:
            continue

        mean_lag = np.mean(cumulative_peak_lags)
        std_lag = np.std(cumulative_peak_lags)

        plt.hist(cumulative_peak_lags, bins=bins, color=channel_mapping_indices_to_color(channel_idx, well))
        plt.xlabel('Peak lag (s)', fontsize=14)
        plt.ylabel('Count', fontsize=14)
        plt.suptitle(f'Peak lags histogram for Channel {channel_mapping_indices_to_actual(channel_idx)} - Mean: {mean_lag:.3f}s, Std: {std_lag:.3f}s')
        plt.title(f"Days {', '.join(str(d) for d in days)}, Recordings {', '.join(str(r) for r in recordings)}", fontsize=8) 
        plt.savefig(f'{output_path}/cross_analysis_{well}_{channel_mapping_indices_to_actual(channel_idx)}.png')
        plt.close()

def create_spike_times_diff_histogram(all_wells_spike_times, days, recordings, well, output_path, bins=20):
    for channel_idx, cumulative_spike_times in all_wells_spike_times.items():
        if len(cumulative_spike_times) == 0:
            continue
            
        mean_spike_time = np.mean(cumulative_spike_times)
        std_spike_time = np.std(cumulative_spike_times)

        plt.hist(cumulative_spike_times, bins=bins, color=channel_mapping_indices_to_color(channel_idx, well))
        plt.xlabel('Spike time diff (s)', fontsize=14)
        plt.ylabel('Count', fontsize=14)
        plt.suptitle(f'Spike time difference histogram for Channel {channel_mapping_indices_to_actual(channel_idx)} - Mean: {mean_spike_time:.3f}s, Std: {std_spike_time:.3f}s')
        plt.title(f"Days {', '.join(str(d) for d in days)}, Recordings {', '.join(str(r) for r in recordings)}", fontsize=8) 
        plt.savefig(f'{output_path}/cross_analysis_{well}_{channel_mapping_indices_to_actual(channel_idx)}.png')
        plt.close()


