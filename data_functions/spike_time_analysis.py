import numpy as np
from scipy import signal
import neo
from elephant.conversion import BinnedSpikeTrain
from elephant.spike_train_correlation import cross_correlation_histogram
import quantities as pq
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import seaborn as sns
import os
import re
import pandas as pd
from itertools import combinations
from matplotlib.widgets import Slider

# SPIKE TIME ANALYSIS FUNCTIONS
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from data_functions.general_functions import channel_mapping_indices_to_actual, channel_mapping_indices_to_color

def compute_binned_spike_train(spike_time, bin_size_seconds, recording_duration_seconds=900):
    """
    Parameters:
    -----------
    spike_time : numpy array
        Data array with shape (1,); 1D array of a single channel's spike times
    bin_size_seconds : float
        Bin size for spike train in seconds 
    
    Returns normalized spike train
    """
    st = neo.SpikeTrain(
        spike_time * pq.s,
        t_start=0 * pq.s,
        t_stop=recording_duration_seconds * pq.s
    )
    return BinnedSpikeTrain(st, bin_size=bin_size_seconds * pq.s, tolerance=1e-10)


def compute_crosscorr_for_pair(
    spike_times_ch1,
    spike_times_ch2,
    bin_size_seconds,
    max_lag_seconds,
    recording_duration_seconds=900,
):
    """
    Compute binned, normalized correlation between two spike trains,
    returning lags (in seconds) and cross-correlation values 
    """
    train1 = compute_binned_spike_train(
        spike_times_ch1, bin_size_seconds, recording_duration_seconds
    )
    train2 = compute_binned_spike_train(
        spike_times_ch2, bin_size_seconds, recording_duration_seconds
    )

    window_bins = int(round(max_lag_seconds/bin_size_seconds))
    cc_hist, lags = cross_correlation_histogram(train1, train2, window=[-window_bins, window_bins])

    cc_hist = cc_hist.flatten().magnitude / (2 * window_bins + 1)

    lags_seconds = lags * bin_size_seconds
    return lags_seconds, cc_hist

def create_crosscorrelogram_plots(
        spike_times,
        channel_indices,
        dirname,
        well,
        output_path,
        ylim_bounds,
        fs=50000,
        min_lag_seconds=0.006,
        max_lag_seconds=10.0,
        bin_size_seconds=0.002,
        recording_duration_seconds=900,
    ):
    """
    Create cross-correlogram plots for given channel indices    
    Parameters:
    -----------
    spike_times : numpy array
        Data array with shape (num_channels,) where each entry is a 1D array
        of spike times for that channel (seconds).
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
    for ch1, ch2 in combinations(channel_indices, 2):
        # One figure per channel-pair for this single recording
        fig, ax = plt.subplots(figsize=(16, 6))

        lags_seconds, crosscorr = compute_crosscorr_for_pair(
            np.asarray(spike_times[ch1]),
            np.asarray(spike_times[ch2]),
            bin_size_seconds=bin_size_seconds,
            max_lag_seconds=max_lag_seconds,
            recording_duration_seconds=recording_duration_seconds,
        )

        # smoothing function
        smooth_crosscorr = gaussian_filter(crosscorr, sigma=2)

        ax.plot(lags_seconds, crosscorr, alpha=0.6, linewidth=2, label='Cross-correlogram')
        ax.plot(lags_seconds, smooth_crosscorr, alpha=0.6, linewidth=2.5, linestyle='--', label='Smoothed cross-correlogram')
        ax.axvline(0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.set_xlabel(
            f'{channel_mapping_indices_to_actual(ch2)} leads {channel_mapping_indices_to_actual(ch1)} ------ '
            f'{channel_mapping_indices_to_actual(ch1)} leads {channel_mapping_indices_to_actual(ch2)}',
            fontsize=14,
        )
        ax.set_ylabel('Cross-correlation', fontsize=14)
        fig.suptitle(
            f'Cross-correlogram for Channels {channel_mapping_indices_to_actual(ch1)} & {channel_mapping_indices_to_actual(ch2)}',
            fontsize=16,
        )
        ax.set_title(f'{dirname}', fontsize=8)
        # ax.set_xlim(-max_lag_seconds, max_lag_seconds)
        ax.legend()
        # ax.set_ylim(ylim_bounds[0], ylim_bounds[1])

        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        plt.savefig(
            f'{output_path}/cross_correlograms/{channel_mapping_indices_to_actual(ch1)}_vs_{channel_mapping_indices_to_actual(ch2)}.png'
        )
        plt.close(fig)


def create_joint_crosscorrelogram_plots(
        spike_times_dict,
        channel_indices,
        dirname,
        well,
        output_path,
        ylim_bounds,
        fs=50000,
        min_lag_seconds=0.006,
        max_lag_seconds=10.0,
        bin_size_seconds=0.002,
        recording_duration_seconds=900,
        interactive=False,
    ):
    """
    Create cross-correlogram plots for given channel indices    
    Parameters:
    -----------
    spike_times_dict : dictionary of numpy array
        Dictionary of key recording number and value data array with shape (num_channels,); each index has the spikes of that indexed channel
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
    interactive : boolean
        
    """
    #TODO: fix naming scheme
    # Extract day from dirname (e.g., "d65_r000_wB2") if possible
    day_match = re.search(r"d(\d+)", dirname)
    day_str = day_match.group(1) if day_match else "?"
    recording_ids = sorted(spike_times_dict.keys())
    recordings_str = ", ".join(f"r{r:03d}" for r in recording_ids)

    for ch1, ch2 in combinations(channel_indices, 2):
        # One figure per channel-pair, with multiple recordings overlaid
        fig, ax = plt.subplots(figsize=(16, 6))
        # Manually position the main axes so that it never overlaps the widget area
        # [left, bottom, width, height] in figure coordinates
        ax.set_position([0.08, 0.12, 0.7, 0.78])
        line_handles = []

        for recording_num, spike_times in spike_times_dict.items():
            init_lags_seconds, init_crosscorr = compute_crosscorr_for_pair(
                np.asarray(spike_times[ch1]),
                np.asarray(spike_times[ch2]),
                bin_size_seconds=bin_size_seconds,
                max_lag_seconds=max_lag_seconds,
                recording_duration_seconds=recording_duration_seconds,
            )

            # smoothing function
            smooth_crosscorr = gaussian_filter(init_crosscorr, sigma=2)

            (line,) = ax.plot(
                init_lags_seconds,
                smooth_crosscorr,
                alpha=0.6,
                linewidth=2.5,
                label=f"{recording_num:03d}",
            )
            line_handles.append(line)

        ax.legend(title="Recording")
        ax.axvline(0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.set_xlabel(f'{channel_mapping_indices_to_actual(ch1)} leads {channel_mapping_indices_to_actual(ch2)} ------ {channel_mapping_indices_to_actual(ch2)} leads {channel_mapping_indices_to_actual(ch1)}', fontsize=14)
        ax.set_ylabel('Cross-correlation', fontsize=14)

        fig.suptitle(
            f'Smoothed Cross-correlograms - Day {day_str}, Well {well}, '
            f'Channels {channel_mapping_indices_to_actual(ch1)} & {channel_mapping_indices_to_actual(ch2)}',
            fontsize=16,
        )
        ax.set_title(f'Recordings: {recordings_str}', fontsize=8)
        # ax.set_xlim(-max_lag_seconds, max_lag_seconds)
        # ax.set_ylim(ylim_bounds[0], ylim_bounds[1])

        ax.grid(True, alpha=0.3)

        if interactive:
            # Sliders on the right side: adjust both bin size and max lag for ALL recordings.
            # Axes position is fixed above; just drop the widgets into the free space on the right.
            ax_bin = fig.add_axes([0.85, 0.55, 0.04, 0.3])
            ax_max = fig.add_axes([0.85, 0.12, 0.04, 0.3])

            bin_slider = Slider(
                ax=ax_bin,
                label='Bin size (s)',
                valmin=min_lag_seconds,
                valmax=bin_size_seconds * 2,
                valinit=bin_size_seconds,
                valstep=0.005
            )
            max_slider = Slider(
                ax=ax_max,
                label='Max lag (s)',
                valmin=min_lag_seconds,
                valmax=recording_duration_seconds / 2,
                valinit=max_lag_seconds,
            )

            def update(_):
                current_bin = max(bin_slider.val, 1e-4)
                current_max = max(max_slider.val, current_bin)

                for (recording_num, spike_times), line in zip(
                    spike_times_dict.items(), line_handles
                ):
                    new_lags, new_crosscorr = compute_crosscorr_for_pair(
                        np.asarray(spike_times[ch1]),
                        np.asarray(spike_times[ch2]),
                        bin_size_seconds=current_bin,
                        max_lag_seconds=max_lag_seconds,
                        recording_duration_seconds=recording_duration_seconds,
                    )
                    line.set_xdata(new_lags)
                    line.set_ydata(new_crosscorr)

                # ax.set_xlim(-current_max, current_max)
                ax.relim()
                ax.autoscale_view(True, True, True)
                fig.canvas.draw_idle()

            bin_slider.on_changed(update)
            max_slider.on_changed(update)

            plt.show()
        else:
            fig.tight_layout()
            plt.savefig(
                f'{output_path}/d{day_str}_w{well}_{channel_mapping_indices_to_actual(ch1)}_vs_{channel_mapping_indices_to_actual(ch2)}.png'
            )
            plt.close(fig)

def create_autocorrelogram_plots(
        spike_times,
        channel_indices,
        dirname,
        well,
        output_path,
        fs=50000,
        min_lag_seconds=0.006,
        max_lag_seconds=10.0,
        bin_size_seconds=0.002,
        recording_duration_seconds=900,
    ):
    """
    Create per-recording autocorrelogram plots: one figure per
    channel per recording.
    """
    for channel_idx in channel_indices:
        lags_seconds, autocorr = compute_crosscorr_for_pair(
            np.asarray(spike_times[channel_idx]),
            np.asarray(spike_times[channel_idx]),
            bin_size_seconds=bin_size_seconds,
            max_lag_seconds=max_lag_seconds,
            recording_duration_seconds=recording_duration_seconds,
        )

        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(
            lags_seconds,
            autocorr,
            channel_mapping_indices_to_color(channel_idx, well),
            linewidth=1,
            label='Autocorrelogram',
        )
        ax.legend()
        ax.set_xlabel('Lag (s)', fontsize=14)
        ax.set_ylabel('Correlation Coefficient', fontsize=14)
        fig.suptitle(
            f'Autocorrelogram for Channel {channel_mapping_indices_to_actual(channel_idx)}',
            fontsize=16,
        )
        ax.set_title(f'{dirname}', fontsize=8)
        # ax.set_xlim(min_lag_seconds, max_lag_seconds)
        ax.set_ylim(-0.5, 1)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        plt.savefig(
            f'{output_path}/auto_correlograms/{channel_mapping_indices_to_actual(channel_idx)}.png'
        )
        plt.close()


def create_joint_autocorrelogram_plots(
        spike_times_dict,
        channel_indices,
        day,
        well,
        output_path,
        fs=50000,
        min_lag_seconds=0.006,
        max_lag_seconds=10.0,
        bin_size_seconds=0.002,
        recording_duration_seconds=900,
    ):
    """
    Create autocorrelogram plots for each channel, overlaying all recordings
    for a given day and well. Figures are saved in a shared folder, one
    per channel.
    """
    recording_ids = sorted(spike_times_dict.keys())
    recordings_str = ", ".join(f"{r:03d}" for r in recording_ids)

    for channel_idx in channel_indices:
        fig, ax = plt.subplots(figsize=(10, 6))

        for recording_num in recording_ids:
            spike_times = spike_times_dict[recording_num]
            lags_seconds, autocorr = compute_crosscorr_for_pair(
                np.asarray(spike_times[channel_idx]),
                np.asarray(spike_times[channel_idx]),
                bin_size_seconds=bin_size_seconds,
                max_lag_seconds=max_lag_seconds,
                recording_duration_seconds=recording_duration_seconds,
            )

            ax.plot(
                lags_seconds,
                autocorr,
                linewidth=1,
                alpha=0.6,
                label=f"{recording_num:03d}",
            )

        ax.legend(title="Recording")
        ax.set_xlabel('Lag (s)', fontsize=14)
        ax.set_ylabel('Correlation Coefficient', fontsize=14)

        fig.suptitle(
            f'Autocorrelograms - Day {day}, Well {well}, '
            f'Channel {channel_mapping_indices_to_actual(channel_idx)}\n'
            f'Recordings: {recordings_str}',
            fontsize=14,
        )

        # ax.set_xlim(min_lag_seconds, max_lag_seconds)
        ax.set_ylim(-0.5, 1) #PARAMETER
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(
            output_path
            / f'd{day}_w{well}_ch{channel_mapping_indices_to_actual(channel_idx)}.png'
        )
        plt.close(fig)


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
