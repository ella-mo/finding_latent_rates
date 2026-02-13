import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import os, re, h5py, sys
from pathlib import Path
from scipy import stats 
from scipy.io import savemat
import pandas as pd
import seaborn as sns

# Add project root to Python path to allow importing data_functions
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from data_functions.general_functions import channel_mapping_indices_to_actual, channel_mapping_indices_to_color

def set_up(data, channel_num_idx, t_axis, min_amplitude=None, distance=None):
    ref_data = data[:,channel_num_idx]
    other_data = np.delete(data, channel_num_idx, axis=1)

    if min_amplitude is None:
        min_amplitude = np.mean(ref_data)
        # print(f'Mean amplitude used as minimum: {min_amplitude}')
    max_amplitude = np.max(ref_data)

    peak_inds, properties = sp.signal.find_peaks(ref_data, height=min_amplitude*0.5, prominence=max_amplitude*0.25,distance=distance) #height=(min_amplitude, max_amplitude/2), distance=distance)

    peak_heights = properties['peak_heights']
    IPIs = np.diff(t_axis[peak_inds])  # same units as t_axis

    return ref_data, other_data, peak_inds, IPIs, peak_heights

def make_detect_peaks_figs(ref_data, peak_inds, IPIs, well, channel_num_idx, t_axis, visualizations_folder, max_rate_for_plot):
    # Plot rate trace with peaks
    plt.figure()
    plt.plot(t_axis, ref_data, color=channel_mapping_indices_to_color(channel_num_idx, well))
    plt.plot(t_axis[peak_inds], ref_data[peak_inds], 'ro')
    plt.suptitle(f'Channel {channel_mapping_indices_to_actual(channel_num_idx)}')
    plt.xlabel('Time (sec)')
    plt.ylabel('Rate (Hz)')
    plt.ylim([0, max_rate_for_plot])
    plt.xlim([350, 370])

    specific_save_folder = Path(f'{visualizations_folder}/peaks_ipis')
    os.makedirs(specific_save_folder, exist_ok=True)
    plt.savefig(Path(f'{specific_save_folder}/peaks_ipis_{channel_mapping_indices_to_actual(channel_num_idx)}.png'))
    plt.close()

    
    # Overlay histogram
    plt.figure(1)
    plt.hist(IPIs, bins=np.arange(0, 30, 0.5),
                label=f'Ch {channel_mapping_indices_to_actual(channel_num_idx)}',
                color=channel_mapping_indices_to_color(channel_num_idx, well))
    # plt.ylim([0, 20])
    plt.title(f'Histogram of Channel {channel_mapping_indices_to_actual(channel_num_idx)} - Mean {np.mean(IPIs):.3f} ± {np.std(IPIs):.3f} s')
    plt.xlabel('Inter-peak Intervals (s)')
    plt.ylabel('Counts')
    specific_save_folder = Path(f'{visualizations_folder}/ipi_histogram')
    os.makedirs(specific_save_folder, exist_ok=True)
    plt.savefig(Path(f'{specific_save_folder}/ipi_histogram_{channel_mapping_indices_to_actual(channel_num_idx)}.png'))
    plt.close()

    # Overlay histogram
    plt.figure(2)
    bins=np.arange(0, 30, 0.5)
    plt.hist(IPIs, bins, alpha=0.4, 
                label=f'Ch {channel_mapping_indices_to_actual(channel_num_idx)}')
    mu, sigma = stats.norm.fit(IPIs)
    # Generate x values for the fitted curve
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    # Calculate the PDF of the fitted Gaussian
    p = stats.norm.pdf(x, mu, sigma)
    # Plot the fitted curve
    bin_width = np.diff(bins)[0]
    p_counts = stats.norm.pdf(x, mu, sigma) * len(IPIs) * bin_width
    plt.plot(x, p_counts, 'k', linewidth=2, label=...)
    plt.suptitle(f'Normalized Histogram of Channel {channel_mapping_indices_to_actual(channel_num_idx)}, \nmean: {mu:.3f}, stdev: {sigma:.3f}')
    plt.xlabel('Inter-peak Intervals (s)')
    plt.ylabel('Counts')
    specific_save_folder = Path(f'{visualizations_folder}/gaussian_ipi_histogram')
    os.makedirs(specific_save_folder, exist_ok=True)
    plt.savefig(Path(f'{specific_save_folder}/gaussian_ipi_histogram_{channel_mapping_indices_to_actual(channel_num_idx)}.png'))
    plt.close()


def make_peak_histograms(peak_heights, channel_num_idx, visualizations_folder, max_rate_for_plot):
    bin_width = 30
    # bin_width = 12
    # max_data = 250
    
    bins = np.arange(0, max_rate_for_plot, bin_width)

    plt.figure()
    plt.hist(peak_heights, bins=bins, alpha=0.4, 
                label=f'Ch {channel_mapping_indices_to_actual(channel_num_idx)}')
    plt.ylim([0, 35])
    plt.xlabel('Rate amplitude (Hz)')
    plt.ylabel('Count')
    
    ax = plt.gca()    
    # Plot KDE on the same axes
    ax2 = ax.twinx()  # Create a second y-axis for the KDE
    sns.kdeplot(peak_heights, ax=ax2, color='red', linewidth=2, label='KDE')
    ax2.set_ylabel('Density', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_ylim([0, None])  # Start from 0
    
    plt.legend(loc='upper right')
    plt.title(f'Amplitude Histogram of Channel {channel_mapping_indices_to_actual(channel_num_idx)} with KDE')

    specific_save_folder = Path(f'{visualizations_folder}/peak_histogram')
    os.makedirs(specific_save_folder, exist_ok=True)
    plt.savefig(Path(f'{specific_save_folder}/peak_histogram_{channel_mapping_indices_to_actual(channel_num_idx)}.png'))
    plt.close()



def aligned_ensemble(ref_data, other_data, peak_inds, channel_num_idx, win_width, bin_size, visualizations_folder, max_rate_for_plot):
    # win_width is number of samples
    aligned_t_axis = np.arange(-win_width,win_width) * bin_size
    
    #within clusters
    aligned_epochs_within = np.full((len(peak_inds),len(aligned_t_axis)), np.nan)

    plt.figure()
    for ind_num, ind in enumerate(peak_inds):
        start = ind - win_width
        end = ind + win_width
        # print(ind_num, ind, start, end, ref_data.shape[0])

        if start < 0 or end > ref_data.shape[0]:
            continue    

        aligned_epochs_within[ind_num,:] = ref_data[start:end]

    plt.plot(aligned_t_axis, aligned_epochs_within.T, alpha=0.5)
    plt.plot(aligned_t_axis, np.nanmean(aligned_epochs_within,axis=0),
        'black',linewidth=2,label='Mean epoch')
    plt.xlabel('Time from peak (sec)')
    plt.ylabel('Rate (Hz)')
    plt.ylim(0, max_rate_for_plot)
    plt.title(f'High rate epochs aligned to peaks, same cluster: Channel {channel_mapping_indices_to_actual(channel_num_idx)}')
    plt.legend()

    specific_save_folder = Path(f'{visualizations_folder}/aligned_ensemble_within_cluster')
    os.makedirs(specific_save_folder, exist_ok=True)
    plt.savefig(Path(f'{specific_save_folder}/aligned_ensemble_within_cluster_{channel_mapping_indices_to_actual(channel_num_idx)}.png'))
    plt.close()

    #cross-cluster
    num_other_channels = np.delete(np.arange(0,16), channel_num_idx)
    # Process each other channel individually
    for i, other_ch_idx in enumerate(num_other_channels):
        aligned_epochs_across = np.full((len(peak_inds), len(aligned_t_axis)), np.nan)
        
        for ind_num, ind in enumerate(peak_inds):
            start = ind - win_width
            end = ind + win_width
            if start < 0 or end > ref_data.shape[0]:
                continue 
                   
            aligned_epochs_across[ind_num,:] = other_data[start:end, i]
        
        plt.figure()
        # Plot all epochs for this channel
        plt.plot(aligned_t_axis, aligned_epochs_across.T, alpha=0.3, linewidth=0.5)
        
        # Plot mean for this channel
        mean_epoch = np.nanmean(aligned_epochs_across, axis=0)
        plt.plot(aligned_t_axis, mean_epoch, alpha=0.7, linewidth=1.5, 
                label=f'Channel {channel_mapping_indices_to_actual(other_ch_idx)} mean', color='r')
    
        # Plot reference channel mean
        plt.plot(aligned_t_axis, np.nanmean(aligned_epochs_within, axis=0),
                linewidth=2, label=f'Reference channel {channel_mapping_indices_to_actual(channel_num_idx)} mean',
                color='b')
    
        plt.xlabel('Time from peak (sec)')
        plt.ylabel('Rate (Hz)')
        plt.ylim(0, max_rate_for_plot)
        plt.suptitle(f'High rate epochs aligned to peaks, cross cluster \nReference channel {channel_mapping_indices_to_actual(channel_num_idx)} vs Channel {channel_mapping_indices_to_actual(other_ch_idx)}')
        plt.legend()
        plt.tight_layout()

        specific_save_folder = Path(f'{visualizations_folder}/aligned_ensemble_across_clusters/ref_{channel_mapping_indices_to_actual(channel_num_idx)}')
        os.makedirs(specific_save_folder, exist_ok=True)
        plt.savefig(Path(f'{specific_save_folder}/aligned_ensemble_across_clusters_ref_{channel_mapping_indices_to_actual(channel_num_idx)}_other_{channel_mapping_indices_to_actual(other_ch_idx)}.png'))
        plt.close()

def ipi_distribution(ipis, visualizations_folder):
    ipis = {key: value for key, value in ipis.items() if value.size > 0}
    overall_mean = np.mean(np.concatenate(list(ipis.values())))
    print(f'Overall mean IPI across all channels: {overall_mean:.4f} s')
    plt.figure()
    plt.boxplot(ipis.values(), labels=[channel_mapping_indices_to_actual(k) for k in ipis.keys()])
    plt.title('IPI distribution for entire well')
    plt.xlabel('Channel')
    plt.ylabel('IPI (s)')

    specific_save_folder = Path(f'{visualizations_folder}/ipi_distribution')
    os.makedirs(specific_save_folder, exist_ok=True)
    plt.savefig(Path(f'{specific_save_folder}/ipi_distribution.png'))
    plt.close()

def cross_recording_ipis_histogram(cross_recording_ipis, d_r_ws, well, output_path, bins=20):
    from data_functions.general_functions import channel_mapping_indices_to_actual, channel_mapping_indices_to_color

    for channel_idx, cumulative_ipis in cross_recording_ipis.items():
        mean_lag = np.mean(cumulative_ipis)
        std_lag = np.std(cumulative_ipis)

        plt.hist(cumulative_ipis, bins=bins, color=channel_mapping_indices_to_color(channel_idx, well))
        plt.xlabel('IPIs (s)', fontsize=14)
        plt.ylabel('Count', fontsize=14)
        plt.suptitle(f'IPIs histogram for Channel {channel_mapping_indices_to_actual(channel_idx)} - Mean: {mean_lag:.3f}s, Std: {std_lag:.3f}s')
        plt.title(', '.join(d_r_ws), fontsize=8) 
        plt.savefig(f'{output_path}/cross_analysis__{well}_{channel_mapping_indices_to_actual(channel_idx)}.png')
        plt.close()