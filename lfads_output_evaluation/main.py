import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import os
import h5py 
from pathlib import Path
from scipy import stats 
from scipy.io import savemat
import sys
import pandas as pd
import seaborn as sns
# Add project root to Python path to allow importing data_functions
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from data_functions import stitch_data


def set_up(data, channel_num_idx, t_axis):
    ref_data = data[:,channel_num_idx]
    other_data = np.delete(data, channel_num_idx, axis=1)
    # peak_inds = sp.signal.find_peaks(ref_data, prominence=30)[0]

    ref_data_max = np.max(ref_data)
    peak_inds, properties = sp.signal.find_peaks(ref_data, height=ref_data_max/2, prominence=ref_data_max/2)

    peak_heights = properties['peak_heights']
    IPIs = np.diff(t_axis[peak_inds])  # same units as t_axis

    return ref_data, other_data, peak_inds, IPIs, peak_heights


def channel_mapping_indices_to_actual(channel_num_idx):
    return str((channel_num_idx // 4 + 1) * 10 + (channel_num_idx % 4 + 1))
    # return channel_num_idx


def make_detect_peaks_figs(ref_data, peak_inds, IPIs, channel_num_idx, t_axis, visualizations_folder, max_rate_for_plot):
    # Plot rate trace with peaks
    plt.figure()
    plt.plot(t_axis, ref_data, 'b')
    plt.plot(t_axis[peak_inds], ref_data[peak_inds], 'ro')
    plt.suptitle(f'Channel {channel_mapping_indices_to_actual(channel_num_idx)}')
    plt.xlabel('Time (sec)')
    plt.ylabel('Rate (Hz)')
    # plt.ylim([0, 500])
    plt.ylim([0, max_rate_for_plot])
    
    # if channel_mapping_indices_to_actual(channel_num_idx) == '41':
    #     plt.show()

    specific_save_folder = Path(f'{visualizations_folder}/peaks_ipis')
    os.makedirs(specific_save_folder, exist_ok=True)
    plt.savefig(Path(f'{specific_save_folder}/peaks_ipis_{channel_mapping_indices_to_actual(channel_num_idx)}.png'))
    plt.close()

    
    # Overlay histogram
    plt.figure(1)
    plt.hist(IPIs, bins=np.arange(0, 30, 0.5), alpha=0.4, 
                label=f'Ch {channel_mapping_indices_to_actual(channel_num_idx)}')
    # plt.ylim([0, 20])
    plt.title(f'Histogram of Channel {channel_mapping_indices_to_actual(channel_num_idx)}')
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


if __name__ == '__main__':
    current_path = Path.cwd()
    # base_name = 'toy_data_samp_size_120'
    # base_name = 'toy_data_samp_size_100_l0_1'
    # base_name = 'toy_data_samp_size_120_l0_1'
    # base_name = 'd73_r000_wD4_12s'
    base_name = 'd73_r000_wD4_b0_02_sl90_0_o0_0'

    # parameters
    bin_size = 0.02
    expected_end_time = 900.0
    win_width = 100 # for aligned ensemble
    overlap = 0 # second overlap when binning, included in sample_len
    max_rate_for_plot = 3 # Hz

    # parameters: do which functions
    do_detect_all_channel_peaks = True
    do_make_detect_peaks_figs = True
    do_aligned_peaks = True
    do_make_peak_histograms = True

    # Assumes shape (num_recording, num_samples, num_channels)
    output_file = current_path.parent / "data" / f"lfads_output_{base_name}.h5"
    train_indices = current_path.parent / "data"/ f"train_indices_{base_name}.npy"
    valid_indices = current_path.parent / "data"/ f"valid_indices_{base_name}.npy"
    
    # save folders
    visualizations_folder = Path(f'{current_path}/visualizations/{base_name}')
    files_folder = Path(f'{current_path}/files')
    os.makedirs(visualizations_folder, exist_ok=True)
    os.makedirs(files_folder, exist_ok=True)
    
    # make data file
    data_file = Path(f'{files_folder}/lfads_rates_recordings_{base_name}.npy')
    print(f'data file: {data_file}')
    if not data_file.exists():
        if Path(output_file).exists():
            print("Generating data_file from output_file...")
            data = stitch_data(output_file, "rates", train_indices, valid_indices, bin_size, overlap, files_folder)
            np.save(data_file, data)
        else:
            raise FileNotFoundError(f"Missing output_file: {output_file}")
    data = np.load(data_file)
    print(f'data shape: {data.shape}')

    t_axis = np.arange(0, data.shape[0]) * bin_size
    print(f'time axis: {t_axis.shape}')
    if np.abs(t_axis[-1] - expected_end_time) > 2*bin_size:
        raise Exception('Time axis not close to correct duration')


    if do_detect_all_channel_peaks:
        ipis = {}

        if not do_make_detect_peaks_figs:
            do_make_detect_peaks_figs = True

    if do_make_detect_peaks_figs or do_aligned_peaks or do_make_peak_histograms:
        for channel_num_idx in range(data.shape[1]):
            ref_data, other_data, peak_inds, IPIs, peak_heights = set_up(data, channel_num_idx, t_axis)
            if do_detect_all_channel_peaks:
                ipis[channel_num_idx] = IPIs

            if len(IPIs) > 0:
                if do_make_detect_peaks_figs:
                    make_detect_peaks_figs(ref_data, peak_inds, IPIs, channel_num_idx, t_axis, visualizations_folder, max_rate_for_plot)
                    
                if do_aligned_peaks:
                    aligned_ensemble(ref_data, other_data, peak_inds, channel_num_idx, win_width, bin_size, visualizations_folder, max_rate_for_plot)

                if do_make_peak_histograms:
                    make_peak_histograms(peak_heights, channel_num_idx, visualizations_folder, max_rate_for_plot)
            else:
                print('IPIs length 0')

    if do_detect_all_channel_peaks:
        try:
            if ipis is not None:
               ipi_distribution(ipis, visualizations_folder)           
        except:
            pass