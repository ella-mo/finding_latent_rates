import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import os
import h5py 
from pathlib import Path
from scipy.stats import norm
from scipy.io import savemat
import sys

def calculate_threshold(curr_channel_data):
    median_val = np.median(curr_channel_data)
    absolute_deviations = np.abs(curr_channel_data - median_val)
    mad = np.median(absolute_deviations)
    stdev = mad / 0.6745
    threshold = 4 * stdev

    return threshold

def prepare_output(lfads_output, train_indices, valid_indices, data_file, bin_size, overlap):
    drop_bins = int(overlap/bin_size)
    print(f'drop bins: {drop_bins}')
    train_idx = np.load(train_indices)
    valid_idx = np.load(valid_indices)

    # Regex matches the dates in the file names
    with h5py.File(lfads_output) as f:
        base_name = os.path.splitext(os.path.basename(f.filename))[0]
        # Merge train and valid data for factors and rates
        train_factors = f["train_factors"][:, drop_bins:, :] 
        valid_factors = f["valid_factors"][:, drop_bins:, :] 
        print(f'train factors shape: {train_factors.shape}, valid factors shape: {valid_factors.shape}')

        train_rates = f["train_output_params"][:, drop_bins:, :]  / bin_size
        valid_rates = f["valid_output_params"][:, drop_bins:, :]  / bin_size
        print(f'train rate shape: {train_rates.shape}, valid rates shape: {valid_rates.shape} ')
    
    n_sessions = len(train_idx) + len(valid_idx)
    all_indices = np.concatenate([train_idx, valid_idx])
    
    factors = np.concatenate([train_factors, valid_factors], axis=0)
    rates = np.concatenate([train_rates, valid_rates], axis=0)

    sort_order = np.argsort(all_indices)
    factors = factors[sort_order]  
    rates = rates[sort_order]  

    print(f'factors shape: {factors.shape}')
    print(f'rates shape: {rates.shape}')

    # Stiched rates on one plot from electrodes 
    recordings = rates.reshape(-1, rates.shape[2]) 
    print(f'recordings shape: {recordings.shape}')

    curr_path = Path.cwd()
    mat_file = Path(curr_path) / "files" / f"{base_name}_stitched_binned.mat"
    print(f'mat_file: {mat_file}')
    savemat(mat_file, {'data': recordings})


    np.save(data_file, recordings)

    return factors, rates


def set_up(data, channel_num_idx, t_axis):
    ref_data = data[:,channel_num_idx]
    other_data = np.delete(data, channel_num_idx, axis=1)
    # peak_inds = sp.signal.find_peaks(ref_data, prominence=30)[0]

    threshold = calculate_threshold(ref_data)
    peak_inds = sp.signal.find_peaks(ref_data, width=6, prominence=5)[0]
    IPIs = np.diff(t_axis[peak_inds])  # same units as t_axis

    return ref_data, other_data, peak_inds, IPIs, threshold


def channel_mapping_indices_to_actual(channel_num_idx):
    return str((channel_num_idx // 4 + 1) * 10 + (channel_num_idx % 4 + 1))
    # return channel_num_idx


def make_detect_peaks_figs(ref_data, peak_inds, IPIs, threshold, channel_num_idx, t_axis, visualizations_folder):
    # Plot rate trace with peaks
    plt.figure()
    plt.plot(t_axis, ref_data, 'b')
    plt.plot(t_axis[peak_inds], ref_data[peak_inds], 'ro')
    plt.suptitle(f'Channel {channel_mapping_indices_to_actual(channel_num_idx)} \n height threshold: {threshold:.3f}')
    plt.xlabel('Time (sec)')
    plt.ylabel('Rate (Hz)')
    plt.ylim([0, 350])
    
    # if channel_mapping_indices_to_actual(channel_num_idx) == '41':
    #     plt.show()

    specific_save_folder = Path(f'{visualizations_folder}/peaks_ipis')
    os.makedirs(specific_save_folder, exist_ok=True)
    plt.savefig(Path(f'{specific_save_folder}/peaks_ipis_{channel_mapping_indices_to_actual(channel_num_idx)}.png'))
    plt.close()

    if len(IPIs) > 0:
        # Overlay histogram
        plt.figure(1)
        plt.hist(IPIs, bins=np.arange(0, 30, 0.5), alpha=0.4, 
                    label=f'Ch {channel_mapping_indices_to_actual(channel_num_idx)}')
        plt.ylim([0, 20])
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
        mu, sigma = norm.fit(IPIs)
        # Generate x values for the fitted curve
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        # Calculate the PDF of the fitted Gaussian
        p = norm.pdf(x, mu, sigma)
        # Plot the fitted curve
        bin_width = np.diff(bins)[0]
        p_counts = norm.pdf(x, mu, sigma) * len(IPIs) * bin_width
        plt.plot(x, p_counts, 'k', linewidth=2, label=...)
        plt.suptitle(f'Normalized Histogram of Channel {channel_mapping_indices_to_actual(channel_num_idx)}, \nmean: {mu}, stdev: {sigma}')
        plt.xlabel('Inter-peak Intervals (s)')
        plt.ylabel('Counts')
        specific_save_folder = Path(f'{visualizations_folder}/gaussian_ipi_histogram')
        os.makedirs(specific_save_folder, exist_ok=True)
        plt.savefig(Path(f'{specific_save_folder}/gaussian_ipi_histogram_{channel_mapping_indices_to_actual(channel_num_idx)}.png'))
        plt.close()

def aligned_ensemble(ref_data, other_data, peak_inds, channel_num_idx, win_width, bin_size, visualizations_folder):
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
    plt.ylim(0, 350)
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
        plt.ylim(0, 300)
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
    base_name = 'd73_r000_wA3_12s'

    # parameters
    bin_size = 0.005
    expected_end_time = 900.0
    win_width = 400 # for aligned ensemble
    overlap = 2 # 2 second overlap when binning, included in sample_len

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
            factors, rates = prepare_output(output_file, train_indices, valid_indices, data_file, bin_size, overlap)
        else:
            raise FileNotFoundError(f"Missing output_file: {output_file}")
    data = np.load(data_file)
    print(f'data shape: {data.shape}')

    t_axis = np.arange(0, data.shape[0]) * bin_size
    if np.abs(t_axis[-1] - expected_end_time) > 2*bin_size:
        raise Exception('Time axis not close to correct duration')

    # parameters: do which functions
    do_detect_all_channel_peaks = True
    do_make_detect_peaks_figs = True
    do_aligned_peaks = True

    if do_detect_all_channel_peaks:
        ipis = {}

        if not do_make_detect_peaks_figs:
            do_make_detect_peaks_figs = True

    if do_make_detect_peaks_figs or do_aligned_peaks:
        for channel_num_idx in range(data.shape[1]):
            ref_data, other_data, peak_inds, IPIs, threshold = set_up(data, channel_num_idx, t_axis)
            if do_detect_all_channel_peaks:
                ipis[channel_num_idx] = IPIs

            if do_make_detect_peaks_figs:
                make_detect_peaks_figs(ref_data, peak_inds, IPIs, threshold, channel_num_idx, t_axis, visualizations_folder)
            if do_aligned_peaks and len(IPIs) > 0:
                aligned_ensemble(ref_data, other_data, peak_inds, channel_num_idx, win_width, bin_size, visualizations_folder)
    
    if do_detect_all_channel_peaks:
        try:
            if ipis is not None:
               ipi_distribution(ipis, visualizations_folder)           
        except:
            pass

