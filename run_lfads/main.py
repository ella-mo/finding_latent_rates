import numpy as np
import glob as glob
from pathlib import Path
import h5py, os, re
import sys


file = '/oscar/data/slizarra/emohanra/waveformVariability_rerun/bin_files/20250502_NIN_B1_D73_and_NIN_B4_D59/000/20250502_NIN-B1 D73 and NIN-B4 D59(000)_D4.bin'
file = Path(file)

num_channels = 16
fs = 50000

# PARAMETERS 
bin_size = 0.005  # seconds per bin
sample_len = 12  # sample len in s
overlap = 2 # 2 second overlap when binning, included in sample_len
split_frac = 0.75 # train - test split
DEBUG = False

recording_duration = 15 * 60 # seconds (force 15 min)

print(f'{sample_len} seconds per sample')

def extract_threshold_waveforms(signal, threshold, fs):
    """
    Extracts spike-aligned waveforms and timing information.

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
    waveforms : (num_crossings, num_samples) array
        Extracted waveforms 
    crossing_times : (num_crossings,) array
        Crossing times in seconds.
    relative_time_axis : (num_samples,) array
        Time axis in seconds relative to crossing.
    """
    samples = int(round(0.001 * fs))       # 1 ms before and after
    window = np.arange(-samples, samples + 1)
    num_samples = len(window)

    # Detect upward and downward threshold crossings
    pos_idx = np.where(np.diff(np.concatenate([[0], signal > threshold])) == 1)[0]
    neg_idx = np.where(np.diff(np.concatenate([[0], signal < -threshold])) == -1)[0]

    crossings = np.sort(np.concatenate([pos_idx, neg_idx]))
    num_crossings = len(crossings)

    waveforms = np.full((num_crossings, num_samples), np.nan, dtype=float)

    for i, t in enumerate(crossings):
        sample_idx = t + window
        valid = (sample_idx >= 0) & (sample_idx < len(signal))
        waveforms[i, valid] = signal[sample_idx[valid]]

    crossing_times = crossings / fs

    return waveforms, crossing_times

def calculate_threshold(curr_channel_data):
    median_val = np.median(curr_channel_data)
    absolute_deviations = np.abs(curr_channel_data - median_val)
    mad = np.median(absolute_deviations)
    stdev = mad / 0.6745
    threshold = 4 * stdev

    return threshold

def extract_info_from_bin_file(bin_file, output_dir):
    """Extract day, recording number, and well from bin file path and name"""
    bin_path = Path(bin_file)
    bin_name = bin_path.name
    
    # Extract well from bin filename (e.g., "20250509_NIN-B1_D80(001)_C5.bin" -> "C5")
    well_match = re.search(r'_([A-D][1-8])\.bin$', bin_name)
    if not well_match:
        raise ValueError(f"Could not extract well from bin filename: {bin_name}")
    well = well_match.group(1)
    
    # Extract day from grandparent directory (e.g., "20250509_NIN-B1_D80" or "20250410_NIN_B1_D51" -> "80" or "51")
    grandparent_dir = bin_path.parent.parent.name
    day_match = re.search(r'B1_D(\d+)', grandparent_dir)
    if not day_match:
        raise ValueError(f"Could not extract day from grandparent directory: {grandparent_dir}")
    day = day_match.group(1)
    
    # Extract recording number from bin filename (e.g., "20250509_NIN-B1_D80(001)_C5.bin" -> "001")
    rec_match = re.search(r'\((\d+)\)', bin_name)
    if not rec_match:
        raise ValueError(f"Could not extract recording number from bin filename: {bin_name}")
    recording = rec_match.group(1)
    
    filename = f'd{day}_r{recording}_w{well}'
    out_path = Path(f'{output_dir} / {filename}.h5')
    # os.makedirs(output_dir, exist_ok=True)

    return out_path, filename


def bin_data(data, num_channels, recording_duration, fs, sample_len, bin_size, overlap):
    """
    Bin spike times into overlapping windows.

    Parameters
    ----------
    data : array, shape (samples, channels)
    num_channels : int
    recording_duration : float
        Total recording length in seconds.
    fs : float
        Sampling frequency (Hz).
    sample_len : float
        Length of each trial/window in seconds (e.g., 12 for 12 s windows).
    bin_size : float
        Bin size in seconds (e.g., 0.005 for 5 ms).
    overlap : float
        Overlap between consecutive windows in seconds (e.g., 2 for 2 s overlap).

    Returns
    -------
    binned_trials : array, shape (n_trials, n_timesteps, num_channels)
    """

    # Convert durations to bins
    n_timesteps = int(np.round(sample_len / bin_size))
    window_size = n_timesteps * bin_size
    stride = window_size - overlap

    # Window start times: first at -overlap, last ends at recording_duration
    window_starts = np.arange(-overlap, recording_duration, stride)[:-1]
    if DEBUG:
        print(f'window starts {window_starts}')

    n_trials = len(window_starts)

    # Initialize output array
    binned_trials = np.zeros((n_trials, n_timesteps, num_channels), dtype=np.float32)

    for ch in range(num_channels):
        x = data[:, ch]
        threshold = calculate_threshold(x)
        _, spike_times = extract_threshold_waveforms(x, threshold, fs)
        if ch==0 and DEBUG:
            print(f'{spike_times}')

        # Bin spikes for each window
        for i, start_time in enumerate(window_starts):
            end_time = start_time + window_size
            mask = (spike_times >= start_time) & (spike_times < end_time)
            spikes_in_window = spike_times[mask]
            if ch == 0 and DEBUG:
                print(f'spikes in window: {spikes_in_window}')

            if spikes_in_window.size == 0:
                continue

            relative_times = spikes_in_window - start_time
            bin_indices = np.floor(relative_times / bin_size).astype(int)
            np.add.at(binned_trials[i, :, ch], bin_indices, 1)

    return binned_trials


#Prep output paths
output_dir = Path.cwd()
out_path, filename = extract_info_from_bin_file(file, output_dir)

# Load bin file
data = np.memmap(file, dtype='float32', mode='r')
data = data.reshape((num_channels, len(data)//num_channels), order='F').T #shape (num_samples, num_channels)
print(f'data shape: {data.shape}')

binned_trials = bin_data(data, num_channels, recording_duration, fs, sample_len, bin_size, overlap)

# Train-val split
# Randomized train/valid split with index tracking
n_sessions = binned_trials.shape[0]
indices = np.arange(n_sessions)

# Shuffle indices reproducibly
rng = np.random.default_rng(seed=0)
rng.shuffle(indices)

# Compute split point
split_point = int(n_sessions * split_frac)

# Split into train and validation indices
train_idx = indices[:split_point]
valid_idx = indices[split_point:]

# Slice data
train_data = binned_trials[train_idx]
valid_data = binned_trials[valid_idx]

# Save index lists for later reconstruction
np.save(f"{output_dir}/train_indices_{filename}.npy", train_idx)
np.save(f"{output_dir}/valid_indices_{filename}.npy", valid_idx)

print(f"Train shape: {train_data.shape}, Valid shape: {valid_data.shape}")
print(f"Saved index lists for reconstruction.")

with h5py.File(out_path, "w") as f:
    f.create_dataset("train_encod_data", data=train_data)
    f.create_dataset("train_recon_data", data=train_data)
    f.create_dataset("valid_encod_data", data=valid_data)
    f.create_dataset("valid_recon_data", data=valid_data)

