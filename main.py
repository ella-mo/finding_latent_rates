import numpy as np
import glob as glob
from pathlib import Path
import h5py, os, re


file = '/oscar/data/slizarra/emohanra/waveformVariability_rerun/bin_files/20250502_NIN_B1_D73_and_NIN_B4_D59/000/20250502_NIN-B1 D73 and NIN-B4 D59(000)_D4.bin'
file = Path(file)

num_channels = 16
fs = 50000
recording_duration = 15 * 60  # seconds (force 15 min)

# PARAMETERS 
bin_size = 0.005  # seconds per bin
n_timesteps = 2000  # number of bins per sample
split_frac = 0.75 # train - test split

output_dir = Path(os.getcwd())
filename = "d73_r000.h5"
out_path = output_dir / filename
# os.makedirs(output_dir, exist_ok=True)


print(f'{bin_size * n_timesteps} seconds per sample')

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

# Load bin file
data = np.memmap(file, dtype='float32', mode='r')
data = data.reshape((num_channels, len(data)//num_channels), order='F').T #shape (num_samples, num_channels)

# Get spike times for each channel and bin
n_bins = int(np.ceil(recording_duration / bin_size))
binned = np.zeros((n_bins, num_channels), dtype=np.float32)

for channel_idx in range(num_channels):
    curr_channel_data = data[:, channel_idx]

    # Calculate threshold by Quian Wuiroga et al., 2004 method
    stdev = np.median(np.abs(curr_channel_data)/0.6745)
    threshold = 4 * stdev

    waveforms, spike_times = extract_threshold_waveforms(curr_channel_data, threshold, fs)

    spike_times = spike_times[spike_times < recording_duration]
    bin_ids = np.floor(spike_times / bin_size).astype(int)
    np.add.at(binned[:, channel_idx], bin_ids, 1)

remainder = binned.shape[0] % n_timesteps
if remainder > 0:
    pad_width = n_timesteps - remainder
    binned = np.pad(binned, ((0, pad_width), (0, 0)), constant_values=0)
    print(f"  Padded recording with {pad_width} bins to reach multiple of {n_timesteps}")

n_trials = binned.shape[0] // n_timesteps
binned_trials = binned.reshape(n_trials, n_timesteps, num_channels)

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
np.save(f"{output_dir}/train_indices.npy", train_idx)
np.save(f"{output_dir}/valid_indices.npy", valid_idx)

print(f"Train shape: {train_data.shape}, Valid shape: {valid_data.shape}")
print(f"Saved index lists for reconstruction.")

with h5py.File(out_path, "w") as f:
    f.create_dataset("train_encod_data", data=train_data)
    f.create_dataset("train_recon_data", data=train_data)
    f.create_dataset("valid_encod_data", data=valid_data)
    f.create_dataset("valid_recon_data", data=valid_data)

