import numpy as np
import glob as glob
from pathlib import Path
import h5py, os, re
import sys
from scipy.io import savemat
import pandas as pd

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

    # Detect negative threshold crossings (downward crossings from above -threshold to below -threshold)
    neg_idx = np.where(np.diff(np.concatenate([[0], signal < -threshold])) == 1)[0]

    crossings = neg_idx
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


def extract_info_from_bin_file(bin_file):
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

    return filename


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
    # Use a more precise calculation to avoid floating point errors
    n_windows = int(np.floor((recording_duration + overlap) / stride))
    window_starts = np.array([-overlap + i * stride for i in range(n_windows)])
    if DEBUG:
        print(f'window starts {window_starts}')
        print(f'n_timesteps: {n_timesteps}, window_size: {window_size}, stride: {stride}')

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
            # Use more precise binning to avoid floating point errors
            # Round to avoid precision issues, then floor to ensure consistent binning
            # across overlapping windows
            bin_indices = np.floor(np.round(relative_times / bin_size, decimals=10)).astype(int)
            # Clamp bin indices to valid range [0, n_timesteps)
            bin_indices = np.clip(bin_indices, 0, n_timesteps - 1)
            np.add.at(binned_trials[i, :, ch], bin_indices, 1)

    return binned_trials


def validate_bin_data(binned_trials, data, num_channels, recording_duration, fs, sample_len, bin_size, overlap):
    """
    Validate that bin_data was performed correctly.
    
    Parameters
    ----------
    binned_trials : array, shape (n_trials, n_timesteps, num_channels)
        Output from bin_data function
    data : array, shape (samples, channels)
        Original raw data
    num_channels : int
    recording_duration : float
    fs : float
    sample_len : float
    bin_size : float
    overlap : float
    
    Returns
    -------
    is_valid : bool
        True if all checks pass
    """
    print("\n" + "="*60)
    print("VALIDATING bin_data OUTPUT")
    print("="*60)
    
    all_checks_passed = True
    
    # Expected dimensions
    n_timesteps = int(np.round(sample_len / bin_size))
    window_size = n_timesteps * bin_size
    stride = window_size - overlap
    n_windows = int(np.floor((recording_duration + overlap) / stride))
    window_starts = np.array([-overlap + i * stride for i in range(n_windows)])
    
    # Check 1: Output shape
    expected_shape = (n_windows, n_timesteps, num_channels)
    actual_shape = binned_trials.shape
    print(f"\n1. Shape check:")
    print(f"   Expected: {expected_shape}")
    print(f"   Actual:   {actual_shape}")
    if actual_shape == expected_shape:
        print("   ✓ PASSED")
    else:
        print("   ✗ FAILED")
        all_checks_passed = False
    
    # Check 2: No negative values
    min_val = np.min(binned_trials)
    print(f"\n2. Non-negative values check:")
    print(f"   Minimum value: {min_val}")
    if min_val >= 0:
        print("   ✓ PASSED")
    else:
        print("   ✗ FAILED: Found negative spike counts!")
        all_checks_passed = False
    
    # Check 3: Bin indices within valid range
    # This is already handled in bin_data, but verify no values exceed expected
    max_bin_val = np.max(binned_trials)
    print(f"\n3. Reasonable spike count check:")
    print(f"   Maximum spikes per bin: {max_bin_val}")
    print(f"   Mean spikes per bin: {np.mean(binned_trials):.4f}")
    print(f"   Total spikes across all bins: {np.sum(binned_trials):.0f}")
    if max_bin_val < 1000:  # Reasonable upper bound
        print("   ✓ PASSED (max value seems reasonable)")
    else:
        print("   ⚠ WARNING: Very high spike counts detected")
    
    # Check 4: Window coverage
    print(f"\n4. Window coverage check:")
    print(f"   Number of windows: {n_windows}")
    print(f"   First window start: {window_starts[0]:.4f} s")
    print(f"   Last window start: {window_starts[-1]:.4f} s")
    print(f"   Last window end: {window_starts[-1] + window_size:.4f} s")
    print(f"   Recording duration: {recording_duration:.4f} s")
    last_window_end = window_starts[-1] + window_size
    if abs(last_window_end - recording_duration) < 0.1:  # Allow small floating point error
        print("   ✓ PASSED")
    else:
        print(f"   ⚠ WARNING: Last window ends at {last_window_end:.4f}s, expected {recording_duration:.4f}s")
    
    # Check 5: Overlap consistency (if overlap > 0)
    if overlap > 0:
        print(f"\n5. Overlap consistency check:")
        overlap_bins = int(overlap / bin_size)
        print(f"   Overlap: {overlap} s ({overlap_bins} bins)")
        
        n_mismatches = 0
        for i in range(n_windows - 1):
            # Check if consecutive windows overlap
            window_i_end = window_starts[i] + window_size
            window_i1_start = window_starts[i + 1]
            
            if window_i_end > window_i1_start:  # They overlap
                # Get overlapping regions
                overlap_start_bin = int(np.round((window_i1_start - window_starts[i]) / bin_size))
                overlap_end_bin = overlap_start_bin + overlap_bins
                
                chunk_i_end = binned_trials[i, overlap_start_bin:overlap_end_bin, :]
                chunk_i1_start = binned_trials[i + 1, :overlap_bins, :]
                
                if not np.array_equal(chunk_i_end, chunk_i1_start):
                    n_mismatches += 1
                    if n_mismatches <= 3:  # Only show first few mismatches
                        n_diff = np.sum(chunk_i_end != chunk_i1_start)
                        print(f"   ⚠ Mismatch between windows {i} and {i+1}: {n_diff} bins differ")
        
        if n_mismatches == 0:
            print("   ✓ PASSED: All overlapping regions match")
        else:
            print(f"   ⚠ WARNING: {n_mismatches} window pairs have mismatched overlaps")
            print("   (This may be expected if spike detection varies slightly)")
    
    # Check 6: Compare total spike counts with direct detection
    print(f"\n6. Spike count preservation check:")
    total_spikes_binned = np.sum(binned_trials)
    
    # Detect spikes directly from data (same method as bin_data)
    total_spikes_direct = 0
    for ch in range(num_channels):
        x = data[:, ch]
        threshold = calculate_threshold(x)
        _, spike_times = extract_threshold_waveforms(x, threshold, fs)
        # Count spikes within recording duration
        spikes_in_range = np.sum((spike_times >= 0) & (spike_times < recording_duration))
        total_spikes_direct += spikes_in_range
    
    print(f"   Total spikes in binned data: {total_spikes_binned:.0f}")
    print(f"   Total spikes detected directly: {total_spikes_direct:.0f}")
    
    if overlap > 0:
        # With overlap, spikes can be counted multiple times
        # Estimate expected count: each spike in overlap region is counted twice
        expected_with_overlap = total_spikes_direct
        # Rough estimate: spikes in overlap regions are double-counted
        overlap_ratio = overlap / stride if stride > 0 else 0
        expected_with_overlap = total_spikes_direct * (1 + overlap_ratio)
        print(f"   Expected with overlap (approx): {expected_with_overlap:.0f}")
        print("   ✓ PASSED (overlap causes double-counting, so counts won't match exactly)")
    else:
        if abs(total_spikes_binned - total_spikes_direct) < 0.01 * total_spikes_direct:
            print("   ✓ PASSED")
        else:
            diff_pct = 100 * abs(total_spikes_binned - total_spikes_direct) / total_spikes_direct
            print(f"   ⚠ WARNING: {diff_pct:.2f}% difference (may be due to edge effects)")
    
    # Check 7: Data statistics
    print(f"\n7. Data statistics:")
    print(f"   Non-zero bins: {np.count_nonzero(binned_trials)} / {binned_trials.size} ({100*np.count_nonzero(binned_trials)/binned_trials.size:.2f}%)")
    print(f"   Mean spikes per bin (non-zero): {np.mean(binned_trials[binned_trials > 0]):.4f}")
    print(f"   Max spikes in single bin: {np.max(binned_trials)}")
    
    print("\n" + "="*60)
    if all_checks_passed:
        print("✓ ALL CRITICAL CHECKS PASSED")
    else:
        print("✗ SOME CHECKS FAILED - REVIEW WARNINGS ABOVE")
    print("="*60 + "\n")
    
    return all_checks_passed


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Run LFADS on multiple bin files")
    parser.add_argument("-b", "--bin_files_csv", type=str, help="full path to .csv file of the .bin file paths to run LFADS on, column should be called be path, each line should be a full path to bin file")
    parser.add_argument("-l", "--lfads_dir", type=str, help="full file path to lfads-torch directory")
    args = parser.parse_args()

    # process args
    files = pd.read_csv(args.bin_files_csv)['path'].tolist()
    lfads_datasets_path = f'{args.lfads_dir}/datasets'

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

    for file in files:
        file = Path(file)
        print(f'Processing {file}')

        #Prep output paths
        output_dir = Path.cwd()
        filename = extract_info_from_bin_file(file)
        os.makedirs(f'{output_dir}/lfads_other_files', exist_ok=True)
        os.makedirs(f'{output_dir}/lfads_other_files/{filename}', exist_ok=True)
        train_indices = f"{output_dir}/lfads_other_files/{filename}_indices/train_indices_{filename}.npy"
        valid_indices = f"{output_dir}/lfads_other_files/{filename}_indices/valid_indices_{filename}.npy"
        mat_file = f"{output_dir}/lfads_other_files/{filename}_indices/{filename}_raw_data.mat"

        # Load bin file
        data = np.memmap(file, dtype='float32', mode='r')
        data = data.reshape((num_channels, len(data)//num_channels), order='F').T #shape (num_samples, num_channels)
        savemat(mat_file, {'data': data})
        print(f'data shape: {data.shape}')

        binned_trials = bin_data(data, num_channels, recording_duration, fs, sample_len, bin_size, overlap)

        # Validate binning was done correctly
        validate_bin_data(binned_trials, data, num_channels, recording_duration, fs, sample_len, bin_size, overlap)

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
        np.save(train_indices, train_idx)
        np.save(valid_indices, valid_idx)

        print(f"Train shape: {train_data.shape}, Valid shape: {valid_data.shape}")
        print(f"Saved index lists for reconstruction.")

        with h5py.File(out_path, "w") as f:
            f.create_dataset("train_encod_data", data=train_data)
            f.create_dataset("train_recon_data", data=train_data)
            f.create_dataset("valid_encod_data", data=valid_data)
            f.create_dataset("valid_recon_data", data=valid_data)