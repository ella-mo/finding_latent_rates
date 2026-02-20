import numpy as np
import os
from pathlib import Path
import h5py
from scipy.io import savemat
import sys 

def channel_mapping_indices_to_actual(channel_num_idx):
    return str((channel_num_idx // 4 + 1) * 10 + (channel_num_idx % 4 + 1))

def channel_mapping_indices_to_color(channel_num_idx, well):
    patient_wells = ['B1']

    control_colors = {0: "#57a6ad", 3: "#2c5c39", 11: "#79b559", 14: "#3d99ce"}
    patient_colors = {0: "#9f3b60", 1: "#fdb5ac", 4:"#4f4447", 8:"#fe1d66"}
    
    if well in patient_wells:
        return patient_colors[channel_num_idx]
    else:
        return control_colors[channel_num_idx]

def stitch_data(h5_file, h5_key, train_indices, valid_indices, bin_size, overlap, save_folder=None, do_check=True):
    """
    Inputs: 
    h5_file: h5 file, either binned spikes or lfads output
    type: either "factors", "rates", or "data"
    train_indices: npy file of train indices
    valid_indices: npy file of valid indices
    bin_size: bin size (s)
    overlap: overlap (s)
    save_folder: folder to save output files to, default=current directory
    do_check: check, default = True
    """
    if save_folder is None:
        save_folder = Path.cwd()
    if h5_key not in {"factors", "rates", "data"}:
        raise ValueError(f"Invalid kind '{h5_key}'. Must be 'factors', 'rates', or 'data'.")
    if h5_key == "factors":
        do_check = False
    if h5_key == "rates":
        h5_key = "output_params"
        do_check = False
    if h5_key == "data":
        h5_key = "encod_data"
        

    drop_bins = int(overlap/bin_size)
    print(f'drop bins: {drop_bins}')
    # Convert Path objects to strings for np.load and h5py.File
    train_indices = str(train_indices) if isinstance(train_indices, Path) else train_indices
    valid_indices = str(valid_indices) if isinstance(valid_indices, Path) else valid_indices
    h5_file = str(h5_file) if isinstance(h5_file, Path) else h5_file
    
    train_idx = np.load(train_indices)
    valid_idx = np.load(valid_indices)

    # Regex matches the dates in the file names
    with h5py.File(h5_file) as f:
        base_name = os.path.splitext(os.path.basename(f.filename))[0]
        # Merge train and valid data for factors and rates
        train_data = f[f"train_{h5_key}"][:, drop_bins:, :] 
        valid_data = f[f"valid_{h5_key}"][:, drop_bins:, :] 
        print(f'train_{h5_key} shape: {train_data.shape}, valid_{h5_key} shape: {valid_data.shape}')

        if do_check:
            train_check_data = f[f"train_{h5_key}"][:, 0:, :] 
            valid_check_data = f[f"valid_{h5_key}"][:, 0:, :] 
            print(f'train {h5_key} check shape: {train_check_data.shape}, valid_{h5_key} check shape: {valid_check_data.shape}')


    n_sessions = len(train_idx) + len(valid_idx)
    all_indices = np.concatenate([train_idx, valid_idx])
    
    data = np.concatenate([train_data, valid_data], axis=0)
    if do_check:
        data_check = np.concatenate([train_check_data, valid_check_data], axis=0)

    sort_order = np.argsort(all_indices)
    sorted_indices = all_indices[sort_order]
    data = data[sort_order]  
    print(f'data shape: {data.shape}')
    if do_check and h5_key == "data":
        data_check = data_check[sort_order]
        print(f'Sorted indices (first 10): {sorted_indices[:10]}')
        print(f'Checking if indices are consecutive: {np.all(np.diff(sorted_indices) == 1)}')
        
        # Verify overlap regions match for consecutive chunks
        overlap_bins = int(overlap / bin_size)
        print(f'\nVerifying overlap regions (overlap_bins={overlap_bins})...')
        for i in range(len(data_check)-1):
            chunk_i_end = data_check[i, -overlap_bins:, :]
            chunk_i1_start = data_check[i+1, :overlap_bins, :]
            if np.array_equal(chunk_i_end, chunk_i1_start):
                pass
                # print(f'  Chunks {i} and {i+1} (indices {sorted_indices[i]} and {sorted_indices[i+1]}): overlap matches ✓')
            else:
                n_mismatch = np.sum(chunk_i_end != chunk_i1_start)
                total_elements = chunk_i_end.size
                mismatch_pct = 100 * n_mismatch / total_elements
                print(f'  Chunks {i} and {i+1} (indices {sorted_indices[i]} and {sorted_indices[i+1]}): {n_mismatch} mismatches ({mismatch_pct:.4f}%) ✗')
                
                if sorted_indices[i+1] - sorted_indices[i] != 1:
                    print(f'    WARNING: These chunks are NOT consecutive in original order!')
                
                # Detailed analysis of mismatches
                if n_mismatch > 0:
                    diff_mask = chunk_i_end != chunk_i1_start
                    diff_values = chunk_i_end[diff_mask] - chunk_i1_start[diff_mask]
                    
                    # Find positions of mismatches
                    mismatch_positions = np.where(diff_mask)
                    
                    print(f'    Mismatch details:')
                    print(f'      Total elements in overlap: {total_elements} ({overlap_bins} bins × {chunk_i_end.shape[1]} channels)')
                    print(f'      Absolute differences: min={np.abs(diff_values).min():.6f}, max={np.abs(diff_values).max():.6f}, mean={np.abs(diff_values).mean():.6f}')
                    print(f'      Signed differences: min={diff_values.min():.6f}, max={diff_values.max():.6f}, mean={diff_values.mean():.6f}')
                    
                    # Show first few mismatches with their positions and values
                    n_show = min(10, n_mismatch)
                    print(f'      First {n_show} mismatches:')
                    for idx in range(n_show):
                        bin_idx = mismatch_positions[0][idx]
                        ch_idx = mismatch_positions[1][idx]
                        val_i = chunk_i_end[bin_idx, ch_idx]
                        val_i1 = chunk_i1_start[bin_idx, ch_idx]
                        print(f'        Bin {bin_idx}, Channel {ch_idx}: chunk{i}={val_i:.6f}, chunk{i+1}={val_i1:.6f}, diff={val_i-val_i1:.6f}')
                    
                    # Check if differences are systematic (all same sign) or random
                    if np.all(diff_values > 0):
                        print(f'      Pattern: All mismatches are positive (chunk {i} > chunk {i+1})')
                    elif np.all(diff_values < 0):
                        print(f'      Pattern: All mismatches are negative (chunk {i} < chunk {i+1})')
                    else:
                        print(f'      Pattern: Mixed signs (not systematic)')
                    
                    # Check magnitude relative to typical values
                    if chunk_i_end.size > 0:
                        typical_val = np.median(np.abs(chunk_i_end[chunk_i_end != 0])) if np.any(chunk_i_end != 0) else 1.0
                        max_rel_diff = np.abs(diff_values).max() / typical_val if typical_val > 0 else np.abs(diff_values).max()
                        print(f'      Max relative difference: {max_rel_diff:.4f}x typical value')

    # Stiched rates on one plot from electrodes 
    if h5_key != "factors":
        data = data.reshape(-1, data.shape[2]) 
        print(f'data shape: {data.shape}')

    mat_file = save_folder / f"{base_name}_{h5_key}_stitched_binned.mat"
    # Ensure the directory exists
    mat_file.parent.mkdir(parents=True, exist_ok=True)
    print(f'mat_file: {mat_file}')
    savemat(str(mat_file), {'data': data})
    
    return data


# import numpy as np

# split_frac = 0.75
# # Create a 3D array of shape (5, 3, 4) with random integers between 0 (inclusive) and 10 (exclusive)
# binned_trials = np.random.randint(low=0, high=10, size=(5, 3, 4))
# print("\nUsing np.random.randint():")
# print(binned_trials)

# n_sessions = binned_trials.shape[0]
# indices = np.arange(n_sessions)

# rng = np.random.default_rng(seed=0)
# rng.shuffle(indices)

# # Compute split point
# split_point = int(n_sessions * split_frac)

# # Split into train and validation indices
# train_idx = indices[:split_point]
# valid_idx = indices[split_point:]
# print(f'train_idx {train_idx}')
# print(f'valid_idx {valid_idx}')

# # Slice data
# train_data = binned_trials[train_idx]
# valid_data = binned_trials[valid_idx]
# print(f'train_data {train_data}')
# print(f'valid_data {valid_data}')

# #end
# n_sessions = len(train_idx) + len(valid_idx)

# # Combine indices and data in the same order
# all_indices = np.concatenate([train_idx, valid_idx])
# all_data = np.concatenate([train_data, valid_data], axis=0)

# # Sort by indices to restore original order
# sort_order = np.argsort(all_indices)
# data = all_data[sort_order]  

# print(f'data {data}')