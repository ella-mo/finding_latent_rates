import numpy as np
import os
from pathlib import Path
import h5py
from scipy.io import savemat
import sys 

def stitch_data(binned_h5, train_indices, valid_indices, files_folder, bin_size, overlap):
    #stitches the data back in order
    drop_bins = int(overlap/bin_size)
    print(f'drop bins: {drop_bins}')
    # Inputs
    train_idx = np.load(train_indices)
    valid_idx = np.load(valid_indices)

    with h5py.File(binned_h5, "r") as f:
        base_name = os.path.splitext(os.path.basename(f.filename))[0]
        train_encod_data = f["train_encod_data"][:, drop_bins:, :] 
        valid_encod_data = f["valid_encod_data"][:, drop_bins:, :] 

        train_encod_data_check = f["train_encod_data"][:, 0:, :] 
        valid_encod_data_check = f["valid_encod_data"][:, 0:, :] 

    # Combine back in original order
    n_sessions = len(train_idx) + len(valid_idx)

    # Combine indices and data in the same order
    all_indices = np.concatenate([train_idx, valid_idx])

    all_data = np.concatenate([train_encod_data, valid_encod_data], axis=0)
    if overlap:
        all_data_check = np.concatenate([train_encod_data_check, valid_encod_data_check], axis=0)

    # Sort by indices to restore original order
    sort_order = np.argsort(all_indices)
    data = all_data[sort_order]  
    if overlap:
        data_check = all_data_check[sort_order]  
        sorted_indices = all_indices[sort_order]

        print(f'data shape: {data.shape}')
        print(f'data_check shape: {data_check.shape}')
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

    # Stitch time axis
    data_stitched = data.reshape(-1, data.shape[-1])
    
    #save the files
    output_file = f'{base_name}_binned'
    # np.save(f'{output_file}.npy', data)
    savemat(files_folder / f"{output_file}_stitched.mat", {'data':data_stitched})
    print(f"saved {output_file}_stitched.mat")
    if overlap:
        savemat(files_folder / f"{output_file}_check.mat", {'data':data_check})
        print(f"saved {output_file}_check.mat")

if __name__ == "__main__":
    current_path = Path.cwd()
    files_folder = Path(f'{current_path}/files')
    os.makedirs(files_folder, exist_ok=True)

    base_name =  "d73_r000_wA3_12s"
    data_file = current_path.parent / "data" / f"{base_name}.h5"
    train_indices = current_path.parent /"data"/ f"train_indices_{base_name}.npy"
    valid_indices = current_path.parent /"data"/  f"valid_indices_{base_name}.npy"

    bin_size = 0.005  # seconds per bin
    overlap = 0 # 2 second overlap when binning, included in sample_len

    #RUN FUNCTIONS
    stitch_data(data_file, train_indices, valid_indices, files_folder, bin_size, overlap)

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