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
        print(f'train_encod_data {train_encod_data.shape}')

    # Combine back in original order
    n_sessions = len(train_idx) + len(valid_idx)

    # Combine indices and data in the same order
    all_indices = np.concatenate([train_idx, valid_idx])
    all_data = np.concatenate([train_encod_data, valid_encod_data], axis=0)

    # Sort by indices to restore original order
    sort_order = np.argsort(all_indices)
    data = all_data[sort_order]  
    print(f'data shape: {data.shape}')

    # Stitch time axis
    data_stitched = data.reshape(-1, data.shape[-1])
    
    #save the files
    output_file = files_folder / f'{base_name}_binned_check_work'
    # np.save(f'{output_file}.npy', data)
    savemat(files_folder / f"{output_file}.mat", {'data':data})

if __name__ == "__main__":
    current_path = Path.cwd()
    visualizations_folder = Path(f'{current_path}/visualizations')
    files_folder = Path(f'{current_path}/files')
    os.makedirs(visualizations_folder, exist_ok=True)
    os.makedirs(files_folder, exist_ok=True)

    base_name =  "d73_r000_wD4_12s"
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