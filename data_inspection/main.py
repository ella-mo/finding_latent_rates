import numpy as np
import os
from pathlib import Path
import h5py
from scipy.io import savemat
import sys 
# Add project root to Python path to allow importing data_functions
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from data_functions import stitch_data

if __name__ == "__main__":
    current_path = Path.cwd()
    files_folder = Path(f'{current_path}/files')
    os.makedirs(files_folder, exist_ok=True)

    base_name = 'd73_r000_wD4_b0_02_sl90_0_o0_0'

    data_file = current_path.parent / "data" / f"{base_name}.h5"
    train_indices = current_path.parent /"data"/ f"train_indices_{base_name}.npy"
    valid_indices = current_path.parent /"data"/  f"valid_indices_{base_name}.npy"

    bin_size = 0.02  # seconds per bin
    overlap = 0 # 2 second overlap when binning, included in sample_len

    #RUN FUNCTIONS
    stitch_data(data_file, "data", train_indices, valid_indices, bin_size, overlap, files_folder)

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