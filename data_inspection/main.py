import numpy as np
import os
from pathlib import Path
import h5py
from scipy.io import savemat
import sys 

def stitch_data(binned_h5, train_indices, valid_indices, binned_data_file, bin_size, overlap):
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

    # Combine back in original order
    n_sessions = len(train_idx) + len(valid_idx)
    data = np.empty((n_sessions, *train_encod_data.shape[1:]), dtype=train_encod_data.dtype)

    data[train_idx] = train_encod_data
    data[valid_idx] = valid_encod_data
    order = np.argsort(np.concatenate([train_idx, valid_idx]))
    data = data[order]  
    print(f'data shape: {data.shape}')

    # Stitch time axis
    data_stitched = data.reshape(-1, data.shape[-1])
    
    np.save(binned_data_file, data)
    savemat(files_folder / f"{base_name}_stitched_binned.mat", {'data':data_stitched})

if __name__ == "__main__":
    current_path = Path.cwd()
    visualizations_folder = Path(f'{current_path}/visualizations')
    files_folder = Path(f'{current_path}/files')
    os.makedirs(visualizations_folder, exist_ok=True)
    os.makedirs(files_folder, exist_ok=True)

    base_name =  "d73_r000_wA3"
    data_file = current_path.parent / "data" / f"{base_name}.h5"
    train_indices = current_path.parent /"data"/ f"train_indices_{base_name}.npy"
    valid_indices = current_path.parent /"data"/  f"valid_indices_{base_name}.npy"

    bin_size = 0.005  # seconds per bin
    overlap = 2 # 2 second overlap when binning, included in sample_len

    #RUN FUNCTIONS
    binned_data_file = files_folder / f"{base_name}_binned.npy" # rates back together in order, shape (n_samples, n_bins, n_channels)
    stitch_data(data_file, train_indices, valid_indices, binned_data_file, bin_size, overlap)
