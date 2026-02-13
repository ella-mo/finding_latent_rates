import numpy as np
import os
from pathlib import Path
import sys 

# Add project root to Python path to allow importing data_functions
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from data_functions import create_autocorrelogram_plot, create_peak_lags_histogram, create_spike_times_histogram

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Inspect and analyze raw data")
    parser.add_argument("spike_times_grandfather_folder", type=str, help="full path to grandfather folder of spike times to inspect")
    parser.add_argument("-w", "--well", type=str, help="well to inspect (in parent folder of spike times, which well)") #TO DO: DAY EVENTUALLY WHEN NEEDED
    parser.add_argument("-i", "--indices", type=int, nargs='+', help="channel indices to analyze, 0-indexed") 
    parser.add_argument("-n", "--num_channels", type=int, default=16, help="number of channels in the bin file")
    args = parser.parse_args()

    current_path = Path.cwd()
    files_folder = Path(f'{current_path}/files')
    os.makedirs(files_folder, exist_ok=True)

    all_wells_spike_times = {}
    all_wells_peak_lags_per_channel = {}
    d_r_ws = [] #keep track of which day/recording num/well we looked at

    # PARAMETERS
    min_secs = 0.002
    max_secs = 15
    bin_size_secs = 0.5

    for dirpath, dirnames, _ in os.walk(args.spike_times_grandfather_folder):
        for base_name in dirnames:
            if args.well in base_name:
                print(base_name)
                d_r_ws.append(base_name)
                spike_times_npy = os.path.join(dirpath, base_name, 'spike_times.npy')
                spike_times = np.load(spike_times_npy, allow_pickle=True) #numpy.ndarray
                
                for channel_idx in args.indices:
                    spike_time_diff_channel = np.diff(spike_times[channel_idx])
                    if channel_idx not in all_wells_spike_times:
                        all_wells_spike_times[channel_idx] = list(spike_time_diff_channel)
                    else:
                        all_wells_spike_times[channel_idx].extend(list(spike_time_diff_channel))

                visualizations_folder = Path(f'{current_path}/visualizations')
                os.makedirs(visualizations_folder, exist_ok=True)
                base_name_folder = f'{visualizations_folder}/{base_name}'
                os.makedirs(f'{base_name_folder}/auto_correlograms', exist_ok=True)

                peak_lags_per_channel = create_autocorrelogram_plot(
                    spike_times, 
                    args.indices, 
                    base_name, 
                    args.well,
                    base_name_folder, 
                    min_lag_seconds=min_secs,
                    max_lag_seconds=max_secs,
                    bin_size_seconds=bin_size_secs
                )
                for k, v in peak_lags_per_channel.items():
                    if k not in all_wells_peak_lags_per_channel:
                        all_wells_peak_lags_per_channel[k] = list(v)
                    else:
                        all_wells_peak_lags_per_channel[k].extend(v)

                print('\n')

    cross_analysis_folder = Path(f'{visualizations_folder}/cross_analysis')
    os.makedirs(cross_analysis_folder, exist_ok=True)

    peak_lags_folder = Path(f'{cross_analysis_folder}/peak_lags')
    os.makedirs(peak_lags_folder, exist_ok=True)
    create_peak_lags_histogram(all_wells_peak_lags_per_channel, d_r_ws, args.well, peak_lags_folder, bins=30)

    spike_times_folder = Path(f'{cross_analysis_folder}/spike_times')
    os.makedirs(spike_times_folder, exist_ok=True)
    #mask to consider min and max
    # Mask each channel's spike times individually and keep dictionary structure
    all_wells_spike_times_masked = {
        k: np.asarray(v)[(np.asarray(v) >= min_secs) & (np.asarray(v) <= max_secs)]
        for k, v in all_wells_spike_times.items()
    }
    create_spike_times_histogram(all_wells_spike_times_masked, d_r_ws, args.well, spike_times_folder, bins=30)

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