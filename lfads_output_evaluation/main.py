import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import os, re, h5py, sys
from pathlib import Path
from scipy import stats 
from scipy.io import savemat
import pandas as pd
import seaborn as sns

_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from data_functions.general_functions import stitch_data, channel_mapping_indices_to_actual
from data_functions.lfads_output_analysis import set_up, make_detect_peaks_figs, make_peak_histograms, aligned_ensemble, ipi_distribution, cross_recording_ipis_histogram

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Inspect and analyze LFADS output")
    parser.add_argument("lfads_output_grandfather_folder", type=str, help="full path to grandfather folder of lfads outputs to inspect, ie runs folder")
    parser.add_argument("-w", "--well", type=str, help="well to inspect (in parent folder of spike times, which well)") #TO DO: DAY EVENTUALLY WHEN NEEDED
    parser.add_argument("-i", "--indices", type=int, nargs='+', help="channel indices to analyze, 0-indexed") 
    parser.add_argument("-n", "--num_channels", type=int, default=16, help="number of channels in the bin file")
    args = parser.parse_args()

    current_path = Path.cwd()
    lfads_torch_path = current_path.parent / 'mea-mua-analysis'

    # parameters
    bin_size = 0.02
    expected_end_time = 900.0
    win_width = 200 # for aligned ensemble
    overlap = 0 # second overlap when binning, included in sample_len
    max_rate_for_plot = 1.5 # Hz

    # parameters for functions
    # set_up
    distance_sec = 1 

    # parameters: do which functions
    do_detect_all_channel_peaks = True
    do_make_detect_peaks_figs = True
    do_aligned_peaks = False
    do_make_peak_histograms = True
    do_oscar = True

    for dirpath, dirnames, _ in os.walk(args.lfads_output_grandfather_folder):
        cross_recording_ipis = {}
        d_r_ws = []

        for base_name in dirnames:
            if args.well in base_name:
                d_r_ws.append(base_name)
                # Assumes shape (num_recording, num_samples, num_channels)
                if do_oscar:
                    def get_most_recent_run(parent_dir):
                        parent = Path(parent_dir)
                        run_dirs = [p for p in parent.iterdir() if p.is_dir()]
                        if not run_dirs:
                            raise FileNotFoundError(f"No run dirs under {parent}")
                        # Uses the timestamp prefix to sort newest first
                        return max(run_dirs, key=lambda p: p.name.split("_", 1)[0])

                    runs_root = lfads_torch_path / "runs" / base_name
                    output_file = get_most_recent_run(runs_root) / f"lfads_output_{base_name}.h5"

                    train_indices = lfads_torch_path / "files" / base_name / f"train_indices_{base_name}.npy"
                    valid_indices = lfads_torch_path / "files" / base_name / f"valid_indices_{base_name}.npy"
                else:
                    output_file = current_path.parent / "data" / f"lfads_output_{base_name}.h5"
                    train_indices = current_path.parent / "data"/ f"train_indices_{base_name}.npy"
                    valid_indices = current_path.parent / "data"/ f"valid_indices_{base_name}.npy"
                    
                # save folders
                files_folder = Path(f'{current_path}/files')
                os.makedirs(files_folder, exist_ok=True)
                visualizations_folder = Path(f'{current_path}/visualizations')
                base_name_folder = Path(f'{visualizations_folder}/{base_name}')
                os.makedirs(base_name_folder, exist_ok=True)


                # make data file
                data_file = Path(f'{files_folder}/lfads_rates_recordings_{base_name}.npy')
                print(f'data file: {data_file}')
                if not data_file.exists():
                    if Path(output_file).exists():
                        print("Generating data_file from output_file...")
                        data = stitch_data(output_file, "rates", train_indices, valid_indices, bin_size, overlap, files_folder)
                        np.save(data_file, data)
                    else:
                        raise FileNotFoundError(f"Missing output_file: {output_file}")

                data = np.load(data_file, allow_pickle=True)
                t_axis = np.arange(0, data.shape[0]) * bin_size
                if np.abs(t_axis[-1] - expected_end_time) > 2*bin_size:
                    raise Exception('Time axis not close to correct duration')


                if do_detect_all_channel_peaks:
                    ipis = {}
                    
                    if not do_make_detect_peaks_figs:
                        do_make_detect_peaks_figs = True

                if do_make_detect_peaks_figs or do_aligned_peaks or do_make_peak_histograms:
                    for channel_num_idx in args.indices:
                        if distance_sec is not None:
                            distance = distance_sec * 50
                        else:
                            distance = None

                        ref_data, other_data, peak_inds, IPIs, peak_heights = set_up(data, channel_num_idx, t_axis, distance=distance)

                        if do_detect_all_channel_peaks:
                            ipis[channel_num_idx] = IPIs

                            if channel_num_idx not in cross_recording_ipis:
                                cross_recording_ipis[channel_num_idx] = list(IPIs)
                            else:
                                cross_recording_ipis[channel_num_idx].extend(list(IPIs))
                            
                        if len(IPIs) > 0:
                            if do_make_detect_peaks_figs:
                                make_detect_peaks_figs(ref_data, peak_inds, IPIs, args.well, channel_num_idx, t_axis, base_name_folder, max_rate_for_plot)
                                
                            if do_aligned_peaks:
                                aligned_ensemble(ref_data, other_data, peak_inds, channel_num_idx, win_width, bin_size, base_name_folder, max_rate_for_plot)

                            if do_make_peak_histograms:
                                make_peak_histograms(peak_heights, channel_num_idx, base_name_folder, max_rate_for_plot)
                        else:
                            print('IPIs length 0')

                if do_detect_all_channel_peaks:
                    try:
                        if ipis is not None:
                            ipi_distribution(ipis, base_name_folder)           
                    except:
                        pass

        cross_analysis_folder = Path(f'{visualizations_folder}/cross_analysis')
        os.makedirs(cross_analysis_folder, exist_ok=True)
        cross_recording_ipis_histogram(cross_recording_ipis, d_r_ws, args.well, cross_analysis_folder)