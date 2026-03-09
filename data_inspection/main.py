import numpy as np
import os
from pathlib import Path
import sys 
import pandas as pd
import glob
import re
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to Python path to allow importing data_functions
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from data_functions import (create_correlogram_plots,
                            create_peak_lags_histogram, 
                            create_spike_times_diff_histogram, 
                            parse_filename, 
                            changes_btwn_recordings, 
                            get_well_stats, 
                            make_comparison_figs,
                            test_treatment_effectiveness,
                            channel_mapping_indices_to_actual
                            )

# python main.py '/oscar/home/emohanra/scratch/lizarraga/finding_latent_rates/mea-mua-analysis/files' '/oscar/data/slizarra/emohanra/waveformVariability/bin_files' -d 83 -r 6 -w C2 -vt
# python main.py '/oscar/home/emohanra/scratch/lizarraga/finding_latent_rates/mea-mua-analysis/files' -d 83 -r 6 -w C2 -vt
# python main.py '/oscar/home/emohanra/scratch/lizarraga/finding_latent_rates/mea-mua-analysis/files' -d 65 -r 0 2 4 6 8 10  -w B2 -i 0 5 9 12 13 14 -st
# python main.py '/oscar/home/emohanra/scratch/lizarraga/finding_latent_rates/mea-mua-analysis/files' -d 65 -r 0 -w B1 -i 0 1 4 8 -st
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Inspect and analyze raw data")
    parser.add_argument("lfads_preprocessing_files_folder", type=str, help="full path to grandfather folder of files outputted from LFADS preprocessing to inspect, ie files folder")
    parser.add_argument("bin_files_folder", type=str, nargs='?', default=None, help="full path to grandfather folder of bin_files of voltage recordings")
    parser.add_argument("-d", "--day", type=int, nargs='+', help="day(s) to inspect (in parent folder, which day)") 
    parser.add_argument("-r", "--recording", type=int, nargs='+', help="recordings to inspect (in parent folder, which recording)") 
    parser.add_argument("-w", "--well", type=str, nargs='+', help="well(s) to inspect")
    parser.add_argument("-i", "--indices", type=int, nargs='+', help="channel indices to analyze, 0-indexed, only used for spike time analysis currently") 

    parser.add_argument("-st", "--do_spike_time_analysis", action="store_true", help="whether to do spike time analysis")
    parser.add_argument("-vt", "--do_voltage_threshold_anlaysis", action="store_true", help="whether to do voltage threshold analysis")
    #e.g. if you want to do spike time analysis, add -st flag; voltage threshold analysis, -vt
    parser.add_argument("-n", "--num_channels", default=16, help="number of channels")
    args = parser.parse_args()

    current_path = Path.cwd()
    visualizations_folder = Path(f'{current_path}/visualizations')
    os.makedirs(visualizations_folder, exist_ok=True)
    cross_analysis_folder = Path(f'{visualizations_folder}/cross_analysis')
    os.makedirs(cross_analysis_folder, exist_ok=True)
    peak_lags_folder = Path(f'{cross_analysis_folder}/peak_lags')
    os.makedirs(peak_lags_folder, exist_ok=True)
    spike_times_folder = Path(f'{cross_analysis_folder}/spike_times')
    os.makedirs(spike_times_folder, exist_ok=True)
    treatment_effectiveness_folder = Path(f'{cross_analysis_folder}/treatment_effectiveness')
    os.makedirs(treatment_effectiveness_folder, exist_ok=True)
    voltage_threshold_folder = Path(f'{cross_analysis_folder}/voltage_thresholds')
    os.makedirs(voltage_threshold_folder, exist_ok=True)

    # SPIKE TIME ANALYSIS

    # PARAMETERS
    min_secs = 0.002
    max_secs = 30
    bin_size_secs = 0.15

    if args.do_spike_time_analysis:
        print('Doing spike time analysis...')

        for well in args.well:
            all_wells_spike_time_diff = {}
            all_wells_peak_lags_per_channel = {}

            for day in args.day:
                for recording in args.recording:
                    base_name = f"d{day}_r{recording:03d}_w{well}"
                    filepath = os.path.join(
                        args.lfads_preprocessing_files_folder,
                        base_name,
                        "spike_times.npy"
                    )
                    spike_times = np.load(filepath, allow_pickle=True) #numpy.ndarray

                    for channel_idx in args.indices:
                        spike_time_diff_channel = np.diff(spike_times[channel_idx])
                        if channel_idx not in all_wells_spike_time_diff:
                            all_wells_spike_time_diff[channel_idx] = list(spike_time_diff_channel)
                        else:
                            all_wells_spike_time_diff[channel_idx].extend(list(spike_time_diff_channel))

                    base_name_folder = f'{visualizations_folder}/{base_name}'
                    os.makedirs(f'{base_name_folder}/auto_correlograms', exist_ok=True)
                    os.makedirs(f'{base_name_folder}/cross_correlograms', exist_ok=True)

                    peak_lags_per_channel = create_correlogram_plots(
                        spike_times, 
                        args.indices, 
                        base_name, 
                        well,
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
        
            create_peak_lags_histogram(all_wells_peak_lags_per_channel, args.day, args.recording, well, peak_lags_folder, bins=30)

            #mask to consider min and max
            # Mask each channel's spike times individually and keep dictionary structure
            all_wells_spike_times_masked = {
                k: np.asarray(v)[(np.asarray(v) >= min_secs) & (np.asarray(v) <= max_secs)]
                for k, v in all_wells_spike_time_diff.items()
            }
            create_spike_times_diff_histogram(all_wells_spike_times_masked, args.day, args.recording, well, spike_times_folder, bins=30)      

    # VOLTAGE THRESHOLD ANALYSIS
    if args.do_voltage_threshold_anlaysis:
        print('Doing voltage threshold analysis...')

        threshold_csvs = []
        for day in args.day:
            for recording in args.recording:
                for well in args.well:
                    pattern = os.path.join(
                        args.lfads_preprocessing_files_folder,
                        f"**/d{day}_r{recording:03d}_w{well}**.csv",
                    )
                    threshold_csvs.extend(
                        glob.glob(pattern, recursive=True)
                    )

        merged_df = pd.concat(
            [pd.read_csv(filename).assign(**parse_filename(filename)) 
            for filename in threshold_csvs],
            ignore_index=True)

        # Keep track of data for all requested wells
        all_well_stats = []
        if args.bin_files_folder is None:
            # Create side-by-side threshold heatmap of all wells
            threshold_fig, threshold_axes = plt.subplots(1, len(args.well), figsize=(6*len(args.well), 8))
            if len(args.well) == 1:
                threshold_axes = [threshold_axes]
            # Create side-by-side heatmap of threshold CHANGES for all wells
            change_threshold_fig, change_threshold_axes = plt.subplots(1, len(args.well), figsize=(6*len(args.well), 8))
            if len(args.well) == 1:
                change_threshold_axes = [change_threshold_axes]
        else:
            # Keep track of percent change for all requested wells
            well_percent_change = []
            percent_reduct_fig, percent_reduct_axes = plt.subplots(1, 1, figsize=(10, 6))


        for idx, well in enumerate(args.well):
            well_data = merged_df[merged_df['well'] == well].copy()
            well_data = well_data.sort_values(['day', 'recording'])
            channel_cols = [str(i) for i in range(args.num_channels)]
            
            # Calculate changes between consecutive recordings
            changes = changes_btwn_recordings(well_data, well, channel_cols, voltage_threshold_folder)
            # Calculate changes across all timepoints
            all_well_stats.append(get_well_stats(well_data, well, channel_cols))
            # Create matrix of threshold values (channels x timepoints)
            threshold_matrix = well_data[channel_cols].astype(float).T
            print(threshold_matrix)
            
            if args.bin_files_folder is None:
                timepoint_labels = [f"d{row['day']}_r{row['recording']}" for _, row in well_data.iterrows()]
                
                sns.heatmap(threshold_matrix, annot=True, fmt='.3f', cmap='viridis',
                            xticklabels=timepoint_labels,
                            yticklabels=[f'Ch {channel_mapping_indices_to_actual(ch)}' for ch in range(args.num_channels)],
                            ax=threshold_axes[idx], cbar_kws={'label': 'Threshold'},
                            vmin=4.5, vmax=7)
                threshold_axes[idx].set_title(f'Well {well}')
                threshold_axes[idx].set_xlabel('Timepoint')
                threshold_axes[idx].set_ylabel('Channel')

                # Calculate changes between consecutive recordings
                change_matrix = []
                transition_labels = []
                
                for i in range(len(well_data) - 1):
                    current = well_data.iloc[i][channel_cols].astype(float).values
                    next_rec = well_data.iloc[i + 1][channel_cols].astype(float).values
                    diff = next_rec - current
                    change_matrix.append(diff)
                    transition_labels.append(f"r{well_data.iloc[i]['recording']}→r{well_data.iloc[i+1]['recording']}")
                
                # Transpose so channels are rows, transitions are columns
                change_matrix = np.array(change_matrix).T
                
                try:
                    sns.heatmap(change_matrix, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
                                xticklabels=transition_labels,
                                yticklabels=[f'Ch {channel_mapping_indices_to_actual(ch)}' for ch in range(args.num_channels)],
                                ax=change_threshold_axes[idx], cbar_kws={'label': 'Threshold Change'},
                                vmin=-0.8, vmax=0.3)
                    change_threshold_axes[idx].set_title(f'Well {well} - Threshold Changes')
                    change_threshold_axes[idx].set_xlabel('Transition')
                    change_threshold_axes[idx].set_ylabel('Channel')
                except IndexError:
                    print("Not enough recordings inputted (must have 2 or more)")
            else:
                # Determine treatment effectiveness between two recordings: control and 15/30 minutes after application of treatment 
                treatment_effectiveness_per_well_folder = Path(f'{treatment_effectiveness_folder}/{well}')
                os.makedirs(treatment_effectiveness_per_well_folder, exist_ok=True)
                well_percent_change.append(test_treatment_effectiveness(args.day, well, threshold_matrix, args.bin_files_folder, treatment_effectiveness_per_well_folder))

        if args.bin_files_folder is None:
            # Create comparison DataFrame and visualize comparison
            make_comparison_figs(all_well_stats, voltage_threshold_folder)
            threshold_fig.tight_layout()
            threshold_fig.savefig(f'{voltage_threshold_folder}/all_wells_heatmap.png')
            plt.close(threshold_fig)

            change_threshold_fig.tight_layout()
            change_threshold_fig.savefig(f'{voltage_threshold_folder}/all_wells_changes_heatmap.png')
            plt.close(change_threshold_fig)
        else:
            # Shape to (channels x wells) for heatmap: rows = channels, cols = wells
            # if % is negative, then the number of spikes from the control->treatment *increased*
            percent_matrix = np.vstack(well_percent_change).T
            sns.heatmap(percent_matrix, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
                        xticklabels=args.well,
                        yticklabels=[f'Ch {channel_mapping_indices_to_actual(ch)}' for ch in range(args.num_channels)],
                        ax=percent_reduct_axes, cbar_kws={'label': 'Percent Change'},
                        vmin=-100, vmax=100)
            percent_reduct_fig.tight_layout()
            percent_reduct_fig.savefig(f'{treatment_effectiveness_folder}/percent_change_heatmap.png')
            plt.close(percent_reduct_fig)