import numpy as np
import matplotlib.pyplot as plt
import sys, os

data_file = '/Users/ellamohanram/Documents/GitHub/finding_latent_rates/data/spike_times.npy' # Day 73, Recording 000, Well D4 spike times
spike_times_data = np.load(data_file, allow_pickle=True)

# Load spike times for channels 42 and 44
# Note: channel indices in data may differ from actual channel numbers
channel_42_idx = 13  # Adjust if needed
channel_44_idx = 15  # Adjust if needed - or use hardcoded list below

channel_42_st = spike_times_data[channel_42_idx] 
# peak sample 'times' in samples, not seconds
channel_42_peak_samples = np.array([409, 1096, 1815, 2588, 3445, 4043, 5015, 5837, 6578, 7193, 7909, 9081, 9764, 10313, 11427, 12318, 13017, 13657, 14566, 15495, 16141, 17101, 17807, 18673, 19344, 20155, 20985, 22016, 22691, 23449, 24168, 24924, 25915, 26647, 27449, 28325, 29028, 29772, 30723, 31664, 32617, 33423, 36071, 36935, 37630, 38289, 39557, 40404, 41291, 42116, 42941, 43861, 44739])
channel_44_peak_samples = np.array([424, 1155, 1823, 3450, 4051, 5014, 5855, 6594, 7237, 7962, 9087, 9829, 10450, 11405, 12341, 13058, 13713, 14677, 15519, 16211, 17798, 18729, 19403, 20200, 22058, 23507, 24165, 25084, 25875, 27454, 28315, 29765, 31680, 32607, 33399, 34136, 34945, 36075, 36929, 38355, 39550, 40460, 41262, 42077, 42953, 43857, 44792])

# Convert channel_42 and channel_44 from samples to seconds
bin_size = 0.02
channel_42_pt = channel_42_peak_samples * bin_size
channel_44_pt = channel_44_peak_samples * bin_size

window_size = 4  # seconds
plot_relative_time = True  # If True, plot time relative to Ch44 spike (centered at 0)

# Ensure channel_42_st is a numpy array
channel_42_st = np.array(channel_42_st)

# Create the raster plot
# Each row (trial) corresponds to one channel_44 spike time
# Each row shows channel_42 spikes within ±window_size seconds of that channel_44 spike
plt.figure(figsize=(12, max(6, len(channel_44_pt) * 0.2)))


everything_dict = {}
# Iterate through each channel_44 spike - each one defines a "trial" (row)
for trial_idx, ch_44_spike_time in enumerate(channel_44_pt):
    # Define the time window around this channel_44 spike
    window_start = ch_44_spike_time - window_size
    window_end = ch_44_spike_time + window_size
    
    # Find all channel_42 spikes that fall within this window
    # This creates the raster for this trial (row)
    ch_42_sts_in_window = channel_42_st[(channel_42_st >= window_start) & (channel_42_st <= window_end)]
    ch_42_pts_in_window = channel_42_pt[(channel_42_pt >= window_start) & (channel_42_pt <= window_end)]
    if ch_42_pts_in_window.size == 0:
        print(trial_idx)
        continue

    # Convert to relative time if True (time relative to Ch44 spike, centered at 0)
    if plot_relative_time:
        ch_42_sts_plot = ch_42_sts_in_window - ch_44_spike_time
        ch_44_st_plot = 0  # Reference spike at time 0
        ch_42_pt_plot = np.min(ch_42_pts_in_window - ch_44_spike_time)
    else:
        ch_42_sts_plot = ch_42_sts_in_window
        ch_44_st_plot = ch_44_spike_time
        ch_42_pt_plot = np.min(ch_42_pts_in_window)
    
    # Plot channel_42 spikes for this trial on row (trial_idx + 1)
    if len(ch_42_sts_plot) > 0:
        # Filter out spikes that are outside the plot window (shouldn't happen, but just in case)
        plot_mask = (ch_42_sts_plot >= -window_size) & (ch_42_sts_plot <= window_size)
        ch_42_sts_plot_filtered = ch_42_sts_plot[plot_mask]
        if len(ch_42_sts_plot_filtered) > 0:
            everything_dict[ch_42_pt_plot] = ch_42_sts_plot_filtered
            # plt.plot(ch_42_sts_plot_filtered, (trial_idx + 1) * np.ones_like(ch_42_sts_plot_filtered), '|', 
            #         markersize=8, markeredgewidth=1.5, color='black', alpha=0.7)

for trial_idx, (ch_42_pt_plot, ch_42_sts_plot_filtered) in enumerate(sorted(everything_dict.items())):
    plt.plot(ch_42_sts_plot_filtered, (trial_idx + 1) * np.ones_like(ch_42_sts_plot_filtered), '|', 
                    markersize=8, markeredgewidth=1.5, color='black', alpha=0.7)
    # # Mark the reference channel_44 spike time with a red marker
    # plt.plot([ch_44_st_plot], [trial_idx + 1], '|', 
    #         markersize=12, markeredgewidth=2, color='red', label='Ch44 spike' if trial_idx == 0 else '')

# Customize the plot
xlabel = "Time relative to Ch44 peak (s)" if plot_relative_time else "Time (s)"
plt.xlabel(xlabel, fontsize=12)
plt.ylabel("Trial (aligned to Ch44 peak)", fontsize=12)
plt.title(f"Channel 42 Spike Raster (aligned to Channel 44 peak, ±{window_size}s window)", fontsize=14)
# Add vertical line at time 0 if using relative time
if plot_relative_time:
    plt.axvline(x=0, color='red', linestyle='--', alpha=0.5, linewidth=1)
# plt.yticks(range(1, len(channel_44_pt) + 1), [f"Trial {j}" for j in range(1, len(channel_44_pt) + 1)])
plt.ylim(0.5, len(everything_dict) + 0.5)
# Set x-axis limits to center at 0 with ±window_size on each side
plt.xlim(-window_size, window_size)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
# plt.show()

os.makedirs(os.path.join(os.getcwd(), 'visualizations'),exist_ok=True)
plt.savefig(os.path.join(os.getcwd(), 'visualizations', 'visual.png'))