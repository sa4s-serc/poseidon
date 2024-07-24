import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.signal import savgol_filter
import pickle

# Load data
with open("reward_training_0.pkl", "rb") as file:
    metrics = pickle.load(file)
print(metrics)
delay = metrics[:15000]

invalid_placements = []
total_invalid = 0

for val in delay:
    if val == -10:
        total_invalid+=1
    invalid_placements.append(total_invalid)

# print(delay)
# delay = invalid_placements
delay = np.cumsum(delay)
iterations = range(len(delay))

# Create smoothed delay data
window_length = min(70, len(delay) - 2)  # Must be odd and less than data length
if (window_length % 2) == 0:
    window_length -= 1
poly_order = 3
smoothed_delay = savgol_filter(delay, window_length, poly_order)

# Set Seaborn theme
# plt.style.use('seaborn-v0_8-whitegrid')
# sns.set_theme(style="whitegrid")
sns.set_theme('poster')
sns.set_palette("crest")

# Set up the plot
plt.figure(figsize=(12, 6))
ax = plt.gca()

# Plot the original delay data
sns.lineplot(x=iterations, y=delay, ax=ax, alpha=0.3, linewidth=1, label='Original Delay')

# Plot the smoothed delay data
# sns.lineplot(x=iterations, y=smoothed_delay, ax=ax, linewidth=4)

# Add markers and labels for workload decisions
# for i in range(0, len(delay), 400):
#     plt.plot(i, smoothed_delay[i], marker='o', markersize=6, color="#134B70")
    
#     # Add a vertical line
#     plt.axvline(x=i, color='gray', linestyle='--', alpha=0.5)
    
    # Add text annotation
    # plt.annotate("Next workload\ndecision", (i, smoothed_delay[i]), 
    #              xytext=(10, 10), textcoords='offset points',
    #              fontsize=9, color='black',
    #              bbox=dict(boxstyle="round,pad=0.3", fc='lightgray', ec='black', alpha=0.8),
    #              arrowprops=dict(arrowstyle="->", color='black'))

# Customize the plot
plt.xlabel("Iterations", fontsize=16)
plt.ylabel("Cumulative Invalid Placements", fontsize=16)
# plt.title("", fontsize=14)

# Style the axes
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tick_params(colors='black',labelsize=12)

# plt.plot([], [], label="Workload iterations", color='gray', linestyle='--')
# Add a legend
# plt.legend(loc='upper left')
# add the dotted line for the legend

# Adjust layout and save
plt.tight_layout()
plt.savefig("total_delay_smoothed_seaborn_theme.png", dpi=300, bbox_inches='tight')
plt.show()