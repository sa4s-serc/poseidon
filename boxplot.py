import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import numpy as np
DEEPBLUE = '#4c72b0'
import time
# Load the data from the specified pickle file
i = 10
with open(f"Results/alpha0.5_10/results_cumulative_{10*i}.pkl", "rb") as f:
    result = pickle.load(f)

avg_delay_criticality, avg_delay_vsvbp, avg_delay_poseidon_zero, avg_delay_poseidon_one, avg_delay_neptune_zero, avg_delay_neptune_one, avg_cost_criticality, avg_cost_vsvbp, avg_cost_poseidon_zero, avg_cost_poseidon_one, avg_cost_neptune_zero, avg_cost_neptune_one, avg_decision_time_criticality, avg_decision_time_vsvbp, avg_decision_time_poseidon_zero, avg_decision_time_poseidon_one, avg_decision_time_neptune_zero, avg_decision_time_neptune_one = result

# Data for box plots
data = {
    "avg_delay_criticality": avg_delay_criticality,
    "avg_delay_vsvbp": avg_delay_vsvbp,
    "avg_delay_poseidon_zero": avg_delay_poseidon_zero,
    "avg_delay_poseidon_one": avg_delay_poseidon_one,
    "avg_delay_neptune_zero": avg_delay_neptune_zero,
    "avg_delay_neptune_one": avg_delay_neptune_one,
    "avg_cost_criticality": avg_cost_criticality,
    "avg_cost_vsvbp": avg_cost_vsvbp,
    "avg_cost_poseidon_zero": avg_cost_poseidon_zero,
    "avg_cost_poseidon_one": avg_cost_poseidon_one,
    "avg_cost_neptune_zero": avg_cost_neptune_zero,
    "avg_cost_neptune_one": avg_cost_neptune_one,
    "avg_decision_time_criticality": avg_decision_time_criticality,
    "avg_decision_time_vsvbp": avg_decision_time_vsvbp,
    "avg_decision_time_poseidon_zero": avg_decision_time_poseidon_zero,
    "avg_decision_time_poseidon_one": avg_decision_time_poseidon_one,
    "avg_decision_time_neptune_zero": avg_decision_time_neptune_zero,
    "avg_decision_time_neptune_one": avg_decision_time_neptune_one
}

# Color for all boxes
box_color = DEEPBLUE

sns.set_style("darkgrid")

# Plot the average delays
plt.figure(figsize=(10, 6))
sns.boxplot(data=[data["avg_delay_criticality"], data["avg_delay_vsvbp"], data["avg_delay_poseidon_zero"], data["avg_delay_poseidon_one"], data["avg_delay_neptune_zero"], data["avg_delay_neptune_one"]],
            palette=[box_color]*6)
plt.xticks(ticks=range(6), labels=["CR-EUA", "VSVBP", "Poseidon\n"+r"[$\alpha$ = 0]", "Poseidon\n"+r"[$\alpha$ = 0.5]", "Neptune\n"+r"[$\alpha$ = 0]", "Neptune\n"+r"[$\alpha$ = 0.5]"],fontsize=16)
plt.title('Average Delays',fontsize=16)
plt.ylabel(r'Delay (in $ms$)',fontsize=16)
plt.tight_layout()
plt.savefig("boxplot_average_delays.png")
plt.show()

# Plot the average costs
plt.figure(figsize=(10, 6))
sns.boxplot(data=[data["avg_cost_criticality"], data["avg_cost_vsvbp"], data["avg_cost_poseidon_zero"], data["avg_cost_poseidon_one"], data["avg_cost_neptune_zero"], data["avg_cost_neptune_one"]],
            palette=[box_color]*6)
plt.xticks(ticks=range(6), labels=["CR-EUA", "VSVBP", "Poseidon\n"+r"[$\alpha$ = 0]", "Poseidon\n"+r"[$\alpha$ = 0.5]", "Neptune\n"+r"[$\alpha$ = 0]", "Neptune\n"+r"[$\alpha$ = 0.5]"],fontsize=16)
plt.title('Average Costs',fontsize=16)
plt.ylabel('Cost',fontsize=16)
# plt.yticks(ticks=range(40,130,10))
plt.tight_layout()
plt.savefig("boxplot_average_costs.png")
plt.show()

data = [data["avg_decision_time_criticality"], data["avg_decision_time_vsvbp"], data["avg_decision_time_poseidon_zero"], data["avg_decision_time_poseidon_one"], data["avg_decision_time_neptune_zero"], data["avg_decision_time_neptune_one"]]

for i,data_item in enumerate(data):
    for j,data_element in enumerate(data_item):
        data[i][j]*=1000

# Plot the average decision times
plt.figure(figsize=(10, 6))
sns.boxplot(data=data,
            palette=[box_color]*6)
plt.xticks(ticks=range(6), labels=["CR-EUA", "VSVBP", "Poseidon\n"+r"[$\alpha$ = 0]", "Poseidon\n"+r"[$\alpha$ = 0.5]", "Neptune\n"+r"[$\alpha$ = 0]", "Neptune\n"+r"[$\alpha$ = 0.5]"],fontsize=16)
plt.title('Average Decision Times',fontsize=16)
# plt.yticks(ticks=range(0,200,50))
plt.ylabel(r'Decision Time (in $ms$)',fontsize=16)
plt.tight_layout()
plt.savefig("boxplot_average_decision_times.png")
plt.show()
