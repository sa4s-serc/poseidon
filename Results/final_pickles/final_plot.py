import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import numpy as np
import time
DEEPBLUE = '#4c72b0'


GREEN = '#5f9e6e'
RED = "B55D60"
PURPLE = "#857AAB"
BROWN = "8d7866"
ORANGE = "#CC8963"
BLUE = "#5975a4"
DEEPBROWN = '#937860'
DEEPGREEN = '#55a868'

darker_pastel_colors = ['#4B8BBE', '#306998', '#FFE873', '#FFD43B', '#646464',
                        '#D9EAD3', '#93C47D', '#EAD1DC', '#DD7E6B', '#6FA8DC']

def plot_bar_delay(vsvbp_delay, poseidon_delay_zero, poseidon_delay_one, neptune_delay_zero, neptune_delay_one, criticality_results, title, bar_color="#9ED2BE"):
    # Data to plot
    labels = ['VSVBP', 'Poseidon\n[α = 0]', 'Poseidon\n[α = 0.5]',
              'Neptune\n[α = 0]', 'Neptune\n[α = 0.5]', 'CR-EUA']
    delays = [vsvbp_delay, poseidon_delay_zero, poseidon_delay_one,
              neptune_delay_zero, neptune_delay_one, criticality_results]
    
    ##sort delays and labels also based on delays
    # delays, labels = zip(*sorted(zip(delays, labels)))

    # Creating a DataFrame for seaborn
    data = pd.DataFrame({
        'Solver': labels,
        'Delay': delays
    })

    # Set style and colors
    # sns.set_style('darkgrid')
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['axes.facecolor'] = '#f5f5f5'
    plt.rcParams['grid.color']= 'grey'
    plt.rcParams['grid.alpha'] = 0.5
    plt.rcParams['font.family'] = "cursive"
    background_color = '#E0E0E0'  # Light grey
    bar_color = "#9ED2BE"
    text_color = '#333333'  # Dark grey

    # Creating the bar plot
    fig, ax = plt.subplots(figsize=(12, 6))
    # fig.patch.set_facecolor(background_color)
    # ax.set_facecolor(background_color)
    
    sns.barplot(x='Solver', y='Delay', data=data, ax=ax, color=bar_color)

    # Customizing the plot
    ax.set_title(title, fontsize=16, fontweight='bold', color=text_color)
    ax.set_xlabel('Solver', fontsize=16, color=text_color)
    
    title_y = ""
    
    if "delay" in title.lower():
        title_y = r"Delay (in $ms$)"
    elif "cost" in title.lower():
        title_y = "Cost"
    else:
        title_y = r"Decision Time (in $ms$)"
    
    ax.set_ylabel(title_y, fontsize=16, color=text_color)

    # Adjust y-axis
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: '{:,.0f}'.format(x)))

    # Add value labels on top of the bars
    for i, v in enumerate(delays):
        ax.text(i, v, f'{v:.4f}', ha='center', va='bottom', fontsize=14, color=text_color)

    # Customize grid
    # ax.grid(axis='y', color='white', linestyle='-', linewidth=1, alpha=0.7)
    # ax.set_axisbelow(True)

    # Adjust spines
    for spine in ax.spines.values():
        spine.set_color('#CCCCCC')

    # Adjust tick colors
    ax.tick_params(colors=text_color)

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(f'{title}.png', dpi=400, bbox_inches='tight')
    plt.show()

# Load the data from the specified pickle file
name = "Results/final_pickle/light_cumulative"
with open(name, "rb") as f:
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

fname = 'Results/final_pickle/light.pkl'

with open(fname, "rb") as f:
        result = pickle.load(f)

        avg_delay_criticality, avg_delay_vsvbp, avg_delay_poseidon_zero, avg_delay_poseidon_one, avg_delay_neptune_zero, avg_delay_neptune_one, avg_decision_time_criticality, avg_decision_time_vsvbp, avg_decision_time_poseidon_zero, avg_decision_time_poseidon_one, avg_decision_time_neptune_zero, avg_decision_time_neptune_one, avg_cost_criticality, avg_cost_vsvbp, avg_cost_poseidon_zero, avg_cost_poseidon_one, avg_cost_neptune_zero, avg_cost_neptune_one = result

        avg_delay_criticality /= 100
        avg_delay_vsvbp /= 100
        avg_delay_poseidon_zero /= 100
        avg_delay_neptune_zero /= 100
        avg_delay_poseidon_one /= 100
        avg_delay_neptune_one /= 100

        avg_decision_time_criticality /= 100
        avg_decision_time_vsvbp /= 100
        avg_decision_time_poseidon_zero /= 100
        avg_decision_time_neptune_zero /= 100
        avg_decision_time_poseidon_one /= 100
        avg_decision_time_neptune_one /= 100

        avg_cost_criticality /= 100
        avg_cost_vsvbp /= 100
        avg_cost_poseidon_zero /= 100
        avg_cost_neptune_zero /= 100
        avg_cost_poseidon_one /= 100
        avg_cost_neptune_one /= 100

        plot_bar_delay(avg_delay_vsvbp, avg_delay_poseidon_zero, avg_delay_poseidon_one, avg_delay_neptune_zero,
                    avg_delay_neptune_one, avg_delay_criticality, "CR-EUA vs VSVBP vs Poseidon vs Neptune Average Delay")
        plot_bar_delay(avg_cost_vsvbp, avg_cost_poseidon_zero, avg_cost_poseidon_one, avg_cost_neptune_zero,
                    avg_cost_neptune_one, avg_cost_criticality, "CR-EUA vs VSVBP vs Poseidon vs Neptune Average Cost")
        plot_bar_delay(avg_decision_time_vsvbp, avg_decision_time_poseidon_zero, avg_decision_time_poseidon_one, avg_decision_time_neptune_zero,
                    avg_decision_time_neptune_one, avg_decision_time_criticality, "CR-EUA vs VSVBP vs Poseidon vs Neptune Average Decision Time")
