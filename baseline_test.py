import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap
from core import data_to_solver_input
from core.solvers import *
import pickle
import os
from ortools.linear_solver import pywraplp
from stable_baselines3 import A2C, PPO
import matplotlib.ticker as ticker
from core.solvers.workload import *
import plotly.graph_objects as go


GREEN = '#5f9e6e'
RED = "B55D60"
PURPLE = "#857AAB"
BROWN = "8d7866"
ORANGE = "#CC8963"
BLUE = "#5975a4"
DEEPBLUE = '#4c72b0'
DEEPBROWN = '#937860'
DEEPGREEN = '#55a868'


avg_delays_criticality = []
avg_delays_vsvbp = []
avg_delays_poseidon_zero = []
avg_delays_poseidon_one = []
avg_delays_neptune_zero = []
avg_delays_neptune_one = []

avg_costs_criticality = []
avg_costs_vsvbp = []
avg_costs_poseidon_zero = []
avg_costs_poseidon_one = []
avg_costs_neptune_zero = []
avg_costs_neptune_one = []

avg_decision_times_criticality = []
avg_decision_times_vsvbp = []
avg_decision_times_poseidon_zero = []
avg_decision_times_poseidon_one = []
avg_decision_times_neptune_zero = []
avg_decision_times_neptune_one = []


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


def run_solver(payload, solver_type, alpha=None):
    if alpha is not None:
        payload["solver"]["args"]["alpha"] = alpha
    solver = payload.get("solver", {'type': solver_type})
    # solver_type = solver.get("type")
    solver_args = solver.get("args", {})
    with_db = payload.get("with_db", True)

    solver = eval(solver_type)(**solver_args)
    solver.load_data(data_to_solver_input(
        payload, with_db=with_db, workload_coeff=payload.get("workload_coeff", 1)))
    solver.solve()
    return solver.results()


# workload_generator = WorkloadGenerator(
#     dir_path="Cabspotting", functions=np.array([0.25 for _ in range(4)]), nodes=np.array([[0.25, 0.25], [0.25, 0.75], [0.75, 0.25], [0.5, 0.5], [0.75, 0.75]]))

payload = {
        "with_db": False,
        "solver": {
            "type": "",
            "args": {"alpha": 0.1, "verbose": True, "soften_step1_sol": 1.3}
        },
        "workload_coeff": 1,
        "community": "community-test",
        "namespace": "namespace-test",
        "node_names": [
            "node_a", "node_b", "node_c", "node_d", "node_e"
        ],
        "node_delay_matrix": [[0, 3, 2, 6, 4],
                              [3, 0, 4, 5, 10],
                              [2, 4, 0, 3, 2],
                              [6, 5, 3, 0, 2],
                              [4, 10, 2, 2, 0]],
                             
        "workload_on_source_matrix": [[1,1,1,1,1], [1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1]],
        "node_memories": [
            100, 100, 200, 50, 500, 500
        ],
        "node_cores": [
            50, 50, 50, 25, 100, 100
        ],
        "gpu_node_names": [
        ],
        "gpu_node_memories": [
        ],
        "function_names": [
            "ns/fn_1", "ns/fn_2", "ns/fn_3", "ns/fn_4"
        ],
        "function_memories": [
            50, 10,10,10
        ],
        "function_max_delays": [
            1000, 1000, 1000, 1000
        ],
        "gpu_function_names": [
        ],
        "gpu_function_memories": [
        ],
        "actual_cpu_allocations": {
            "ns/fn_1": {
                "node_a": True,
                "node_b": True,
                "node_c": True,
                "node_d": True,
                "node_e": True,
                
            },
            "ns/fn_2": {
                "node_a": True,
                "node_b": True,
                "node_c": True,
                "node_d": True,
                "node_e": True,
                
                
            },
            "ns/fn_3": {
                "node_a": True,
                "node_b": True,
                "node_c": True,
                "node_d": True,
                "node_e": True,
                
            },
            "ns/fn_4": {
                "node_a": True,
                "node_b": True,
                "node_c": True,
                "node_d": True,
                "node_e": True,
                
            },
        "actual_gpu_allocations": {
        },
    }
    }
    
# RUN CRITICALITY

avg_delay_criticality = 0
avg_delay_vsvbp = 0
avg_delay_poseidon_zero = 0
avg_delay_neptune_zero = 0
avg_delay_poseidon_one = 0
avg_delay_neptune_one = 0

avg_decision_time_criticality = 0
avg_decision_time_vsvbp = 0
avg_decision_time_poseidon_zero = 0
avg_decision_time_neptune_zero = 0
avg_decision_time_poseidon_one = 0
avg_decision_time_neptune_one = 0

avg_cost_criticality = 0
avg_cost_vsvbp = 0
avg_cost_poseidon_zero = 0
avg_cost_neptune_zero = 0
avg_cost_poseidon_one = 0
avg_cost_neptune_one = 0

if input("decision?") == "y":

    workloads = []

    for _ in range(151):
        workload_matrix = workload_generator.get_workload()
        workloads.append(workload_matrix)

    for _ in range(151):
        # print("decision number ", _)
        workload_matrix = workloads[_]
        payload["workload_on_source_matrix"] = np.array(workload_matrix)
        # print("decision number ", _)
        criticality_results = run_solver(payload, "Criticality")
        print("Criticality Results", criticality_results)
        avg_delay_criticality += criticality_results[1][0]
        avg_decision_time_criticality += criticality_results[2][0]
        avg_cost_criticality += criticality_results[4][0]
        
        avg_delays_criticality.append(criticality_results[1][0])
        avg_costs_criticality.append(criticality_results[4][0])
        avg_decision_times_criticality.append(criticality_results[2][0])
        
        # print("decision number ", _)
        # RUN VSVBP
        vsvbp_results = run_solver(payload, "VSVBP")
        print("VSVBP Results", vsvbp_results)
        # print(vsvbp_results)
        avg_delay_vsvbp += vsvbp_results[1][0]
        avg_decision_time_vsvbp += vsvbp_results[2][0]
        avg_cost_vsvbp += vsvbp_results[4][0]
        
        avg_delays_vsvbp.append(vsvbp_results[1][0])
        avg_costs_vsvbp.append(vsvbp_results[4][0])
        avg_decision_times_vsvbp.append(vsvbp_results[2][0])
        # print("decision number ", _)
        # RUN POSEIDON
        # print(poseidon_results)
        poseidon_results_zero = run_solver(payload, "Poseidon", 0)
        print("Poseidon Results zero", poseidon_results_zero)
        avg_delay_poseidon_zero += poseidon_results_zero[1][0]
        avg_decision_time_poseidon_zero += poseidon_results_zero[2][0]
        avg_cost_poseidon_zero += poseidon_results_zero[4][0]
        
        avg_delays_poseidon_zero.append(poseidon_results_zero[1][0])
        avg_costs_poseidon_zero.append(poseidon_results_zero[4][0])
        avg_decision_times_poseidon_zero.append(poseidon_results_zero[2][0])

        poseidon_results_one = run_solver(payload, "Poseidon", 0.5)
        print("Poseidon Results one", poseidon_results_one)
        avg_delay_poseidon_one += poseidon_results_one[1][0]
        avg_decision_time_poseidon_one += poseidon_results_one[2][0]
        avg_cost_poseidon_one += poseidon_results_one[4][0]
        
        avg_delays_poseidon_one.append(poseidon_results_one[1][0])
        avg_costs_poseidon_one.append(poseidon_results_one[4][0])
        avg_decision_times_poseidon_one.append(poseidon_results_one[2][0])

        # print("decision number ", _)
        # RUN NEPTUNE
        neptune_results_zero = run_solver(
            payload, "NeptuneMinDelayAndUtilization", 0)
        print("Neptune Results zero", neptune_results_zero)
        avg_delay_neptune_zero += neptune_results_zero[1][0]
        avg_decision_time_neptune_zero += neptune_results_zero[2][0]
        avg_cost_neptune_zero += neptune_results_zero[4][0]
        
        avg_delays_neptune_zero.append(neptune_results_zero[1][0])
        avg_costs_neptune_zero.append(neptune_results_zero[4][0])
        avg_decision_times_neptune_zero.append(neptune_results_zero[2][0])

        neptune_results_one = run_solver(
            payload, "NeptuneMinDelayAndUtilization", 0.5)
        print("Neptune Results one", neptune_results_one)
        avg_delay_neptune_one += neptune_results_one[1][0]
        avg_decision_time_neptune_one += neptune_results_one[2][0]
        avg_cost_neptune_one += neptune_results_one[4][0]
        
        avg_delays_neptune_one.append(neptune_results_one[1][0])
        avg_costs_neptune_one.append(neptune_results_one[4][0])
        avg_decision_times_neptune_one.append(neptune_results_one[2][0])

        if _ % 10 == 0:
            with open(f'results_{_}.pkl', 'wb') as f:
                pickle.dump([avg_delay_criticality, avg_delay_vsvbp, avg_delay_poseidon_zero, avg_delay_poseidon_one, avg_delay_neptune_zero, avg_delay_neptune_one, avg_decision_time_criticality,
                            avg_decision_time_vsvbp, avg_decision_time_poseidon_zero,avg_decision_time_poseidon_one, avg_decision_time_neptune_zero,avg_decision_time_neptune_one, avg_cost_criticality, avg_cost_vsvbp, avg_cost_poseidon_zero,avg_cost_poseidon_one,avg_cost_neptune_zero, avg_cost_neptune_one], f)

            with open(f'results_cumulative_{_}.pkl', 'wb') as f:
                pickle.dump([avg_delays_criticality, avg_delays_vsvbp, avg_delays_poseidon_zero, avg_delays_poseidon_one, avg_delays_neptune_zero, avg_delays_neptune_one, avg_costs_criticality, avg_costs_vsvbp, avg_costs_poseidon_zero, avg_costs_poseidon_one, avg_costs_neptune_zero, avg_costs_neptune_one, avg_decision_times_criticality, avg_decision_times_vsvbp, avg_decision_times_poseidon_zero, avg_decision_times_poseidon_one, avg_decision_times_neptune_zero, avg_decision_times_neptune_one], f)

    avg_delay_criticality /= 150
    avg_delay_vsvbp /= 150
    avg_delay_poseidon_zero /= 150
    avg_delay_neptune_zero /= 150
    avg_delay_poseidon_one /= 150
    avg_delay_neptune_one /= 150

    avg_decision_time_criticality /= 150
    avg_decision_time_vsvbp /= 150
    avg_decision_time_poseidon_zero /= 150
    avg_decision_time_neptune_zero /= 150
    avg_decision_time_poseidon_one /= 150
    avg_decision_time_neptune_one /= 150

    avg_cost_criticality /= 150
    avg_cost_vsvbp /= 150
    avg_cost_poseidon_zero /= 150
    avg_cost_neptune_zero /= 150
    avg_cost_poseidon_one /= 150
    avg_cost_neptune_one /= 150

    plot_bar_delay(avg_delay_vsvbp, avg_delay_poseidon_zero, avg_delay_poseidon_one, avg_delay_neptune_zero,
                   avg_delay_neptune_one, avg_delay_criticality, "Criticality vs VSVBP vs Poseidon vs Neptune Average Delay")
    plot_bar_delay(avg_cost_vsvbp, avg_cost_poseidon_zero, avg_cost_poseidon_one, avg_cost_neptune_zero,
                   avg_cost_neptune_one, avg_cost_criticality, "Criticality vs VSVBP vs Poseidon vs Neptune Average Cost")
    plot_bar_delay(avg_decision_time_vsvbp, avg_decision_time_poseidon_zero, avg_decision_time_poseidon_one, avg_decision_time_neptune_zero,
                   avg_decision_time_neptune_one, avg_decision_time_criticality, "Criticality vs VSVBP vs Poseidon vs Neptune Average Decision Time")
else:
    with open("Results/alpha0.5_4/results_150.pkl", "rb") as f:
        result = pickle.load(f)

    avg_delay_criticality, avg_delay_vsvbp, avg_delay_poseidon_zero, avg_delay_poseidon_one, avg_delay_neptune_zero, avg_delay_neptune_one, avg_decision_time_criticality, avg_decision_time_vsvbp, avg_decision_time_poseidon_zero, avg_decision_time_poseidon_one, avg_decision_time_neptune_zero, avg_decision_time_neptune_one, avg_cost_criticality, avg_cost_vsvbp, avg_cost_poseidon_zero, avg_cost_poseidon_one, avg_cost_neptune_zero, avg_cost_neptune_one = result

    avg_delay_criticality /= 150
    avg_delay_vsvbp /= 150
    avg_delay_poseidon_zero /= 150
    avg_delay_neptune_zero /= 150
    avg_delay_poseidon_one /= 150
    avg_delay_neptune_one /= 150

    avg_decision_time_criticality /= 150
    avg_decision_time_vsvbp /= 150
    avg_decision_time_poseidon_zero /= 150
    avg_decision_time_neptune_zero /= 150
    avg_decision_time_poseidon_one /= 150
    avg_decision_time_neptune_one /= 150

    avg_cost_criticality /= 150
    avg_cost_vsvbp /= 150
    avg_cost_poseidon_zero /= 150
    avg_cost_neptune_zero /= 150
    avg_cost_poseidon_one /= 150
    avg_cost_neptune_one /= 150

    plot_bar_delay(avg_delay_vsvbp, avg_delay_poseidon_zero, avg_delay_poseidon_one, avg_delay_neptune_zero,
                   avg_delay_neptune_one, avg_delay_criticality, "CR-EUA vs VSVBP vs Poseidon vs Neptune Average Delay")
    plot_bar_delay(avg_cost_vsvbp, avg_cost_poseidon_zero, avg_cost_poseidon_one, avg_cost_neptune_zero,
                   avg_cost_neptune_one, avg_cost_criticality, "CR-EUA vs VSVBP vs Poseidon vs Neptune Average Cost")
    plot_bar_delay(avg_decision_time_vsvbp, avg_decision_time_poseidon_zero, avg_decision_time_poseidon_one, avg_decision_time_neptune_zero,
                   avg_decision_time_neptune_one, avg_decision_time_criticality, "CR-EUA vs VSVBP vs Poseidon vs Neptune Average Decision Time")



