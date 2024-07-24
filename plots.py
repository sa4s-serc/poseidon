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


def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')


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


def plot_delay_inference_reward(poseidon_results, neptune_results, alpha_values):
    fig, axes = plt.subplots(2, 1, figsize=(15, 20))
    window_size = 5

    metrics = ['Reward', 'Delay', 'Inference Time',]

    for i, metric in enumerate(metrics):
        ax = axes[i-1]
        if i==0:
            continue
        else:
            for alpha in alpha_values:
                smoothed_poseidon = moving_average(
                    poseidon_results[alpha][i], window_size)
                ax.plot(smoothed_poseidon, label=f"Poseidon (α={alpha})")
                smoothed_neptune = moving_average(neptune_results[alpha][i], window_size)
                ax.plot(smoothed_neptune, label=f"Neptune (α={alpha})", linestyle='--', linewidth=2)

            ax.set_title(f"Comparison of Poseidon and Neptune - {metric}")
            ax.set_xlabel("Workload iterations")
            ax.set_ylabel(metric)
            ax.legend()
            sns.despine()

    plt.tight_layout()
    plt.savefig("delay_inference_reward_comparison.png", dpi=300)
    plt.close()


def plot_distributions(poseidon_results, neptune_results, alpha_values):
    metrics = ['Reward', 'Delay', 'Inference Time',]
    
    
    for metric_index, metric in enumerate(metrics):
        if metric_index == 0:
            continue
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

        data = []
        labels = []
        for alpha in alpha_values:
            data.append(poseidon_results[alpha][metric_index])
            labels.append(f"Poseidon (α={alpha})")
            data.append(neptune_results[alpha][metric_index])
            labels.append(f"Neptune (α={alpha})")
        # labels.append("Neptune")

        sns.boxplot(data=data, ax=ax1)
        sns.violinplot(data=data, ax=ax2)

        for ax in [ax1, ax2]:
            ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.set_title(
                f"{metric} Distribution - {'Box' if ax == ax1 else 'Violin'} Plot")
            ax.set_ylabel(metric)

        plt.tight_layout()
        plt.savefig(
            f"{metric.lower().replace(' ', '_')}_distribution.png", dpi=300)
        plt.close()


def plot_heatmap(poseidon_results, neptune_results, alpha_values):
    metrics = ['Delay', 'Inference Time', 'Reward']
    for metric_index, metric in enumerate(metrics):
        # Prepare data with consistent shape
        max_length = max(
            max(len(poseidon_results[alpha][metric_index]) for alpha in alpha_values),
            len(neptune_results[metric_index])
        )
        
        data = []
        for alpha in alpha_values:
            result = poseidon_results[alpha][metric_index]
            # Pad shorter lists with NaN
            padded_result = result + [np.nan] * (max_length - len(result))
            data.append(padded_result)
        
        neptune_result = neptune_results[metric_index]
        padded_neptune = neptune_result + [np.nan] * (max_length - len(neptune_result))
        data.append(padded_neptune)
        
        # Convert to numpy array
        data_array = np.array(data)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(data_array, ax=ax, cmap='viridis', cbar_kws={'label': metric})
        
        # Set y-axis labels
        labels = [f"Poseidon (α={alpha})" for alpha in alpha_values] + ["Neptune"]
        ax.set_yticklabels(labels, rotation=0)
        
        # Set x-axis ticks and labels
        num_ticks = min(10, data_array.shape[1])
        tick_locations = np.linspace(0, data_array.shape[1] - 1, num_ticks).astype(int)
        ax.set_xticks(tick_locations)
        ax.set_xticklabels(tick_locations, rotation=45, ha='right')
        
        ax.set_title(f"{metric} Heatmap Comparison")
        ax.set_xlabel("Workload iterations")
        
        plt.tight_layout()
        plt.savefig(f"{metric.lower().replace(' ', '_')}_heatmap.png", dpi=300)
        plt.close()


def plot_workload_distribution(workloads):
    """
    Visualize workload distribution over time.

    :param workloads: List of tuples (workload_matrix, timestamp)
    """
    func_num = len(workloads[0][0])
    nodes_num = len(workloads[0][0][0])
    timestamps = [w[1] for w in workloads]

    # Prepare data for heatmap
    heatmap_data = np.zeros((func_num * nodes_num, len(workloads)))
    for i, (workload_matrix, _) in enumerate(workloads):
        heatmap_data[:, i] = np.array(workload_matrix).flatten()

    # Create heatmap
    fig, ax = plt.subplots(figsize=(15, 10))
    sns.heatmap(heatmap_data, ax=ax, cmap='YlOrRd',
                cbar_kws={'label': 'Workload'})

    # Set labels
    ax.set_xlabel('Timestamp')
    ax.set_ylabel('Function-Node Pair')
    ax.set_title('Workload Distribution Over Time')

    # Set x-axis ticks to show timestamps
    num_ticks = min(10, len(timestamps))  # Limit to 10 ticks for readability
    tick_indices = np.linspace(0, len(timestamps) - 1, num_ticks, dtype=int)
    ax.set_xticks(tick_indices)
    ax.set_xticklabels([timestamps[i]
                       for i in tick_indices], rotation=45, ha='right')

    # Set y-axis ticks to show function-node pairs
    y_labels = [
        f'F{f+1}-N{n+1}' for f in range(func_num) for n in range(nodes_num)]
    ax.set_yticks(np.arange(func_num * nodes_num) + 0.5)
    ax.set_yticklabels(y_labels)

    plt.tight_layout()
    plt.savefig("workload_distribution_heatmap.png", dpi=300)
    plt.close()

    # Line plot for total workload per function over time
    fig, ax = plt.subplots(figsize=(15, 10))
    for f in range(func_num):
        function_workload = [sum(w[0][f]) for w in workloads]
        ax.plot(timestamps, function_workload, label=f'Function {f+1}')

    ax.set_xlabel('Timestamp')
    ax.set_ylabel('Total Workload')
    ax.set_title('Total Workload per Function Over Time')
    ax.legend()

    plt.tight_layout()
    plt.savefig("workload_per_function_line_plot.png", dpi=300)
    plt.close()


def plot_3d_workload_surface(workloads):
    """
    Create a 3D surface plot of workload distribution over time and functions.

    :param workloads: List of tuples (workload_matrix, timestamp)
    """
    func_num = len(workloads[0][0])
    timestamps = [w[1] for w in workloads]

    # Prepare data for 3D surface
    X, Y = np.meshgrid(range(len(timestamps)), range(func_num))
    Z = np.array([[sum(w[0][f]) for w in workloads] for f in range(func_num)])

    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Create custom colormap
    colors = ['#FFA07A', '#FA8072', '#E9967A', '#F08080',
              '#CD5C5C', '#DC143C', '#B22222', '#8B0000']
    n_bins = len(colors)
    cmap = LinearSegmentedColormap.from_list('custom', colors, N=n_bins)

    surf = ax.plot_surface(X, Y, Z, cmap=cmap, edgecolor='none', alpha=0.8)

    ax.set_xlabel('Timestamp')
    ax.set_ylabel('Function')
    ax.set_zlabel('Total Workload')
    ax.set_title('3D Workload Distribution Over Time and Functions')

    # Customize x-axis ticks
    num_ticks = min(5, len(timestamps))
    tick_indices = np.linspace(0, len(timestamps) - 1, num_ticks, dtype=int)
    ax.set_xticks(tick_indices)
    ax.set_xticklabels([timestamps[i]
                       for i in tick_indices], rotation=45, ha='right')

    # Add colorbar
    fig.colorbar(surf, ax=ax, label='Workload Intensity', pad=0.1)

    plt.tight_layout()
    plt.savefig("3d_workload_surface.png", dpi=300)
    plt.close()


def plot_workload_correlation_matrix(workloads):
    """
    Create a correlation matrix heatmap of workloads between functions.

    :param workloads: List of tuples (workload_matrix, timestamp)
    """
    func_num = len(workloads[0][0])

    # Prepare data for correlation matrix
    func_workloads = np.array(
        [[sum(w[0][f]) for w in workloads] for f in range(func_num)])

    # Calculate correlation matrix
    corr_matrix = np.corrcoef(func_workloads)

    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm',
                vmin=-1, vmax=1, center=0, ax=ax)

    ax.set_title('Correlation Matrix of Workloads Between Functions')
    ax.set_xticklabels([f'F{i+1}' for i in range(func_num)])
    ax.set_yticklabels([f'F{i+1}' for i in range(func_num)])

    plt.tight_layout()
    plt.savefig("workload_correlation_matrix.png", dpi=300)
    plt.close()


def plot_solver_performance_radar(poseidon_results, neptune_results, alpha_values):
    """
    Create a radar plot comparing solver performances across multiple metrics.

    :param poseidon_results: Dict of Poseidon results for different alpha values
    :param neptune_results: Neptune results
    :param alpha_values: List of alpha values used for Poseidon
    """
    metrics = ['Avg Delay', 'Avg Inference Time',
               'Avg Reward', 'Std Delay', 'Std Inference Time']
    # metrics = ['Reward', 'Delay', 'Inference Time', 'Workload',]

    # Prepare data
    neptune_stats = [
        np.mean(neptune_results[1]),  # Avg Delay
        np.mean(neptune_results[2]),  # Avg Inference Time
        np.mean(neptune_results[0]),  # Avg Reward (assuming index 2 is reward)
        np.std(neptune_results[1]),   # Std Delay
        np.std(neptune_results[2])    # Std Inference Time
    ]

    poseidon_stats = {alpha: [
        np.mean(poseidon_results[alpha][1]),  # Avg Delay
        np.mean(poseidon_results[alpha][2]),  # Avg Inference Time
        # Avg Reward (assuming index 2 is reward)
        np.mean(poseidon_results[alpha][0]),
        np.std(poseidon_results[alpha][1]),   # Std Delay
        np.std(poseidon_results[alpha][2])    # Std Inference Time
    ] for alpha in alpha_values}

    # Plotting
    num_metrics = len(metrics)
    angles = np.linspace(0, 2*np.pi, num_metrics, endpoint=False).tolist()
    angles += angles[:1]  # complete the polygon

    fig, ax = plt.subplots(
        figsize=(12, 10), subplot_kw=dict(projection='polar'))

    # Plot Neptune
    neptune_stats += neptune_stats[:1]  # complete the polygon
    ax.plot(angles, neptune_stats, 'o-', linewidth=2, label='Neptune')
    ax.fill(angles, neptune_stats, alpha=0.25)

    # Plot Poseidon for each alpha
    for alpha in alpha_values:
        stats = poseidon_stats[alpha] + \
            poseidon_stats[alpha][:1]  # complete the polygon
        ax.plot(angles, stats, 'o-', linewidth=2,
                label=f'Poseidon (α={alpha})')
        ax.fill(angles, stats, alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_title('Solver Performance Comparison')
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

    plt.tight_layout()
    plt.savefig("solver_performance_radar.png", dpi=300)
    plt.close()

def plot_cost_comparison(poseidon_results, neptune_results, alpha_values):
    # print(neptune_results[4])
    fig, ax = plt.subplots(figsize=(12, 8))
    window_size = 5

    for alpha in alpha_values:
        smoothed_poseidon = moving_average(
            poseidon_results[alpha][4], window_size)
        ax.plot(smoothed_poseidon, label=f"Poseidon (α={alpha})")

        smoothed_neptune = moving_average(neptune_results[alpha][4], window_size)
        ax.plot(smoothed_neptune, label=f"Neptune (α={alpha})", linestyle='--', linewidth=2)

    ax.set_title("Comparison of Poseidon and Neptune - Cost")
    ax.set_xlabel("Workload iterations")
    ax.set_ylabel("Cost")
    ax.legend()
    sns.despine()

    plt.tight_layout()
    plt.savefig("cost_comparison.png", dpi=300)
    plt.close()

if __name__ == "__main__":
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
                              [4, 10, 2, 2, 0],],
                             
        "workload_on_source_matrix": [[100, 0, 0], [1, 0, 0],[100, 0, 0], [1, 0, 0]],
        "node_memories": [
            100, 100, 200, 50, 500
        ],
        "node_cores": [
            50, 50, 50, 25, 100
        ],
        "gpu_node_names": [
        ],
        "gpu_node_memories": [
        ],
        "function_names": [
            "ns/fn_1", "ns/fn_2", "ns/fn_3", "ns/fn_4"
        ],
        "function_memories": [
            50, 10, 10, 10
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
   
    # alpha_values = [0.92, 0.94, 0.96, 0.98, 1]
    alpha_values = [0,1]
    if input("Run the solvers and generate plots? (y/n): ").lower() == 'y':

        poseidon_results = {alpha: run_solver(
            payload, "Poseidon", alpha) for alpha in alpha_values}

        payload["solver"]["type"] = "NeptuneMinDelayAndUtilization"
        
        neptune_results = run_solver(payload, "NeptuneMinDelayAndUtilization")
        
        neptune_results = {alpha: run_solver(
            payload, "NeptuneMinDelayAndUtilization", alpha) for alpha in alpha_values}


        # Save the results to file
        with open("Results/poseidon_results.pkl", "wb") as f:
            pickle.dump(poseidon_results, f)

        with open("Results/neptune_results.pkl", "wb") as f:
            pickle.dump(neptune_results, f)

        # with open("Results/workloads.pkl", "wb") as f:
        #     pickle.dump(workloads, f)

    elif os.path.exists("Results/poseidon_results.pkl") and os.path.exists("Results/neptune_results.pkl") and os.path.exists("Results/workloads.pkl"):
        with open("Results/run_test_medium/poseidon_results.pkl", "rb") as f:
            poseidon_results = pickle.load(f)

        with open("Results/run_test_medium/neptune_results.pkl", "rb") as f:
            neptune_results = pickle.load(f)

        inference_poseidon_zero = poseidon_results[0][2]
        inference_poseidon_one = poseidon_results[1][2]
        inference_neptune_zero = neptune_results[0][2]
        inference_neptune_one = neptune_results[1][2]
        
        #print averages
        print("Poseidon Zero: ", np.mean(inference_poseidon_zero))
        print("Poseidon One: ", np.mean(inference_poseidon_one))
        print("Neptune Zero: ", np.mean(inference_neptune_zero))
        print("Neptune One: ", np.mean(inference_neptune_one))

        # with open("Results/workloads.pkl", "rb") as f:
        #     workloads = pickle.load(f)

    else:
        print("No saved results found. Please run the solvers first.")
        exit(1)

    # Generate plots
    plot_delay_inference_reward(
        poseidon_results, neptune_results, alpha_values)
    plot_distributions(poseidon_results, neptune_results, alpha_values)
    # plot_heatmap(poseidon_results, neptune_results, alpha_values)
    # plot_workload_distribution(workloads)
    # plot_3d_workload_surface(workloads)
    # plot_workload_correlation_matrix(workloads)
    # plot_solver_performance_radar(
    #     poseidon_results, neptune_results, alpha_values)
    plot_cost_comparison(poseidon_results, neptune_results, alpha_values)

    print("All plots have been generated and saved.")
