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
                              [4, 10, 2, 2, 0]],
                             
        "workload_on_source_matrix": [[1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1]],
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
    alpha_values = [1,0.5,0]
    if input("Run the solvers and generate plots? (y/n): ").lower() == 'y':
        vsvbp_results = run_solver(payload,"VSVBP")

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
            
        with open("Results/vsvbp_results.pkl","wb") as f:
            pickle.dump(vsvbp_results,f)

        # with open("Results/workloads.pkl", "wb") as f:
        #     pickle.dump(workloads, f)

    elif os.path.exists("Results/poseidon_results.pkl") and os.path.exists("Results/neptune_results.pkl") and os.path.exists("Results/workloads.pkl"):
        with open("Results/poseidon_results.pkl", "rb") as f:
            poseidon_results = pickle.load(f)

        with open("Results/neptune_results.pkl", "rb") as f:
            neptune_results = pickle.load(f)

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
