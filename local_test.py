from core import data_to_solver_input
from core.solvers import *
import matplotlib.pyplot as plt

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

solver_string = "Poseidon"

payload = {
    "with_db": False,
    "solver": {
        "type": solver_string,
        "args": {"alpha": 0.9, "verbose": True, "soften_step1_sol": 1.3}
    },
    "workload_coeff": 1,
    "community": "community-test",
    "namespace": "namespace-test",
    "node_names": [
        "node_a", "node_b", "node_c"
    ],
    "node_delay_matrix": [[0, 3, 2],
                          [3, 0, 4],
                          [2, 4, 0]],
    "workload_on_source_matrix": [[100, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]],
    "node_memories": [
        100, 100, 200
    ],
    "node_cores": [
        50, 50, 50
    ],
    "gpu_node_names": [
    ],
    "gpu_node_memories": [
    ],
    "function_names": [
        "ns/fn_1", "ns/fn_2","ns/fn_3","ns/fn_4"
    ],
    "function_memories": [
        5, 5 , 10 ,15
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
        },
        "ns/fn_2": {
            "node_a": True,
            "node_b": True,
            "node_c": True,
        },
        "ns/fn_3": {
            "node_a": True,
            "node_b": True,
            "node_c": True,
        },
        "ns/fn_4": {
            "node_a": True,
            "node_b": True,
            "node_c": True,
        },
    },
    "actual_gpu_allocations": {
    },
        
    
}

payload["cores_matrix"] = [[1,1,1]] * len(payload["function_names"])
payload["workload_on_destination_matrix"] = [[1,1,1]] * len(payload["function_names"])

solver = payload.get("solver", {'type': solver_string})
solver_type = solver.get("type")
solver_args = solver.get("args", {})
with_db = payload.get("with_db", True)

solver = eval(solver_type)(**solver_args)
solver.load_data(data_to_solver_input(payload, with_db=with_db, workload_coeff=payload.get("workload_coeff", 1))) 
solver.solve()


alpha_values = [0.92,0.94,0.96,0.98,1]

plt.figure()
plt.subplot(1, 2, 1)

window_size = 5  # Adjust window size for smoothing

for alpha in alpha_values:
    payload["solver"]["args"]["alpha"] = alpha
    solver = payload.get("solver", {'type': solver_string})
    solver_type = solver.get("type")
    solver_args = solver.get("args", {})
    with_db = payload.get("with_db", True)

    solver = eval(solver_type)(**solver_args)
    solver.load_data(data_to_solver_input(payload, with_db=with_db, workload_coeff=payload.get("workload_coeff", 1))) 
    solver.solve()
    
    poseidon_results = solver.results()
    smoothed_poseidon_delay = moving_average(poseidon_results[0], window_size)
    plt.plot(smoothed_poseidon_delay, label=f"alpha={alpha}")

# Switch payload solver
payload["solver"]["type"] = "NeptuneMinDelayAndUtilization"

# Get Neptune results
solver = payload.get("solver", {'type': solver_string})
solver_type = solver.get("type")
solver_args = solver.get("args", {})
with_db = payload.get("with_db", False)

solver = eval(solver_type)(**solver_args)
solver.load_data(data_to_solver_input(payload, with_db=with_db, workload_coeff=payload.get("workload_coeff", 1)))
solver.solve()

neptune_results = solver.results()

# Plotting delay
smoothed_neptune_delay = moving_average(neptune_results[0], window_size)
smoothed_neptune_step2_delay = moving_average(neptune_results[3], window_size)
plt.plot(smoothed_neptune_delay, label="Neptune")
plt.plot(smoothed_neptune_step2_delay, label="Neptune Step 2")
plt.legend()
plt.title("Comparison of Poseidon and Neptune - Delay")
plt.xlabel("Workload iterations")
plt.ylabel("Delay")

plt.subplot(1, 2, 2)

# Plotting inference time
smoothed_poseidon_inference = moving_average(poseidon_results[2], window_size)
smoothed_neptune_inference = moving_average(neptune_results[2], window_size)
plt.plot(smoothed_poseidon_inference, label="Poseidon")
plt.plot(smoothed_neptune_inference, label="Neptune")
plt.legend()
plt.title("Comparison of Poseidon and Neptune - Inference Time")
plt.xlabel("Workload iterations")
plt.ylabel("Inference Time")

plt.tight_layout()
plt.savefig("poseidon_vs_neptune.png")
plt.show()






