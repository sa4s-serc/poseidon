from core import data_to_solver_input
from core.solvers import *
import matplotlib.pyplot as plt
import numpy as np


solver_string = "Poseidon"

payload = {
    "with_db": False,
    "solver": {
        "type": solver_string,
        "args": {"alpha": 0.7, "verbose": True, "soften_step1_sol": 1.3}
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
    "workload_on_source_matrix": [[100, 0, 0], [1, 0, 0]],
    "node_memories": [
        100, 100, 200
    ],
    "node_cores": [
        100, 50, 50
    ],
    "gpu_node_names": [
    ],
    "gpu_node_memories": [
    ],
    "function_names": [
        "ns/fn_1", "ns/fn_2"
    ],
    "function_memories": [
        5, 5
    ],
    "function_max_delays": [
        1000, 1000
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
        }
    },
    "actual_gpu_allocations": {
    },
        
    
}

payload["cores_matrix"] = [[1,1,1]] * len(payload["function_names"])
payload["workload_on_destination_matrix"] = [[1,1,1]] * len(payload["function_names"])

alpha_values = [0.90 +0.01*i for i in range(11)]

plt.figure()

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
    # Smooth the results
    alpha_res = np.convolve(poseidon_results[3], np.ones(10)/10, mode='valid')
    plt.plot(poseidon_results[3], label=f"alpha={alpha}")
    
plt.legend()
plt.xlabel("Iterations")
plt.ylabel("Reward Cumulative with alpha")
plt.savefig("alpha_plot.png")
plt.show()