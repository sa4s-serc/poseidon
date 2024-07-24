import requests
import pprint

input = {
    "with_db": False,
    "solver": {
        "type": "NeptuneMinDelayAndUtilization",
        "args": {"alpha": 1, "verbose": True, "soften_step1_sol": 1.3}
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

input["cores_matrix"] = [[1,1,1]] * len(input["function_names"])
input["workload_on_destination_matrix"] = [[1,1,1]] * len(input["function_names"])

response = requests.request(method='get', url="http://localhost:5000/", json=input)

pprint.pprint(response.json())
