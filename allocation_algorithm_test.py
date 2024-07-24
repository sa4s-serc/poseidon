import pprint

import requests

for solver_type in [
    "NeptuneMinDelayAndUtilization",
    # "NeptuneMinDelay",
    # "NeptuneMinUtilization",
    # "VSVBP",
    # "Criticality",
    # "CriticalityHeuristic",
    # "MCF"
]:
    # solver_type = "NeptuneMinDelayAndUtilization"

    inputs = [
        # Simplest input
        # One node, one function
        # No function allocated
        {
            "solver": {
                "type": solver_type,
                "args": {"alpha": 0.0, "verbose": False}
            },
            "with_db": False,
            "cpu_coeff": 1,
            "community": "community-test",
            "namespace": "namespace-test",
            "node_names": [
                "node_a"
            ],
            "node_memories": [
                100
            ],
            "node_cores": [
                100
            ],
            "gpu_node_names": [],
            "gpu_node_memories": [],
            "function_names": [
                "ns/fn_1"
            ],
            "function_memories": [
                10
            ],
            "function_max_delays": [
                100
            ],
            "gpu_function_names": [],
            "gpu_function_memories": [],
            "actual_cpu_allocations": {
                "ns/fn_1": {},
            },
            "actual_gpu_allocations": {
            }
        },
        # Simplest input
        # One node, one function
        # The function was already allocated
        {
            "solver": {
                "type": solver_type,
                "args": {"alpha": 0.0, "verbose": False}
            },
            "community": "community-test",
            "namespace": "namespace-test",
            "node_names": [
                "node_a"
            ],
            "node_memories": [
                100
            ],
            "node_cores": [
                100
            ],
            "gpu_node_names": [],
            "gpu_node_memories": [],
            "function_names": [
                "ns/fn_1"
            ],
            "function_memories": [
                10
            ],
            "function_max_delays": [
                100
            ],
            "gpu_function_names": [],
            "gpu_function_memories": [],
            "actual_cpu_allocations": {
                "ns/fn_1": {
                    "node_a": True,
                },
            },
            "actual_gpu_allocations": {
            }
        },
        # Simplest input
        # One node, two functions
        # Both of them were not allocated
        {
            "solver": {
                "type": solver_type,
                "args": {"alpha": 0.0, "verbose": False}
            },
            "community": "community-test",
            "namespace": "namespace-test",
            "node_names": [
                "node_a"
            ],
            "node_memories": [
                100
            ],
            "node_cores": [
                100
            ],
            "gpu_node_names": [],
            "gpu_node_memories": [],
            "function_names": [
                "ns/fn_1", "ns/fn_2"
            ],
            "function_memories": [
                10, 10
            ],
            "function_max_delays": [
                100, 100
            ],
            "gpu_function_names": [],
            "gpu_function_memories": [],
            "actual_cpu_allocations": {
            },
            "actual_gpu_allocations": {
            }
        },
        # Simplest input
        # One node, two functions
        # Only one of them is allocated
        {
            "solver": {
                "type": solver_type,
                "args": {"alpha": 0.0, "verbose": False}
            },
            "community": "community-test",
            "namespace": "namespace-test",
            "node_names": [
                "node_a"
            ],
            "node_memories": [
                100
            ],
            "node_cores": [
                100
            ],
            "gpu_node_names": [],
            "gpu_node_memories": [],
            "function_names": [
                "ns/fn_1", "ns/fn_2"
            ],
            "function_memories": [
                10, 10
            ],
            "function_max_delays": [
                100, 100
            ],
            "gpu_function_names": [],
            "gpu_function_memories": [],
            "actual_cpu_allocations": {
                "ns/fn_1": {
                    "node_a": True,
                },
            },
            "actual_gpu_allocations": {
            }
        },
        # Simplest input
        # One node, two functions
        # Both of them were allocated
        {
            "solver": {
                "type": solver_type,
                "args": {"alpha": 0.0, "verbose": False}
            },
            "community": "community-test",
            "namespace": "namespace-test",
            "node_names": [
                "node_a"
            ],
            "node_memories": [
                100
            ],
            "node_cores": [
                100
            ],
            "gpu_node_names": [],
            "gpu_node_memories": [],
            "function_names": [
                "ns/fn_1", "ns/fn_2"
            ],
            "function_memories": [
                10, 10
            ],
            "function_max_delays": [
                100, 100
            ],
            "gpu_function_names": [],
            "gpu_function_memories": [],
            "actual_cpu_allocations": {
                "ns/fn_1": {
                    "node_a": True,
                },
                "ns/fn_2": {
                    "node_a": True,
                },
            },
            "actual_gpu_allocations": {
            }
        },
        # Many node, many functions
        # None of them were allocated
        {
            "solver": {
                "type": solver_type,
                "args": {"alpha": 0.0, "verbose": False}
            },
            "community": "community-test",
            "namespace": "namespace-test",
            "node_names": [f"node_{i}" for i in range(20)],
            "node_memories": [100 for i in range(20)],
            "node_cores": [100 for i in range(20)],
            "gpu_node_names": [],
            "gpu_node_memories": [],
            "function_names": [f"ns/fn_{i}" for i in range(5)],
            "function_memories": [10 for i in range(5)],
            "function_max_delays": [100 for i in range(5)],
            "gpu_function_names": [],
            "gpu_function_memories": [],
            "actual_cpu_allocations": {
            },
            "actual_gpu_allocations": {
            }
        },
        # Many node, many functions
        # All of them were allocated
        {
            "solver": {
                "type": solver_type,
                "args": {"alpha": 0.0, "verbose": False}
            },
            "community": "community-test",
            "namespace": "namespace-test",
            "node_names": [f"node_{i}" for i in range(20)],
            "node_memories": [100 for i in range(20)],
            "node_cores": [100 for i in range(20)],
            "gpu_node_names": [],
            "gpu_node_memories": [],
            "function_names": [f"ns/fn_{i}" for i in range(5)],
            "function_memories": [10 for i in range(5)],
            "function_max_delays": [100 for i in range(5)],
            "gpu_function_names": [],
            "gpu_function_memories": [],
            "actual_cpu_allocations": {
                "ns/fn_0": {
                    "node_1": True,
                },
                "ns/fn_1": {
                    "node_1": True,
                },
                "ns/fn_2": {
                    "node_1": True,
                },
                "ns/fn_3": {
                    "node_1": True,
                },
                "ns/fn_4": {
                    "node_1": True,
                },
            },
            "actual_gpu_allocations": {
            }
        },
    ]

    for input_request in inputs:
        response = requests.request(method='get', url="http://localhost:5000/", json=input_request)
        pprint.pprint(response.json())
