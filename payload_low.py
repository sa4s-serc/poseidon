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
    