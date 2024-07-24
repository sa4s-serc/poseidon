import numpy as np
from typing import List


class Data:

    def __init__(self, nodes: List[str] = None, functions: List[str] = None):
        self.nodes = nodes if nodes else []                     # List of node names
        self.functions = functions if functions else []         # List of function names
        self.node_memory_matrix: np.array = np.array([])        # Matrix of node memory
        self.function_memory_matrix: np.array = np.array([])    # Matrix of function memory
        self.node_delay_matrix: np.array = np.array([])         # Matrix of node delay
        self.workload_matrix: np.array = np.array([])           # Matrix of workload
        self.max_delay_matrix: np.array = np.array([])          # Matrix of max delay
        self.response_time_matrix: np.array = np.array([])      # Matrix of response time(Q + E) per function
        self.node_cores_matrix: np.array = np.array([])         # number of cores per node
        self.cores_matrix: np.array = np.array([])              # TODO: what is this?
        self.old_allocations_matrix: np.array = np.array([])    # old allocations
        self.core_per_req_matrix: np.array = np.array([])       # TODO: what is this?

        self.gpu_function_memory_matrix: np.array = np.array([])# Matrix of GPU function memory
        self.gpu_node_memory_matrix: np.array = np.array([])    # Matrix of GPU node memory
        self.prev_x = np.array([])                              # previous routing policy

        self.node_costs: np.array = np.array([])                # cost of each node (cost per unit time)
        self.node_budget: int = 0                               # budget for entire network(community)
