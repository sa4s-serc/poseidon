import numpy as np


def minimize_network_delay(data, objective, x):
    for f in range(len(data.functions)):
        for i in range(len(data.nodes)):
            for j in range(len(data.nodes)):
                objective.SetCoefficient(
                    x[i, f, j], float(data.node_delay_matrix[i, j] * data.workload_matrix[f, i])
                )
    objective.SetMinimization()


def maximize_handled_requests(data, objective, x):
    for f in range(len(data.functions)):
        for i in range(len(data.nodes)):
            for j in range(len(data.nodes)):
                objective.SetCoefficient(
                    x[i, f, j], float(data.workload_matrix[f, i])
                )
    objective.SetMaximization()


def minimize_node_utilization(data, objective, n):
    for i in range(len(data.nodes)):
        objective.SetCoefficient(n[i], 1)
    objective.SetMinimization()


def minimize_node_delay_and_utilization(data, objective, n, x, alpha):
    # offset_util = -alpha * 1 / (len(data.nodes))
    for i in range(len(data.nodes)):
        objective.SetCoefficient(n[i], float(alpha / (len(data.nodes))))

    total_workload = np.sum(data.workload_matrix)
    if total_workload:
        
        # compute max
        max_workload_delay = 0
        for f in range(len(data.functions)):
            max_func_delay = data.max_delay_matrix[f]
            for i in range(len(data.nodes)):
                max_node_delay = max([delay for delay in data.node_delay_matrix[i] if delay <= max_func_delay])
                workload = data.workload_matrix[f, i]
                max_workload_delay += workload * max_node_delay
        # add terms to objective function
        for f in range(len(data.functions)):
            for i in range(len(data.nodes)):
                workload = data.workload_matrix[f, i]
                for j in range(len(data.nodes)):
                    delay = data.node_delay_matrix[i, j]
                    objective.SetCoefficient(x[i, f, j], float((1 - alpha) * workload * delay / max_workload_delay))
    
    objective.SetMinimization()


def minimize_disruption(data, objective, moved_from, moved_to, allocated, deallocated):
    w = np.ma.size(data.old_allocations_matrix)
    for f in range(len(data.functions)):
        for j in range(len(data.nodes)):
            objective.SetCoefficient(moved_from[f, j], w)
            objective.SetCoefficient(moved_to[f, j], w)
    objective.SetCoefficient(allocated, w - 1)
    objective.SetCoefficient(deallocated, w + 1)
    objective.SetMinimization()
