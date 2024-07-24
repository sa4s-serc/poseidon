def maximize_handled_requests(data, model, x):
    objective_max = []
    for j in range(len(data.nodes)):
        for r in range(data.requests_received):
            objective_max.append(x[j,r])
    model.Maximize(sum(objective_max))

def minimize_utilization(data, model, y):
    objective_min = []
    for j in range(len(data.nodes)):
        objective_min.append(y[j])
    model.Minimize(sum(objective_min))