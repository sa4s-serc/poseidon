def constrain_coverage(data, model, x):
    for r in range(data.requests_received):
            for j in range(len(data.nodes)):
                if data.req_node_coverage[j][r]==0:
                    model.Add(x[j, r]==0) 

def constrain_proximity(data, model, x):
    for i in range(len(data.nodes)):
        for r in range(data.requests_received):
            for f in range(len(data.functions)):
                for j in range(len(data.nodes)):
                    if int(data.node_delay_matrix[i][j])> int(data.max_delay_matrix[f]) and data.loc_arrival_r[i][r]==1 and data.req_distribution[f][r]==1:
                        model.Add(
                            x[j, r]==0
                        )         

def constrain_memory(data, model, c, y):
    for j in range(len(data.nodes)):
        suma_constraint = sum([c[f, j] * int(data.function_memory_matrix[f]) for f in range(len(data.functions))])
        model.Add(suma_constraint <= int(data.node_memory_matrix[j])*y[j])



def constrain_cpu(data, model, x, y):
    for j in range(len(data.nodes)):
            model.Add(
                sum([
                    x[j, r] * int(data.core_per_req_matrix[f,j])*int(data.req_distribution[f][r]) for r in range(data.requests_received) for f in range(len(data.functions))
                ]) <= int(data.node_cores_matrix[j])*y[j])
            
def constrain_request_handled(data, model, x):
    for r in range(data.requests_received):
        model.Add(sum([x[j, r] for j in range(len(data.nodes))]) <= 1)


def constrain_c_according_to_x(data, model, c, x):
    for f in range(len(data.functions)):
        for j in range(len(data.nodes)):
            model.Add(
                sum([
                    x[j, r]* int(data.req_distribution[f][r]) for r in range(data.requests_received)
                ]) <= c[f, j] * 1000)
            
def constrain_y_according_to_x(data, model, y, x):
    for j in range(len(data.nodes)):
        model.Add(
            sum([
                x[j, r] for r in range(data.requests_received)
            ]) <= y[j] * 1000) 
        
def constrain_amount_of_instances(data, model, c):
    for f in range(len(data.functions)):
        model.Add(
            sum([c[f,j] for j in range(len(data.nodes))])>= 1
        )

def add_hints(data, model, solver, x):
    for j in range(len(data.nodes)):
        for r in range(data.requests_received):
            model.AddHint(x[j,r], solver.Value(x[j,r]))


def constrain_previous_objective(data, model, solver, x):
    for r in range(data.requests_received):		  
            model.Add(sum([x[j, r] for j in range(len(data.nodes))])
                      == sum([solver.Value(x[j,r]) for j in range(len(data.nodes))]))
