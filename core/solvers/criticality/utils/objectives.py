import numpy as np


def maximize_handled_most_critical_requests(data, model, x):
    objective_max = []
    for j in range(len(data.nodes)):
        for r in range(data.requests_received):
            objective_max.append(x[j,r]*data.CR_matrix[r])
    model.Maximize(sum(objective_max))

def minimize_utilization(data, model, y):
    objective_min = []
    for j in range(len(data.nodes)):
        objective_min.append(y[j])
    model.Minimize(sum(objective_min))


def criticality_heuristic(data, log, S_active, y_j, c_fj, x_jr):
    log("Starting solving problem...")
    temp_req_index=0
    index_distribution = np.zeros(len(data.nodes))
    for j in range(len(data.node_cores_matrix)):
        index_distribution[j] = data.node_cores_matrix[j]

    while temp_req_index in data.requests_index:
        r = data.requests_index[temp_req_index]
        index_j = np.argsort(-index_distribution)#[::-1]
        loc=0
        print("---------------------REQUEST:  ",r," ----------------------------------")
        print("ORDER CHECK NODES:  ", index_j)
        print("CORE NODES:", data.node_cores_matrix)
        for j in index_j:
            for f in range(len(data.functions)):
                if all(S_active[f,:]==0) and loc==0 and data.req_distribution[f][r]:
                    #print(f" **** NO CONTAINERS FOR FUNCTION {f} **** ") 
                    # Option 1 there is no container for function f and there are active nodes we can check
                    active_j=np.where(y_j==1)[0][:]  
                    ordered_active_j = [x for x in index_j if x  in active_j]  
                    #print("Active nodes:  ", ordered_active_j)
                    for j_temp in ordered_active_j:
                        if data.req_node_coverage[j_temp][r]==1 and loc==0:
                            #print("✓ proximity constraint ")
                            if sum(data.function_memory_matrix[f_temp]*c_fj[f_temp,j_temp] for f_temp in range(len(data.functions)))+data.function_memory_matrix[f]<=data.node_memory_matrix[j_temp]: #memory constraint
                                #print("✓ memory constraint ")
                                if sum(x_jr[j_temp,r_temp]*data.core_per_req_matrix[f_temp,j_temp]*data.req_distribution[f_temp,r_temp] for f_temp in range(len(data.functions)) for r_temp in data.requests_index)+data.core_per_req_matrix[f,j_temp]*data.req_distribution[f][r]<= data.node_cores_matrix[j_temp]: #core constraint
                                    #print("✓ core constraint ")
                                    for i in range(len(data.nodes)):
                                        if data.node_delay_matrix[i,j_temp]<data.max_delay_matrix[f] and data.loc_arrival_r[i][r]==1 and data.req_distribution[f][r]==1: #delay constraint
                                            #print("✓ delay constraint, arrived to node: ", i)
                                            loc=1
                                            x_jr[j_temp][r]=1
                                            S_active[f][j_temp]=1
                                            y_j[j_temp]=1
                                            c_fj[f][j_temp]=1
                                            index_distribution[j_temp]=index_distribution[j_temp]-data.core_per_req_matrix[f,j_temp]*data.req_distribution[f,r]
                                            break
                    #Option 2: there is no container for function f and no active nodes (all y_j==0)
                    if j not in ordered_active_j and loc==0:
                        if data.req_node_coverage[j][r]==1:
                            if (sum(data.function_memory_matrix[f_temp]*c_fj[f_temp,j] for f_temp in range(len(data.functions)))+data.function_memory_matrix[f])<=(data.node_memory_matrix[j]): # memory constraint
                                if (sum(x_jr[j,r_temp]*data.core_per_req_matrix[f_temp,j]*data.req_distribution[f_temp,r_temp] for f_temp in range(len(data.functions)) for r_temp in data.requests_index)+data.core_per_req_matrix[f,j]*data.req_distribution[f][r])<=(data.node_cores_matrix[j]) : # core constraint
                                    for i in range(len(data.nodes)):
                                        if data.node_delay_matrix[i,j]<data.max_delay_matrix[f] and data.loc_arrival_r[i][r]==1 and data.req_distribution[f][r]==1: # delay constraint
                                            loc=1
                                            x_jr[j][r]=1
                                            S_active[f][j]=1
                                            y_j[j]=1
                                            index_distribution[j]=index_distribution[j]-data.core_per_req_matrix[f,j]*data.req_distribution[f,r]
                                            c_fj[f][j]=1
                                            break
                if any(S_active[f,:]==1) and loc==0 and data.req_distribution[f][r]==1: 
                    #Option 3: there is already a container for function f in a node, so it checks if request can be allocated to this node
                    active_loc_f=np.where(S_active[f,:]==1)[0][:]
                    ordered_active_loc_f = [x for x in index_j if x  in active_loc_f]
                    for j_temp_active in ordered_active_loc_f:
                        if data.req_node_coverage[j_temp_active][r]==1 and loc==0: # Proximity constraint:
                            if sum(data.function_memory_matrix[f_temp]*c_fj[f_temp,j_temp_active] for f_temp in range(len(data.functions)))<=data.node_memory_matrix[j_temp_active]: # memory constraint
                                if sum(x_jr[j_temp_active,r_temp]*data.core_per_req_matrix[f_temp,j_temp_active]*data.req_distribution[f_temp,r_temp] for f_temp in range(len(data.functions)) for r_temp in data.requests_index)+data.core_per_req_matrix[f,j_temp_active]*data.req_distribution[f][r]<= data.node_cores_matrix[j_temp_active]: #core constraint
                                    for i in range(len(data.nodes)):
                                        if data.node_delay_matrix[i,j_temp_active]<data.max_delay_matrix[f] and data.loc_arrival_r[i][r]==1 and data.req_distribution[f][r]==1: # delay constraint
                                            loc=1
                                            x_jr[j_temp_active][r]=1
                                            index_distribution[j_temp_active]=index_distribution[j_temp_active]-data.core_per_req_matrix[f,j_temp_active]*data.req_distribution[f,r]
                                            break
                    #Option 4: Needs to deploy a container for function f in one of the active nodes
                    if data.req_node_coverage[j][r]==1 and (j not in ordered_active_loc_f) and loc==0 and any(y_j==1): #Proximity constraint:
                        active_j_f=np.where(y_j==1)[0][:]  
                        ordered_active_j_f = [x for x in index_j if x  in active_j_f]
                        for j_temp_f in ordered_active_j_f:
                            if data.req_node_coverage[j_temp_f][r]==1 and loc==0:
                                if sum(data.function_memory_matrix[f_temp]*c_fj[f_temp,j_temp_f] for f_temp in range(len(data.functions)))+data.function_memory_matrix[f]<=data.node_memory_matrix[j_temp_f]: # memory constraint
                                    if sum(x_jr[j_temp_f,r_temp]*data.core_per_req_matrix[f_temp,j_temp_f]*data.req_distribution[f_temp,r_temp] for f_temp in range(len(data.functions)) for r_temp in data.requests_index)+data.core_per_req_matrix[f,j_temp_f]*data.req_distribution[f][r]<= data.node_cores_matrix[j_temp_f]: # core constraint
                                        for i in range(len(data.nodes)):
                                            if data.node_delay_matrix[i,j_temp_f]<data.max_delay_matrix[f] and data.loc_arrival_r[i][r]==1 and data.req_distribution[f][r]==1: # delay constraint
                                                loc=1
                                                x_jr[j_temp_f][r]=1
                                                S_active[f][j_temp_f]=1
                                                y_j[j_temp_f]=1
                                                c_fj[f][j_temp_f]=1
                                                index_distribution[j_temp_f]=index_distribution[j_temp_f]-data.core_per_req_matrix[f,j_temp_f]*data.req_distribution[f,r]
                                                break
                        #Option 5: when need to turn on a new node for a given function f
                        if j not in ordered_active_j_f and loc==0:
                            if data.req_node_coverage[j][r]==1:
                                if sum(data.function_memory_matrix[f_temp]*c_fj[f_temp,j] for f_temp in range(len(data.functions)))+data.function_memory_matrix[f]<=(data.node_memory_matrix[j]): # memory constraint
                                    if sum(x_jr[j,r_temp]*data.core_per_req_matrix[f_temp,j]*data.req_distribution[f_temp,r_temp] for f_temp in range(len(data.functions)) for r_temp in data.requests_index)+data.core_per_req_matrix[f,j]*data.req_distribution[f][r]<= (data.node_cores_matrix[j]): # core constraint
                                        for i in range(len(data.nodes)):
                                            if data.node_delay_matrix[i,j]<data.max_delay_matrix[f] and data.loc_arrival_r[i][r]==1 and data.req_distribution[f][r]==1: # delay constraint
                                                loc=1
                                                x_jr[j][r]=1
                                                S_active[f][j]=1
                                                y_j[j]=1
                                                index_distribution[j]=index_distribution[j]-data.core_per_req_matrix[f,j]*data.req_distribution[f,r]
                                                c_fj[f][j]=1
                                                break

        print("Core capacity: ", index_distribution)
        print("Active nodes: " ,y_j)
        temp_req_index=temp_req_index+1
    
    for f in range (len(data.functions)):
        if all(c_fj[f,:]==0) and any(y_j==1): # use nodes that are already being used
            rand_node= np.where(y_j==1)[0][:]  
            for t in rand_node:
                if sum(data.function_memory_matrix[f_temp]*c_fj[f_temp,t] for f_temp in range(len(data.functions)))+data.function_memory_matrix[f]<=(data.node_memory_matrix[t]*y_j[t]):
                    c_fj[f][rand_node]=1
                    S_active[f][rand_node]=1
                    y_j[rand_node]=1

        if all(c_fj[f,:]==0): # when there are no requests
            rand_node= np.random.choice(range(len(data.nodes)))
            c_fj[f][rand_node]=1
            S_active[f][rand_node]=1
            y_j[rand_node]=1
