import numpy as np
from .geo import *


def prepare_requests(data):
    # Amount of request received in time-slot
    for f in range(len(data.functions)):
        for i in range(len(data.nodes)):
            data.workload_matrix[f][i]=round(data.workload_matrix[f][i])
    data.requests_received = int(np.sum(data.workload_matrix))

    print("--------NODES_LEN [N]--------------",data.nodes)
    print("--------REQUESTS [R]---------------",data.requests_received)
    print("--------M_F_LEN [F]---------------", len(data.function_memory_matrix))

    # Set of requests within coverage of node i
    data.req_node_coverage = []  
    
    # Identifies which user sent the request [users_location x requests_received]
    data.matrix_size = (data.num_users, data.requests_received)
    data.req_by_user = np.zeros(data.matrix_size)
    data.row_indices = np.random.randint(0, data.matrix_size[0], data.matrix_size[1])
    data.req_by_user[data.row_indices, np.arange(data.matrix_size[1])] = 1
    
    # 1 if request r arrives to node i [N x R]
    data.loc_arrival_r=np.zeros([int(len(data.nodes)),int(data.requests_received)])

def prepare_req_distribution(data):
    data.req_distribution = np.zeros([int(len(data.functions)),int(data.requests_received)])
    r = 0
    while r<data.requests_received:
        for i in range(len(data.nodes)):
            for f in range(len(data.functions)):
                dif = data.workload_matrix[f][i]
                while dif >0 and r<data.requests_received:
                    data.req_distribution[f][r]=1
                    data.loc_arrival_r[i][r]=1
                    r=r+1
                    dif = dif-1

# COVERAGE REQUEST-NODE
def prepare_coverage(data):
    for i in range(len(data.nodes)):
        node_latitude = data.node_coords[i,0]
        node_longitude = data.node_coords[i,1]
        temp = []
        for r in range(data.requests_received):
            for u in range(data.num_users):
                if data.req_by_user[u][r]==1:
                    request_latitude = data.user_coords[u,0]
                    request_longitude = data.user_coords[u,1]
                    dist_geo = haversine(node_longitude, node_latitude, request_longitude, request_latitude)
                    print("dist_geo",dist_geo)
                    if dist_geo <= data.radius[0]:
                        temp.append(1)
                    else:
                        temp.append(0)
        
        data.req_node_coverage.append(temp)