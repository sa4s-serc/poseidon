import random
import numpy as np
from ...vsvbp.utils.geo import haversine
from .geo import *

def prepare_aux_vars(data, danger_radius_km):
    data.req_by_user=[]
    data.CR_matrix = []
    data.D = (round(random.uniform(-0.010, 0.010),3),round(random.uniform(-0.010, 0.010),3)) 
    data.D_rad = danger_radius_km # Influence range of danger source D (radius) in km
    data.U_per = 0  # Perception range of individual uj in km
    data.T_1 = 0 # Starting point of period
    data.T_2 = 1 # Ending point of period
    data.T = data.T_2-data.T_1 # Total interval time
    data.lambd = 0.5 # Severity of the stimulus event
    data.nej = 0 # Emotional fluctuation of uj ---> nej ∈ (0, 1)
    data.se_j =0 # Individual sensitivity uj 


def prepare_criticality(data):
    data.U_per = np.full(data.num_users,0.2)  # Perception range of individual uj in km
    data.nej = np.random.uniform(0,1,data.num_users) # Emotional fluctuation of uj ---> nej ∈ (0, 1)
    data.se_j = np.random.uniform(0.05,0.1,data.num_users) # Individual sensitivity uj 

    data.D = (data.node_coords[0,0], data.node_coords[0,1])
    data.live_positions = rvoAlgorithm(data)
    du_dt_temp = du_dt_function(data, 0, data.live_positions)
    data.requests_index, data.CR_matrix = criticality(data, 0, du_dt_temp)
    print("-------Criticality---------",data.CR_matrix)

def prepare_live_position(data):
    data.live_positions_requests =[]
    for r in range(data.requests_received):
        for u in range(data.num_users):
            if data.req_by_user[u][r]==1:
                data.live_positions_requests.append(data.live_positions[0][u])

def prepare_coverage_live(data):
    for i in range(len(data.nodes)):		       
        node_latitude = data.node_coords[i,0]		            
        node_longitude = data.node_coords[i,1]		           
        temp = []		      
        for r in range(data.requests_received):		          
            for u in range(data.num_users):		           
                if data.req_by_user[u][r]==1:		                  
                    request_latitude = data.live_positions_requests[r][0]		                     
                    request_longitude = data.live_positions_requests[r][1]		                       
                    dist_geo = haversine(node_longitude, node_latitude, request_longitude, request_latitude)		                        
                    if dist_geo <= data.radius[0]:		                      
                        temp.append(1)		                          
                    else:		                       
                        temp.append(0)		                           
                            
        data.req_node_coverage.append(temp)
        print("-----------position------------------", data.req_node_coverage)