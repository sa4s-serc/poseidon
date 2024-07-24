import datetime

import numpy as np
from ortools.linear_solver import pywraplp
from ..solver import Solver
from ortools.sat.python import cp_model
from .utils import *
from ..neptune.utils.output import convert_c_matrix, convert_x_matrix
from ..workload import WorkloadGenerator
import time
class VSVBP(Solver):

    def __init__(self, num_users=8, **kwargs):
        super().__init__(**kwargs)
        self.num_users = num_users
        self.model = cp_model.CpModel()
        self.solver = cp_model.CpSolver()
        self.x, self.c, self.y = {}, {}, {}
        self.first_step = True
        self.delay_history = []
        self.sample_workloads = []
        self.inference_time = []
        self.cost_history = []
        self.copy_data = None
        self.workload_generator = WorkloadGenerator(
            dir_path="Cabspotting", functions=np.array([0.1 for _ in range(10)]), nodes=np.array([[0.25, 0.25], [0.25, 0.75], [0.75, 0.25],[0.5,0.5],[0.75,0.75]]))

        
        
    def get_delay(self):
        delay = 0
        # delay_step2 = 0
        x = output_xjr(self.data, self.solver, self.status, self.x, self.c, self.y)
        x,c = output_x_and_c(self.data, self.solver, self.c, x)
        for f in range(len(self.data.functions)):
            for i in range(len(self.data.nodes)):
                for j in range(len(self.data.nodes)):
                    delay += x[i, f, j] * self.data.node_delay_matrix[i, j] * self.data.workload_matrix[f, i]
                    # delay_step2 += self.step2_x[i, f, j] * self.data.node_delay_matrix[i, j] * self.data.workload_matrix[f, i]
        return delay
    
    def get_cost(self):
        cost  = 0
        x = output_xjr(self.data, self.solver, self.status, self.x, self.c, self.y)
        x,c = output_x_and_c(self.data, self.solver, self.c, x)
        for i in range(len(self.data.functions)):
            for j in range(len(self.data.nodes)):
                if int(self.solver.Value(self.c[i,j])) == 1:
                    cost += self.data.node_costs[j]
                
        return cost

    def load_data(self, data):
        self.prepare_data(data)
        super().load_data(data)
        # if len(self.sample_workloads) == 0:
        #         self.data.workload_matrix = self.workload_generator.reset_timestamp()
        # else:
        #     self.data.workload_matrix = self.workload_generator.get_workload()

    def prepare_data(self, data):
        data.num_users = self.num_users
        data.node_coords = delay_to_geo(data.node_delay_matrix)
        data.user_coords = place_users_close_to_nodes(data.num_users, data.node_coords)
        data.radius = get_radius(data.node_coords)
        prepare_requests(data)
        prepare_req_distribution(data)
        prepare_coverage(data)

    def init_vars(self):
        init_x(self.data, self.model, self.x)
        init_c(self.data, self.model, self.c)
        init_y(self.data, self.model, self.y)

    def init_constraints(self):
        if self.first_step:
            constrain_coverage(self.data, self.model, self.x)
            constrain_proximity(self.data, self.model, self.x)
            constrain_memory(self.data, self.model, self.c, self.y)
            constrain_cpu(self.data, self.model, self.x, self.y)
            constrain_request_handled(self.data, self.model, self.x)
            constrain_c_according_to_x(self.data, self.model, self.c, self.x)    
            constrain_y_according_to_x(self.data, self.model, self.y, self.x)
            constrain_amount_of_instances(self.data, self.model, self.c)
        else:
            add_hints(self.data, self.model, self.solver, self.x)
            constrain_previous_objective(self.data, self.model, self.solver,self.x)
            
    
    def init_objective(self):
        if self.first_step:
            maximize_handled_requests(self.data, self.model, self.x)
        else:
            minimize_utilization(self.data, self.model, self.y)


    def solve(self):
        
        # for _ in range(151):
            
        self.load_data(self.data)
        self.init_vars()
        self.init_constraints()
        self.init_objective()
        time_start = time.time()
        self.solver.Solve(self.model)
        self.status = self.solver.Solve(self.model)
        self.inference_time.append(time.time()-time_start)
        self.delay_history.append(self.get_delay())
        self.cost_history.append(self.get_cost())
        self.sample_workloads.append([self.data.workload_matrix,self.workload_generator.timestamp])
        self.first_step = False
        self.init_objective()
        
            
        print(self.delay_history,self.cost_history,self.inference_time)

    def results(self):
        
        return [],self.delay_history,self.inference_time,self.sample_workloads,self.cost_history
    


