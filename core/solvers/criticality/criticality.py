from ..vsvbp import VSVBP
from .utils import *
from ..neptune.utils.output import convert_c_matrix, convert_x_matrix
from ..vsvbp.utils.output import output_x_and_c
import time

class Criticality(VSVBP):
    def __init__(self, danger_radius_km=0.5, **kwargs):
        super().__init__(**kwargs)
        self.danger_radius_km = danger_radius_km
        self.delay_history = []
        self.sample_workloads = []
        self.inference_time = []
        self.cost_history = []

 
    def prepare_data(self, data):
        prepare_aux_vars(data, self.danger_radius_km)
        super().prepare_data(data)
        prepare_criticality(data)
        prepare_live_position(data)        
        prepare_coverage_live(data)


    def init_objective(self):
        if self.first_step:
            maximize_handled_most_critical_requests(self.data, self.model, self.x)
        else:
            super().init_objective()


class CriticalityHeuristic(Criticality):
    def init_vars(self): 
        self.x_jr, self.c_fj, self.y_j, self.S_active = init_all_vars(self.data)

    def init_constraints(self): pass
    def init_objective(self): pass

    def get_delay(self):
        delay = 0
        # delay_step2 = 0
        x,c = output_x_and_c(self.data, None, self.c_fj, self.x_jr)
        # x,c = output_x_and_c(self.data, self.solver, self.c, x)
        for f in range(len(self.data.functions)):
            for i in range(len(self.data.nodes)):
                for j in range(len(self.data.nodes)):
                    delay += x[i, f, j] * self.data.node_delay_matrix[i, j] * self.data.workload_matrix[f, i]
                    # delay_step2 += self.step2_x[i, f, j] * self.data.node_delay_matrix[i, j] * self.data.workload_matrix[f, i]
        return delay
    
    def get_cost(self):
        cost  = 0
        x,c = output_x_and_c(self.data, None, self.c_fj, self.x_jr)
        for i in range(len(self.data.functions)):
            for j in range(len(self.data.nodes)):
                if int(self.solver.Value(self.c[i,j])) == 1:
                    cost += self.data.node_costs[j]
                
        return cost

    def solve(self):
        self.init_vars()
        time_start = time.time()
        criticality_heuristic(self.data, self.log, self.S_active, self.y_j, self.c_fj, self.x_jr)
        self.inference_time.append(time.time() - time_start)
        self.delay_history.append(self.get_delay())
        self.cost_history.append(self.get_cost())
        

    def results(self):
        return [], self.delay_history, self.inference_time, self.cost_history