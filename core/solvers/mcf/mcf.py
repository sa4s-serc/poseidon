from ..vsvbp import VSVBP
from .utils import prepare_order_requests
from ..criticality import CriticalityHeuristic
from ..criticality.utils import *
from ..workload import WorkloadGenerator
import time
class MCF(CriticalityHeuristic):
    
    def __init__(self, danger_radius_km=0.5, **kwargs):
        super().__init__(danger_radius_km, **kwargs)
        super().init_vars()
        self.sample_workloads = []
        self.inference_time = []
        self.cost_history = []
        self.copy_data = None
        self.workload_generator = WorkloadGenerator(
            dir_path="Cabspotting", functions=np.array([0.1 for _ in range(10)]), nodes=np.array([[0.25, 0.25], [0.25, 0.75], [0.75, 0.25],[0.5,0.5],[0.75,0.75]]))

    
    def prepare_data(self, data):
        VSVBP.prepare_data(self, data)
        prepare_order_requests(data)

    def get_delay(self):
        delay = 0
        # delay_step2 = 0
        for f in range(len(self.data.functions)):
            for i in range(len(self.data.nodes)):
                for j in range(len(self.data.nodes)):
                    delay += self.x_jr[i, f, j] * self.data.node_delay_matrix[i, j] * self.data.workload_matrix[f, i]
                    # delay_step2 += self.step2_x[i, f, j] * self.data.node_delay_matrix[i, j] * self.data.workload_matrix[f, i]
        return delay
    
    def get_cost(self):
        cost  = 0
        for i in range(len(self.data.functions)):
            for j in range(len(self.data.nodes)):
                if self.c_fj[i,j] == 1:
                    cost += self.data.node_costs[j]
                
        return cost
    
    def solve(self):

        for _ in range(151):
            
            if len(self.sample_workloads) == 0:
                self.data.workload_matrix = self.workload_generator.reset_timestamp()
            else:
                self.data.workload_matrix = self.workload_generator.get_workload()
            self.prepare_data(self.data)
            time_start = time.time()
            criticality_heuristic(self.data, self.log, self.S_active, self.y_j, self.c_fj, self.x_jr)   
            self.inference_time.append(time.time() - time_start)
            self.delay_history.append(self.get_delay()[0])
            self.delay_history_step2.append(self.get_delay()[1])
            self.cost_history.append(self.get_cost())
            
    def results(self):
        return [],self.delay_history,self.inference_time,self.sample_workloads,self.cost_history