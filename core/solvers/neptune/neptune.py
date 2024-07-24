from ..solver import Solver
from .neptune_step1 import *
from .neptune_step2 import *
from .utils.output import convert_x_matrix, convert_c_matrix
from ..workload import WorkloadGenerator
import copy
import time
class NeptuneBase(Solver):
    def __init__(self, step1=None, step2_delete=None, step2_create=None, **kwargs):
        super().__init__(**kwargs)
        self.step1 = step1
        self.step2_delete = step2_delete
        self.step2_create = step2_create
        
        self.solved = False
        self.delay_history = []
        self.delay_history_step2 = []
        self.sample_workloads = []
        self.inference_time = []
        self.cost_history = []
        self.copy_data = None
        self.workload_generator = WorkloadGenerator(
            dir_path="Cabspotting", functions=np.array([0.1 for _ in range(10)]), nodes=np.array([[0.25, 0.25], [0.25, 0.75], [0.75, 0.25],[0.5,0.5],[0.75,0.75]]))

    def init_vars(self): pass
    def init_constraints(self): pass

    def get_delay(self):
        delay = 0
        delay_step2 = 0
        for f in range(len(self.data.functions)):
            for i in range(len(self.data.nodes)):
                for j in range(len(self.data.nodes)):
                    delay += self.step1_x[i, f, j] * self.data.node_delay_matrix[i, j] * self.data.workload_matrix[f, i]
                    delay_step2 += self.step2_x[i, f, j] * self.data.node_delay_matrix[i, j] * self.data.workload_matrix[f, i]
        return delay,delay_step2
    
    def get_cost(self):
        cost  = 0
        for i in range(len(self.data.functions)):
            for j in range(len(self.data.nodes)):
                if self.step1_c[i,j] == 1:
                    cost += self.data.node_costs[j]
                
        return cost
    
    def solve(self):
        
        for _ in range(1):
            ##reset solver data
            if self.copy_data:  
                self.data = copy.deepcopy(self.copy_data)
            self.step1.solver.Clear()
            self.step1.objective.Clear()
            self.step1.load_data(self.data)
            # if len(self.sample_workloads) == 0:
            #     self.data.workload_matrix = self.workload_generator.reset_timestamp()
            # else:
            #     self.data.workload_matrix = self.workload_generator.get_workload()
            # self.step1.workload_matrix = self.data.workload_matrix
            time_start = time.time()
            self.step1.init_vars()
            self.step1.init_constraints()
            self.sample_workloads.append([self.data.workload_matrix,self.workload_generator.timestamp])
            self.step1.solve()
            self.step1_x, self.step1_c = self.step1.results()
            self.data.max_score = self.step1.score()
            
            self.step2_delete.solver.Clear()
            self.step2_delete.objective.Clear()
            self.step2_delete.load_data(self.data)
            # self.step2_delete.workload_matrix = self.data.workload_matrix
            self.step2_delete.init_vars()
            self.step2_delete.init_constraints()
            self.solved = self.step2_delete_solved = self.step2_delete.solve()
            self.step2_x, self.step2_c = self.step2_delete.results()
            if not self.solved:
                self.step2_create.objective.Clear()
                self.step2_create.solver.Clear()
                self.step2_create.load_data(self.data)
                # self.step2_create.workload_matrix = self.data.workload_matrix
                self.step2_create.init_vars()
                self.step2_create.init_constraints()
                self.solved = self.step2_create.solve()
                self.step2_x, self.step2_c = self.step2_create.results()
        
            self.inference_time.append(time.time() - time_start)
            self.delay_history.append(self.get_delay()[0])
            self.delay_history_step2.append(self.get_delay()[1])
            self.cost_history.append(self.get_cost())

        return self.solved
    
    def results(self): 
        
        return self.delay_history_step2,self.delay_history,self.inference_time,self.sample_workloads,self.cost_history
        
    def score(self):
        return { "step1": self.step1.score(), "step2": self.step2_delete.score() if self.step2_delete_solved else self.step2_create.score() }

class NeptuneMinDelayAndUtilization(NeptuneBase):
    def __init__(self, **kwargs):
        super().__init__(
            NeptuneStep1CPUMinDelayAndUtilization(**kwargs), 
            NeptuneStep2MinDelayAndUtilization(mode="delete", **kwargs),
            NeptuneStep2MinDelayAndUtilization(mode="create", **kwargs),
            **kwargs
            )


class NeptuneMinDelay(NeptuneBase):
    def __init__(self, **kwargs):
        super().__init__(
            NeptuneStep1CPUMinDelay(**kwargs), 
            NeptuneStep2MinDelay(mode="delete", **kwargs),
            NeptuneStep2MinDelay(mode="create", **kwargs),
            **kwargs
            )

class NeptuneMinUtilization(NeptuneBase):
    def __init__(self, **kwargs):
        super().__init__(
            NeptuneStep1CPUMinUtilization(**kwargs), 
            NeptuneStep2MinUtilization(mode="delete", **kwargs),
            NeptuneStep2MinUtilization(mode="create", **kwargs),
            **kwargs
            )
