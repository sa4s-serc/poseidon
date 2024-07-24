from .dataset import CabspottingUserFactory
import numpy as np
class WorkloadGenerator:

    def __init__(self, dir_path, functions, nodes):
        
        self.normalization_constant = 1000000
        self.functions = functions
        self.nodes = nodes
        self.dir_path = dir_path
        self.user_factory = CabspottingUserFactory(
            dataset_dir=self.dir_path, node_coordinates=self.nodes, functions=self.functions)
        self.timestamp = 1

    def get_workload(self):
        self.timestamp += 100
        workload = np.array(self.user_factory.get_user_workload(self.timestamp/self.normalization_constant)).transpose()
        return workload
    
    def reset_timestamp(self):
        workload = np.array(self.user_factory.get_user_workload(1/self.normalization_constant)).transpose()
        self.timestamp = 1
        return workload
    
