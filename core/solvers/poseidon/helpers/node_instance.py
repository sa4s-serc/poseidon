class node_instance:
    NUM_FEATURES = 2 # number of features in the observation space [available_cores, available_cpu_memory, available_gpu_memory]
    def __init__(self,id ,node_name, node_memory, node_cores, gpu_node_memory, running_cost):
        self.id = id
        self.name = node_name
        self.cpu_memory = node_memory
        self.gpu_memory = gpu_node_memory
        self.cores = node_cores
        self.running_cost = running_cost
        # self.function_instances = [] # dont maintani these as already maintained in the function_instance
        # self.prev_function_instances = [] # single source of truth
        self.available_cores = node_cores
        self.available_cpu_memory = node_memory
        self.available_gpu_memory = gpu_node_memory
    
    def __str__(self):
        return f"{self.name} Available Cores: {self.available_cores}, Available CPU Memory: {self.available_cpu_memory}, Available GPU Memory: {self.available_gpu_memory}"
    
    def __repr__(self):
        return f"node_instance({self.name}, {self.cpu_memory}, {self.cores}, {self.gpu_memory}, {self.running_cost})"
    
    def place_function(self, function_instance):
        # check if the function can be placed on the node
        if self.available_cores < 1 or self.available_cpu_memory < function_instance.cpu_memory or self.available_gpu_memory < function_instance.gpu_memory:
            return False
                
        self.available_cores -= 1 #NOTE: assuming that each function instance requires 1 core
        self.available_cpu_memory -= function_instance.cpu_memory
        self.available_gpu_memory -= function_instance.gpu_memory
        # self.function_instances.append(function_instance)
        # self.prev_function_instances.append(function_instance)
        
        return True
    
    
    
    
