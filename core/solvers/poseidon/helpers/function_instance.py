
class function_instance:
    # number of features in the observation space [cpu_memory, gpu_memory, max_delay, response_time, phi, workload]
    NUM_FEATURES = 3

    def __init__(self,id, function_name, function_memory, gpu_function_memory, response_time, current_placement=[]):
        self.id = id
        self.name = function_name
        self.cpu_memory = function_memory
        self.gpu_memory = gpu_function_memory
        self.response_time = response_time
        self.current_placement = current_placement
        self.prev_placement = current_placement
        self.phi = 0  # TODO: what is this?
        # TODO: what is this? (amount of work done by the function) (number of instructions executed by the function)
        self.workload = 0

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"function_instance({self.name}, {self.cpu_memory}, {self.gpu_memory}, {self.max_delay}, {self.response_time}, {self.current_placement})"

    def __lt__(self, other):
        return self.score() < other.score()

    def score(self):
        # TODO: add an appropriate comparison function
        return 0

    def calculate_disruptions(self):
        creations = 0
        deletions = 0
        migrations = 0

        for node_id in self.current_placement:
            if node_id not in self.prev_placement:
                creations += 1

        for node_id in self.prev_placement:
            if node_id not in self.current_placement:
                deletions += 1

        # migration is min of creations and deletions
        migration = deletions if 0 < deletions < creations else creations

        return creations, deletions, migration

    
