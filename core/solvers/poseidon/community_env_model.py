import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from .helpers.node_instance import node_instance as node
from .helpers.function_instance import function_instance as function
from .helpers.metrics_monitor import metrics_monitor as metrics
from .helpers.helper import linear_minmax_scale
from ortools.linear_solver import pywraplp
import copy


class community_env_model(gym.Env):
    def __init__(self, alpha=0.94, tradeoff_epsilon=[1, 0], max_nodes=5, metric_tracking=True, function_instances=[], node_instances=[], inter_node_delays=[], data=[]):
        super(community_env_model, self).__init__()
        self.max_nodes = max_nodes
        self.tradeoff_epsilon_matrix = np.array(tradeoff_epsilon)
        self.alpha = alpha
        self.total_delay = 0
        self.metric_tracking = metric_tracking
        if self.metric_tracking:
            self.metrics = metrics()
        self.cost = 0
        self.delay = 0
        self.cum_reward = 0

        self.function_instances = list(function_instances)
        # NOTE: this is a queue of function instances that need to be placed. We keep modifying this and use the other one to track all functions.
        self.function_queue = list(copy.deepcopy(function_instances))
        self.node_instances = node_instances
        self.inter_node_delays = inter_node_delays
        self.delay_bounds = [0,sum([sum(src_node_delays) for src_node_delays in inter_node_delays])
                             ]
        self.cost_bounds = [0, sum([node.running_cost for node in node_instances])
                            ]
        self.disruption_bounds = [0, 0]
        self.data = data
        self.solver = pywraplp.Solver.CreateSolver('SCIP')
        self.x = {}
        self.init_x()
        self.objective = self.solver.Objective()
        self.available_node_cores = copy.deepcopy(self.data.node_cores_matrix)

        self.observation_space = spaces.Dict({
            "inter_node_delays": spaces.Box(
                low=0, high=np.inf, shape=(max_nodes, max_nodes)
            ),
            "available_resources": spaces.Box(
                low=0, high=np.inf, shape=(max_nodes, node.NUM_FEATURES
                                           )),
            "function_parameters": spaces.Box(
                low=0, high=np.inf, shape=(1, function.NUM_FEATURES)),
            "function_workload": spaces.Box(
                low=0, high=np.inf, shape=(1, max_nodes)
            ),
            "total_delay": spaces.Box(
                low=0, high=np.inf, shape=(1,)
            )
        })

        self.action_space = spaces.MultiBinary(max_nodes)
        
    def init_x(self):
        for f in range(len(self.data.functions)):
            for i in range(len(self.data.nodes)):
                for j in range(len(self.data.nodes)):
                    self.x[i, f, j] = self.solver.NumVar(0,1, f"x[{i}][{f}][{j}]")

    def add_function_instance(self, function_instance):
        self.function_instances[function_instance.name] = function_instance

        self.cost_bounds[1] += function_instance.running_cost * \
            len(self.node_instances)
        self.disruption_bounds[1] += len(self.node_instances) + \
            (1 / (len(self.node_instances) + 2))

    def populate_node_instances(self, node_instances, inter_node_delays):
        self.node_instances = node_instances
        self.inter_node_delays = inter_node_delays

        self.delay_bounds = [0, max(src_node_delays.values()
                                    for src_node_delays in inter_node_delays.values())]

    def __get_observation(self):

        available_resources = np.zeros((self.max_nodes, node.NUM_FEATURES))
        function_parameters = np.zeros((1, function.NUM_FEATURES))
        function_workload = np.zeros((1, self.max_nodes))
        inter_node_delay_matrix = self.inter_node_delays
        for node_instance in self.node_instances:
            available_resources[node_instance.id] = [
                node_instance.available_cores, node_instance.available_cpu_memory]
        total_delay = np.array([self.total_delay])
        avg_cpu_memory, avg_gpu_memory, var_cpu_memory, var_gpu_memory = 0, 0, 0, 0

        if len(self.function_queue) > 0:
            curr_function = self.function_queue[0]
            remaining_function_queue = np.array(self.function_queue[1:])
            function_workload = self.data.workload_matrix[self.function_queue[0].id]
            print(f"Function Workload: {function_workload.shape}")
            function_workload = np.reshape(function_workload,(1, self.max_nodes))   

            if len(remaining_function_queue) > 0:
                avg_cpu_memory = np.mean(
                    [function_instance.cpu_memory for function_instance in remaining_function_queue])
                var_cpu_memory = np.var(
                    [function_instance.cpu_memory for function_instance in remaining_function_queue])

            function_parameters = [curr_function.cpu_memory, avg_cpu_memory, var_cpu_memory]
            function_parameters = np.array(function_parameters)
            function_parameters = np.reshape(
                function_parameters, (1, function.NUM_FEATURES))
            

        observation = {
            "inter_node_delays": inter_node_delay_matrix,
            "available_resources": available_resources,
            "function_parameters": function_parameters,
            "function_workload": function_workload,
            "total_delay": total_delay,
        }

        return observation

    def reset(self, seed=None):
        super(community_env_model, self).reset(seed=seed)

        self.function_queue = copy.deepcopy(self.function_instances)
        # sort functions based on total workload
        self.function_queue = sorted(self.function_queue, key=lambda x: sum(self.data.workload_matrix[x.id]))
        self.function_queue = sorted(self.function_queue)
        self.available_node_cores = copy.deepcopy(self.data.node_cores_matrix)
        for function_instance in self.function_instances:
            function_instance.prev_placement = function_instance.current_placement
            function_instance.current_placement = []

        for node_instance in self.node_instances:
            node_instance.available_cores = node_instance.cores
            node_instance.available_cpu_memory = node_instance.cpu_memory
            node_instance.available_gpu_memory = node_instance.gpu_memory

        self.solver.Clear()
        self.x.clear()
        self.objective.Clear()
        self.delay = 0    
        
        ##initialize x
        self.init_x()
        
        
        return self.__get_observation(), {}

    def get_scaled_reward(self, chosen_nodes):
        delay_penalty = 0
        cost_penalty = 0
        disruption_penalty = 0
        current_function = self.function_queue[0]

        # calculate the delay penalty
        delay_penalty += self.calculate_total_delay_per_function_instance(
            current_function)

        print(f"Delay: {delay_penalty}, Delay min: {self.delay_bounds[0]}, Delay max: {self.delay_bounds[1]}")
        self.metrics.total_delay_history.append(delay_penalty)
        # calculate the cost penalty
        cost_penalty += self.calculate_cost(current_function)

        print(
            f"Cost: {cost_penalty}, Cost min: {self.cost_bounds[0]}, Cost max: {self.cost_bounds[1]}")

        self.metrics.total_cost_history.append(cost_penalty)

        # calculate the disruption penalty
        creations, deletions, migrations = current_function.calculate_disruptions()
        disruption_penalty += migrations + \
            (1 / (deletions + 2)) - (1 / (creations + 2)
                                     )  # TODO: check this term looks sus

        self.metrics.total_disruption_history.append(disruption_penalty)
        self.disruption_bounds[0] = min(
            self.disruption_bounds[0], disruption_penalty)
        self.disruption_bounds[1] = max(
            self.disruption_bounds[1], disruption_penalty)

        delay_penalty = linear_minmax_scale(
            delay_penalty, self.delay_bounds[0], self.delay_bounds[1])
        cost_penalty = linear_minmax_scale(
            cost_penalty, self.cost_bounds[0], self.cost_bounds[1])
        disruption_penalty = linear_minmax_scale(
            disruption_penalty, self.disruption_bounds[0], self.disruption_bounds[1])

        print(
            f"Delay Penalty: {delay_penalty}, Cost Penalty: {cost_penalty}, Disruption Penalty: {disruption_penalty}")

        delay_cost_tradeoff = float((1-self.alpha) * \
            delay_penalty + (self.alpha) * cost_penalty)

        reward = -np.dot(self.tradeoff_epsilon_matrix,
                         [delay_cost_tradeoff, disruption_penalty])

        return reward

    def calculate_total_delay_per_function_instance(self, function_instance):
        # NOTE: assumes currently one request per for each node and uses minimum delay as heuristics
        # TODO: update this to use actual requests workload and maybe include actual routing policy
        total_delay = 0

        for src_node_id in range(len(self.data.nodes)):
            for dest_node_id in range(len(self.data.nodes)):
                total_delay += self.data.node_delay_matrix[src_node_id, dest_node_id] * self.data.workload_matrix[function_instance.id, src_node_id]*self.x[src_node_id,function_instance.id,dest_node_id].solution_value()
        if total_delay>self.delay_bounds[1]:
            self.delay_bounds[1] = total_delay
        self.delay+=total_delay
        return total_delay
    
    
    def calculate_cost(self, function_instance):
        cost = sum(
            self.node_instances[node_id].running_cost for node_id in function_instance.current_placement)
        
        self.cost_bounds[1] = max(self.cost_bounds[1], cost)
        self.cost+=cost
        return cost

    def init_constraints(self,chosen_nodes):
        
        
        ##all routes of a function instance at each node should sum to 1 and only on valid nodes should be non zero
        f = self.function_queue[0].id
        for i in range(len(self.data.nodes)):
            ##sum 1 and only those non zero which are in chosen nodes
            self.solver.Add(
                self.solver.Sum([
                    self.x[i, f, j] for j in chosen_nodes
                ]) == 1 )
            ##others should be 0
            for j in range(len(self.data.nodes)):
                if j not in chosen_nodes:
                    self.solver.Add((self.x[i,f,j] == 0) ) 
            
        for j in range(len(self.data.nodes)):
            self.solver.Add(
                self.solver.Sum([
                    self.x[i, f, j] * self.data.workload_matrix[f, i] * self.data.core_per_req_matrix[f, j]
                    for i in range(len(self.data.nodes))
                ]) <= self.available_node_cores[j]
            )
            
        

            
    def init_objective(self):
        
        ##minimize node delay. Iterate over all i,j and then sum over all f for the term workload*routing*delay.This sum should be minimized
        f = self.function_queue[0].id
        for i in range(len(self.data.nodes)):
            for j in range(len(self.data.nodes)):
                self.objective.SetCoefficient(self.x[i, f, j], float(self.data.workload_matrix[f, i]*self.data.node_delay_matrix[i, j]))
        
        self.objective.SetMinimization()
            

    def step(self, action):

        chosen_nodes = np.where(action == 1)[0]
        
        done = False
        info = {}
        reward = 0
        cost = 0
        self.delay = 0
        self.cost = 0
        

        current_function = self.function_queue[0]

        if len(chosen_nodes) == 0:
            reward = -10
        elif any(self.node_instances[node_id].available_cpu_memory < current_function.cpu_memory and self.node_instances[node_id].available_gpu_memory < current_function.gpu_memory for node_id in chosen_nodes):
            reward = -10
        else:
            
            for node in chosen_nodes:
                if self.node_instances[node].available_cpu_memory < current_function.cpu_memory:
                    print(f"Node {node} does not have enough memory")
                    self.node_instances[node].available_cpu_memory = 0
                    chosen_nodes = np.delete(chosen_nodes, np.where(chosen_nodes == node))
                else:
                    self.node_instances[node].available_cpu_memory -= current_function.cpu_memory
            
            current_function.current_placement.clear()
            current_function.current_placement.extend(chosen_nodes)
            
            print(
            f"Placing function {self.function_queue[0].name} on nodes {chosen_nodes}")
            
            ##get routing policy based on the current placement
            self.init_constraints(chosen_nodes)
            self.init_objective()

            result=self.solver.Solve()
            print("Solved")
            print(f" Result:{result}")
            
            status  = (result != 2)
           
            if status:
                
                for i in range(len(self.data.nodes)):
                    for j in chosen_nodes:
                        print(f"x[{i}][{current_function.id}][{j}]: {self.x[i,current_function.id,j].solution_value()}")
                
                ##update available cores
                for i in range(len(self.data.nodes)):
                    for j in chosen_nodes:
                        self.available_node_cores[j] -= self.data.workload_matrix[current_function.id, i] * self.data.core_per_req_matrix[current_function.id, j]*self.x[i,current_function.id,j].solution_value()
                print(f"Available cores: {self.available_node_cores}")
                reward = self.get_scaled_reward(chosen_nodes)
                print(f"Reward: {reward}")
            else:
                reward = -10
            

        self.function_queue.pop(0)
        observation = self.__get_observation()
        done = len(self.function_queue) == 0

        
        
        self.cum_reward += reward
        
        if done:
            self.metrics.total_reward_history.append(self.cum_reward)
            self.cum_reward = 0
            self.metrics.total_reward_history.append(reward)
            self.metrics.total_cost_history.append(self.cost)
            self.metrics.total_delay_history.append(self.delay)
            
        

        return observation, reward, done, False, info
