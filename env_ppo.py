"""
This module sets up the community layer environment of Neptune
with the help of OpenAI Gymnasium. It defines the metrics, nodes, functions and the environment itself.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import copy

def scale(x, minv, maxv):
    
    """
    Utility function to scale the reward between -1 and 1
    """

    if x > maxv:
        return 1
    if x < minv:
        return -1

    if minv == np.inf or maxv == -np.inf:
        return 0
    if minv == maxv:
        return 0
    return 2 * ((x - minv) / (maxv - minv)) - 1


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class Metrics:
    """
    Monitors the metrics of the system.
    We monitor the following details:
        - network_delays: The network delays between the nodes
        - functions: The functions to be placed in the community layer
        - nodes: The nodes in the community layer
        - old_placements: The old placements of the functions
        - new_placements: The new placements of the functions
        - delay_history: The history of the delays in the system
        - disruption_history: The history of the disruptions in the system
        - cost_history: The history of the costs in the system
    """

    def __init__(self, nodes, functions, network_delays) -> None:
        
        """
        Arguments:
            nodes {list[Node]} -- A list of nodes in the community layer.
            functions {list[Function]} -- A list of functions to be placed in the community layer.
            network_delays {dict} -- A dictionary containing the network delays between the nodes.
        """
        
        self.network_delays = network_delays
        self.functions = functions
        self.nodes = nodes
        self.old_placements = {i: None for i in range(len(self.nodes))}
        self.new_placements = {i: None for i in range(len(self.nodes))}
        self.delay_history = []
        self.disruption_history = []
        self.cost_history = []

    def get_total_delay(self, func):
        
        """
        Calculate the total delay for a function instance in the system used for reward calculation.
        """
        
        total_delay = 0

        for nodes in self.nodes:
            min_delay = np.inf
            for node in func.current_node:
                min_delay = min(
                    min_delay, self.network_delays[node][nodes.node_id])
            if min_delay != np.inf:
                total_delay += min_delay

        self.delay_history.append(total_delay)
        return total_delay

    def get_total_cost(self,func) -> float:
        
        """
        Calculate the total cost for a function instance in the system used for reward calculation.
        """
        
        total_cost = 0
        for node in func.current_node:
            total_cost += self.nodes[node].runcost
        return total_cost

    def modify_function(self, func):
        self.functions = func.copy()

    def modify_nodes(self, nodes):
        self.nodes = nodes.copy()

    def update_placements(self) -> float:
        """
        Monitors deployed functions and returns reward for disruption.
        We calculate the reward based on the deletions, creations and migrations of the functions.
        """
        
        ## Calculate the deletions, creations and migrations of the functions
        deletions_per_function = {
            func.function_id: 0 for func in self.functions}
        creations_per_function = {
            func.function_id: 0 for func in self.functions}
        placement = self.get_placements()

        ## Comparing the old and new placements to calculate the deletions, creations and migrations.Old to new -> deletions
        for node in list(self.old_placements.keys()):
            if self.old_placements[node]:
                for fun in self.old_placements[node]:
                    if placement[node]:
                        if fun not in placement[node]:
                            deletions_per_function[fun] = deletions_per_function.get(
                                fun, 0) + 1
                    else:
                        deletions_per_function[fun] = deletions_per_function.get(
                            fun, 0) + 1
        
        ## Comparing the old and new placements to calculate the deletions, creations and migrations.New to old -> creations
        for node in list(placement.keys()):
            if placement[node]:
                for fun in placement[node]:
                    if self.old_placements[node]:
                        if fun not in self.old_placements[node]:
                            creations_per_function[fun] = creations_per_function.get(
                                fun, 0) + 1
                    else:
                        creations_per_function[fun] = creations_per_function.get(
                            fun, 0) + 1
        
        ## Calculate the migrations per function
        migration_per_function = {
            func.function_id: 0 for func in self.functions}
        for i in list(migration_per_function.keys()):
            migration_per_function[i] = min(
                deletions_per_function[i], creations_per_function[i]
            ) if deletions_per_function[i] > 0 and creations_per_function[i] > 0 else max(deletions_per_function[i], creations_per_function[i])

        ## Calculate the disruptive reward based on the deletions, creations and migrations
        reward_disruptive = 0
        for i in list(migration_per_function.keys()):
            reward_disruptive += float(
                migration_per_function[i]
                + float(1 / (deletions_per_function[i] + 2))
                - float(1 / (creations_per_function[i] + 2))
            )


        return reward_disruptive

    def get_placements(self):

        """
        Get the current placements of the functions in the nodes.
        """
        
        placements = {i: None for i in range(len(self.nodes))}
        for func in self.functions:
            for test_node in func.current_node:
                if placements[test_node]:
                    placements[test_node].append(func.function_id)
                else:
                    placements[test_node] = [func.function_id]

        return placements

    def plot_metrics(self, window_size : int) -> None:
        
        """
        Generate plots for the metrics of the system
        """
        fig, axs = plt.subplots(3, 1, figsize=(10, 8))

        axs[0].plot(self.delay_history, label='Total Delay')
        axs[0].set_title('Total Delay Over Time')
        axs[0].set_xlabel('Time Steps')
        axs[0].set_ylabel('Total Delay')
        axs[0].legend()

        axs[1].plot(self.disruption_history,
                    label='Disruption Reward', color='orange')
        axs[1].set_title('Disruption Reward Over Time')
        axs[1].set_xlabel('Time Steps')
        axs[1].set_ylabel('Disruption Reward')
        axs[1].legend()

        axs[2].plot(self.cost_history, label='Total Cost', color='green')
        axs[2].set_title('Total Cost Over Time')
        axs[2].set_xlabel('Time Steps')
        axs[2].set_ylabel('Total Cost')
        axs[2].legend()

        plt.tight_layout()
        plt.savefig("metrics.png")
        plt.show()

        plt.close()

        # plot smoothed average quantities
        smooth_delay = np.convolve(self.delay_history, np.ones(
            window_size), 'valid') / window_size
        smooth_disruption = np.convolve(
            self.disruption_history, np.ones(window_size), 'valid') / window_size
        smooth_cost = np.convolve(self.cost_history, np.ones(
            window_size), 'valid') / window_size

        fig, axs = plt.subplots(3, 1, figsize=(10, 8))

        axs[0].plot(smooth_delay, label='Total Delay')
        axs[0].set_title('Smoothed Total Delay Over Time')
        axs[0].set_xlabel('Time Steps')
        axs[0].set_ylabel('Total Delay')
        axs[0].legend()

        axs[1].plot(smooth_disruption,
                    label='Disruption Reward', color='orange')
        axs[1].set_title('Smoothed Disruption Reward Over Time')
        axs[1].set_xlabel('Time Steps')
        axs[1].set_ylabel('Disruption Reward')
        axs[1].legend()

        axs[2].plot(smooth_cost, label='Total Cost', color='green')
        axs[2].set_title('Smoothed Total Cost Over Time')
        axs[2].set_xlabel('Time Steps')
        axs[2].set_ylabel('Total Cost')
        axs[2].legend()

        plt.tight_layout()
        plt.savefig("metrics_smoothed.png")
        plt.show()

        plt.close()


class Node:
    """
    Denotes a single Node in the community layer
    Arguments:
        static_data {dict} -- A dictionary containing the static data of the node.
            - node_id {int} -- The unique identifier of the node.
            - total_cpu_cores {int} -- The total number of CPU cores available in the node.
            - total_cpu_memory {int} -- The total amount of CPU memory available in the node.
            - total_gpu_cores {int} -- The total number of GPU cores available in the node.
            - total_gpu_memory {int} -- The total amount of GPU memory available in the node.
            - processing_power {int} -- The processing power of the node.Used for simulating node layer if required.
            - runcost {int} -- The cost of running a function on the node.We use sigmoid to scale the cost to the same scale for all nodes.
    
    """
    
    

    def __init__(self, static_data: dict) -> None:
        self.node_id = static_data["node_id"]
        self.total_cpu_cores = static_data["total_cpu_cores"]
        self.total_cpu_memory = static_data["total_cpu_memory"]
        self.total_gpu_cores = static_data["total_gpu_cores"]
        self.total_gpu_memory = static_data["total_gpu_memory"]
        self.processing_power = static_data["processing_power"]
        self.available_cpu_cores = self.total_cpu_cores
        self.available_cpu_memory = self.total_cpu_memory
        self.available_gpu_cores = self.total_gpu_cores
        self.available_gpu_memory = self.total_gpu_memory
        self.runcost = sigmoid(self.available_cpu_cores + self.available_cpu_memory +
                               self.available_gpu_memory + self.total_gpu_cores)

    def update_node(
        self, cpu_memory: int, gpu_memory: int, cpu_cores: int, gpu_cores: int
    ) -> None:
        
        """
        Update the available resources of the node after a function is placed on it.
        """

        self.available_cpu_cores -= cpu_cores
        self.available_cpu_memory -= cpu_memory
        self.available_gpu_cores -= gpu_cores
        self.available_gpu_memory -= gpu_memory

    def __str__(self):
        return f"Node {self.node_id} has {self.total_cpu_cores} CPU cores, {self.total_cpu_memory} CPU memory, {self.total_gpu_cores} GPU cores, {self.total_gpu_memory} GPU memory"


class Function:
    """
    Denotes a single function to be placed in the community layer
    """

    def __init__(self, static_data: dict) -> None:
        
        """
        Arguments:
            static_data {dict} -- A dictionary containing the static data of the function.
                - function_id {int} -- The unique identifier of the function.
                - required_cpu_memory {int} -- The amount of CPU memory required by the function.
                - required_gpu_memory {int} -- The amount of GPU memory required by the function.
                - origin_node {int} -- The node where the function request originates.
                - total_instruction {int} -- The total number of instructions in the function.Again used for node level simulation if required.
                - response_time {int} -- The response time of the function. Node layer detail.
                - phi {int} -- The phi value of the function. Used for simulating the node layer if required.
        """
        
        self.function_id = static_data["function_id"]
        self.required_cpu_memory = static_data["required_cpu_memory"]
        self.required_gpu_memory = static_data["required_gpu_memory"]
        self.origin_node = static_data["origin_node"]
        self.current_node = [static_data["origin_node"]]
        self.total_instruction = static_data["total_instruction"]
        self.response_time = static_data["response_time"]
        self.phi = static_data["phi"]

    def __lt__(self, other):
        """
        Function comparator. Used for sorting functions based on their resource requirements.
        """
        return (self.total_instruction + self.required_cpu_memory + self.required_gpu_memory) < (other.total_instruction + other.required_cpu_memory + other.required_gpu_memory)


class Community(gym.Env):
    """
    Custom OpenAI Gym Environment for the Community Layer. This defines the action space, observation space , reward , step to simulate episodes.
    We pass the following details to initialise an environment
        - func {list[Function]} -- A list of functions to be placed in the community layer.
        - nodes {list[Node]} -- A list of nodes in the community layer.
        - metric {Metrics} -- A metric object to monitor the system.This is used to calculate the reward,delays,costs and disruptions.
    """
    def __init__(self, func: list[Function], nodes: list[Node], metric: Metrics) -> None:
        """
        Describe the action and state space , initialise the environment.
        
        """

        self.func = func
        self.nodes = nodes
        self.metric = metric
        self.max_delay = 10
        self.all_functions = copy.deepcopy(func)
        self.curr_index = 0
        
        ## Set the min and max values for the metrics and rewards. Stores history of the metrics and rewards to plot them later.
        
        self.compatible_min = np.inf
        self.compatible_max = -np.inf
        self.disruptive_min = np.inf
        self.disruptive_max = -np.inf
        self.cost_min = np.inf
        self.cost_max = -np.inf
        self.delay_min = np.inf
        self.delay_max = -np.inf
        self.delay_history = []
        self.cost_history = []
        self.disruptions_history = []
        self.reward_max = -np.inf
        self.reward_history = []

        ## Set the hyperparameters for the reward calculation
        self.alpha = 0.64 ## Cost-Network Delay tradeoff
        self.epsilon = 0.05 ## Disruption-Network Delay(and Cost) tradeoff

        self.observation_space = spaces.Dict(
            {
                
                "available_resources": spaces.Box(
                    low=0, high=np.inf, shape=(len(nodes), 4
                                               )),
                "function_parameters": spaces.Box(
                    low=0, high=np.inf, shape=(1, 10)),
                "boolean_placements": spaces.MultiBinary((1, 20)),
            }
        )

        # Action Space - choose one out of 20 nodes for each function.
        self.action_space = spaces.MultiBinary(len(nodes))

    def reset(self, seed=None, options=None):
        
        """
        Used to reset the environment to the initial state after an episode is completed.
        We reset the function queue 
        """
        
        super().reset(seed=seed)
        self.func = sorted(self.all_functions)
        self.metric.functions = self.func.copy()

        for func in self.func:
            func.current_node.clear()
            func.current_node.append(func.origin_node)
        # clear available resources
        for node in self.nodes:
            node.available_cpu_cores = node.total_cpu_cores
            node.available_cpu_memory = node.total_cpu_memory
            node.available_gpu_cores = node.total_gpu_cores
            node.available_gpu_memory = node.total_gpu_memory

        initial_observation = self.__get_obs()

        info = {}
        return initial_observation, info

    def get_reward(self):
        """
        Calculate the reward based on the current state of the system
        We calculate the reward based on the network delay, cost and disruptions.
            - Disruptive Reward: Reward for disruptions in the system
            - Delay Reward: Reward for the network delay
            - Cost Reward: Reward for the cost of the system
        """
        
        ## Calculate the disruptive reward
        
        total_disruptions = self.metric.update_placements()
        self.disruptive_min = min(
            self.disruptive_min, total_disruptions)
        self.disruptive_max = max(
            self.disruptive_max, total_disruptions)
        disruptive_reward = -scale(
            total_disruptions, self.disruptive_min, self.disruptive_max)
        
        ## Calculate the delay reward
        
        delay = self.metric.get_total_delay(self.func[0])
        self.delay_min = min(self.delay_min, delay)
        self.delay_max = max(self.delay_max, delay)
        delay_reward = -scale(delay, self.delay_min, self.delay_max)

        ## Calculate the cost reward
    
        cost = self.metric.get_total_cost(self.func[0])
        self.cost_min = min(self.cost_min, cost)
        self.cost_max = max(self.cost_max, cost)
        cost_reward = -scale(cost, self.cost_min, self.cost_max)

        ## Calculate the delay-cost reward
        delay_cost_reward = self.alpha*delay_reward + (1 - self.alpha)*cost_reward
        
        ## Calculate the total reward based on the tradeoff between delay, cost and disruptions
        total_reward = self.epsilon * disruptive_reward + \
            (1 - self.epsilon) * delay_cost_reward
        
        return (total_reward)

    def __get_obs(self):
        """
        Get the current observation of the system. Used to calculate the reward and the next state.
        We return 
            - available_resources: The available resources in the nodes
            - function_parameters: The parameters of the function to be placed
            - boolean_placements: The boolean placements of the function in the nodes
        """
        if (len(self.func) == 0):
            
            ### If there are no functions to place, return a zero observation
            return {
                "available_resources": np.zeros([len(self.nodes), 4]),
                "function_parameters": np.zeros([1, 10]),
                "boolean_placements": np.zeros([1, 20]),
            }
        else:

            current_function = self.func[0]
            remaining_functions = []
            avg_cpu = 0
            avg_gpu = 0
            var_cpu = 0
            var_gpu = 0
            if len(self.func) > 1:
                
                ## Calculate the average and variance of the remaining functions
                remaining_functions = self.func[1:]
                remaining_functions = np.array(remaining_functions)
                if remaining_functions.size > 0:
                    avg_cpu = np.mean(
                        [func.required_cpu_memory for func in remaining_functions])
                    avg_gpu = np.mean(
                        [func.required_gpu_memory for func in remaining_functions])
                    var_cpu = np.var(
                        [func.required_cpu_memory for func in remaining_functions])
                    var_gpu = np.var(
                        [func.required_gpu_memory for func in remaining_functions])

            observation = {
                "available_resources": np.array(
                    [
                        [
                            node.available_cpu_cores,
                            node.available_cpu_memory,
                            node.available_gpu_cores,
                            node.available_gpu_memory,
                        ]
                        for node in self.nodes
                    ]
                ),
                "function_parameters": np.array(
                    [[
                        current_function.required_cpu_memory,
                        current_function.required_gpu_memory,
                        current_function.origin_node,
                        current_function.total_instruction,
                        current_function.response_time,
                        current_function.phi,
                        avg_cpu,
                        avg_gpu,
                        var_cpu,
                        var_gpu,
                    ]]
                ),
                "boolean_placements": np.array(
                    [[
                        1 if node in current_function.current_node else 0 for node in self.nodes
                    ]]
                ),
            }

            return observation

    def step(self, action : list[int]) -> tuple:
        
        """
        Take a step in the environment based on the action taken.This calculates reward.
        """
        
        reward = 0
        done = False
        info = {}

        ## Get the nodes where the function is placed
        chosen_nodes = []
        for i in range(len(action)):
            if action[i] == 1:
                chosen_nodes.append(i)

        ## If no nodes are chosen, return a negative reward
        if len(chosen_nodes) == 0:
            reward = -1

        ## If the function requires more resources than available, return a negative reward
        elif any(
            [
                self.func[0].required_cpu_memory > self.nodes[node].available_cpu_memory
                and self.func[0].required_gpu_memory > self.nodes[node].available_gpu_memory
                for node in chosen_nodes
            ]
        ):
            reward = -1

        ## If the function is placed in a node, update the resources and calculate the reward
        else:
            for node in chosen_nodes:
                self.nodes[node].update_node(
                    self.func[self.curr_index].required_cpu_memory,
                    self.func[self.curr_index].required_gpu_memory,
                    0,
                    0,
                )

            self.func[0].current_node.clear()
            self.func[0].current_node = chosen_nodes

            reward = self.get_reward()

        observation = self.__get_obs()
        
        ## Remove the function from the queue
        try:
            print(
                f"Function {self.func[0].function_id} placed at {self.func[0].current_node}")
            self.func.pop(0)
        except:
            print("No more functions to place")
            pass

        done = len(self.func) == 0
        
        self.reward_history.append(reward)

        ## Update the metrics and calculate the reward for the episode.
        if done:
            self.delay_history.append(
                sum(self.metric.get_total_delay(f) for f in self.all_functions))
            self.cost_history.append(
                sum(self.metric.get_total_cost(f) for f in self.all_functions)
            )
            
            if reward>self.reward_max:
                self.reward_max = reward
                self.metric.old_placements = self.metric.get_placements()
            
            self.disruptions_history.append(
                self.metric.update_placements()
            )

        return observation, reward, done, False, info

    def add_function(self, param : Function) -> None:
        """
        Add a function to the community layer in dynamic mode
        """
        
        self.all_functions.append(param)
        self.metric.modify_function(self.all_functions)

    def remove_function(self, func_id : int) -> None:

        """
        Remove a function from the community layer in dynamic mode
        """

        for func in self.all_functions:
            if func.function_id == func_id:
                self.all_functions.remove(func)
                self.metric.modify_function(self.all_functions)
                return
        raise ValueError(f"Function {func_id} not in Community")

    def add_node(self, param : Node) -> None:
        self.nodes.append(param)
        self.metric.modify_nodes(self.nodes)

    def remove_node(self, node_id):
        for node in self.nodes:
            if node.node_id == node_id:
                self.nodes.remove(node)
                self.metric.modify_nodes(self.nodes)
                return
        raise ValueError(f"Node {node_id} not in Community")

    def plot_metrics(self, window_size):
        self.metric.plot_metrics(window_size)
