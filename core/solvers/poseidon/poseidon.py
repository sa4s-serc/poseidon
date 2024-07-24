
from .helpers.function_instance import function_instance as function
from .helpers.node_instance import node_instance as node
from .community_env_model import community_env_model as cem
from ortools.linear_solver import pywraplp
from stable_baselines3 import A2C,PPO
from ..workload import WorkloadGenerator
import numpy as np
import datetime
import time
import matplotlib.pyplot as plt
import pickle

class Poseidon:
    def __init__(self, verbose: bool = True, **kwargs):
        self.verbose = verbose
        self.data = None
        self.env = None
        self.agent = None
        self.solver = None
        self.alpha = 0.6
        self.args = kwargs
        self.max_nodes = 5
        self.workload_generator = WorkloadGenerator(
            dir_path="Cabspotting", functions=np.array([0.25 for _ in range(4)]), nodes=np.array([[0.25, 0.25], [0.25, 0.75], [0.75, 0.25],[0.5,0.5],[0.75,0.75]]))
        self.x = {}
        self.c = {}
        
        self.delay_history_untrained = []
        self.sample_workloads_untrained = []
        self.inference_time_untrained = []
        self.reward_history_untrained = []
        self.cost_history_untrained = []
        
        self.delay_history = []
        self.sample_workloads = []
        self.inference_time = []
        self.reward_history = []
        self.cost_history = []

    def plot_learning(self):
        # plot reward cumulative
        
        #plot rolling averages of delay_history and delay_history_untrained
        
        plt.plot(self.delay_history,label='Trained',color='blue')
        plt.plot(self.delay_history_untrained,label='Untrained',color='red')
        plt.xlabel('Episode')
        plt.ylabel('Delay')
        plt.title('Rolling Average of Delay')
        plt.savefig('learning_curve_delay.png')
        plt.show()
        ##plot rolling averages of reward_history and reward_history_untrained
        reward_history_rolling = np.convolve(self.reward_history, np.ones((10,))/10, mode='valid')
        reward_history_untrained_rolling = np.convolve(self.reward_history_untrained, np.ones((10,))/10, mode='valid')
        plt.plot(reward_history_rolling,label='Trained',color='blue')
        plt.plot(reward_history_untrained_rolling,label='Untrained',color='red')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Rolling Average of Reward')
        plt.savefig('learning_curve_reward.png')
        plt.show()
        
        ##plot rolling averages of cost_history and cost_history_untrained
        cost_history_rolling = np.convolve(self.cost_history, np.ones((10,))/10, mode='valid')
        cost_history_untrained_rolling = np.convolve(self.cost_history_untrained, np.ones((10,))/10, mode='valid')
        plt.plot(cost_history_rolling,label='Trained',color='blue')
        plt.plot(cost_history_untrained_rolling,label='Untrained',color='red')
        plt.xlabel('Episode')
        plt.ylabel('Cost')
        plt.title('Rolling Average of Cost')
        plt.savefig('learning_curve_cost.png')
        plt.show()
        
        
    def init_nodes(self):

        nodes = []
        for i in range(len(self.data.nodes)):
            id = i
            name = self.data.nodes[i]
            cores = self.data.node_cores_matrix[i]
            cpu_memory = self.data.node_memory_matrix[i]
            gpu_memory = 0
            cost = self.data.node_costs[i]
            node_instance = node(id=id, node_name=name, node_memory=cpu_memory,
                                 node_cores=cores, gpu_node_memory=gpu_memory, running_cost=cost)
            nodes.append(node_instance)

        # Add empty nodes to fill up to max_nodes
        for i in range(len(self.data.nodes), self.max_nodes):
            id = i
            name = f"node_{i}"
            cores = 0
            cpu_memory = 0
            gpu_memory = 0
            cost = 20
            node_instance = node(id=i, node_name=name, node_memory=cpu_memory,
                                 node_cores=cores, gpu_node_memory=gpu_memory, running_cost=cost)
            nodes.append(node_instance)

        return np.array(nodes)

    def init_functions(self):

        functions = []
        for i in range(len(self.data.functions)):
            id = i
            name = self.data.functions[i]
            cpu_memory = self.data.function_memory_matrix[i]
            gpu_memory = 0
            response_time = 0
            function_instance = function(id=i, function_name=name, function_memory=cpu_memory,
                                         gpu_function_memory=gpu_memory, response_time=response_time, current_placement=[])
            functions.append(function_instance)

        return np.array(functions)

    def modify_network_delays(self):

        # Modify network delays to include new nodes

        inter_node_delays = self.data.node_delay_matrix

        current_size = inter_node_delays.shape[0]

        padded_node_delays = np.full((self.max_nodes, self.max_nodes), 20)

        if current_size < self.max_nodes:
            padded_node_delays[:current_size,
                               :current_size] = inter_node_delays
            self.data.node_delay_matrix = padded_node_delays

    def init_environment(self):
        print("Initializing Environment...")

        nodes = self.init_nodes()
        functions = self.init_functions()
        self.modify_network_delays()
        inter_node_delays = self.data.node_delay_matrix
        self.env = cem(alpha=self.args['alpha'],function_instances=functions, node_instances=nodes,
                       inter_node_delays=inter_node_delays, data=self.data,metric_tracking=True)

    def init_agent(self):

        self.agent = PPO("MultiInputPolicy", self.env,
                         verbose=1, gamma=1,ent_coef=0.15,clip_range=0.8)

    def results(self):
        return self.reward_history,self.delay_history,self.inference_time,self.sample_workloads,self.cost_history

    def score(self) -> float:
        return self.env.Value()

    def load_data(self, data):
        self.data = data
        for key,value in vars(self.data).items():
            print(key,value)

        self.log("Initializing Environment...")
        self.init_environment()
        self.log("Initializing Agent...")
        self.init_agent()

    def log(self, msg: str):
        if self.verbose:
            print(f"{datetime.datetime.now()}: {msg}")

    def solve(self):

        # print(f"Solving for {self.args['alpha']}...")
        # self.data.workload_matrix = self.workload_generator.reset_timestamp()
        # for _ in range(50):
        #     self.agent.learn(total_timesteps=2000)
        #     self.data.workload = self.workload_generator.get_workload()
        # print("Solved!")
        # self.agent.save(f'ppo_model_{self.args["alpha"]*10}_comparisons_solutiontraining')
        # self.data.workload_matrix=self.workload_generator.reset_timestamp()
        # self.sample_workloads.append([self.data.workload_matrix,self.workload_generator.timestamp])
        # del self.agent
        self.agent = PPO.load(f'Models/ppo_model_{self.args["alpha"]*10}_comparisons')

        # # plt.plot(np.convolve(self.env.metrics.total_reward_history, np.ones((10,))/10, mode='valid'),label='Trained',color='blue')
        # # plt.title('Rolling Average of Reward')
        # # plt.legend()
        # # plt.savefig('reward_trained.png')
        # # plt.show()
        # # self.data.workload_matrix = test_workload

        # alpha = self.args['alpha']
        #save reward_training in a pickle file
        # with open(f'delay_training_{alpha}.pkl','wb') as f:
        #     pickle.dump(self.env.metrics.total_delay_history,f)

        ##inference
        print("Inference...")
        
        for i in range(1):
            obs,_ = self.env.reset()
            done = False
            total_delay=0
            total_reward=0
            total_cost = 0
            time_inf = 0
            while not done:
                f = self.env.function_queue[0]
                time_start = time.time()
                action, _states = self.agent.predict(obs)
                time_end = time.time()
                time_inf += time_end - time_start
                obs, reward, done, info,_ = self.env.step(action)
                total_reward+=reward
                total_delay+=self.env.calculate_total_delay_per_function_instance(f)
                total_cost+=self.env.calculate_cost(f)
            self.cost_history.append(total_cost)
            self.reward_history.append(total_reward)
            self.inference_time.append(time_inf)
            # self.data.workload_matrix=self.workload_generator.get_workload()
            self.delay_history.append(total_delay)
            self.sample_workloads.append([self.data.workload_matrix,self.workload_generator.timestamp])
                
        ##save metrics in a pickle file
        
        # with open(f"metrics_{alpha}.pkl",'wb') as f:
        #     pickle.dump([self.reward_history,self.delay_history,self.inference_time,self.sample_workloads,self.cost_history],f)

        # for current_function in range(len(self.data.functions)):
        #     for i in range(len(self.data.nodes)):
        #         for j in range(len(self.data.nodes)):
        #             print(
        #                 f"x[{i}][{current_function}][{j}]: {self.env.x[i,current_function,j].solution_value()}")
        
        
