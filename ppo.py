"""
This script trains a PPO agent to learn the optimal function placement in a community of nodes.
"""

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
import env_ppo as env
import numpy as np
import random
import matplotlib.pyplot as plt
import copy

## Define the function instances
f1 = env.Function({'function_id': 0, 'required_cpu_memory': 1, 'required_gpu_memory': 0,
                   'origin_node': 0, 'total_instruction': 10, 'response_time': 10, 'phi': 0.5})
f2 = env.Function({'function_id': 1, 'required_cpu_memory': 10, 'required_gpu_memory': 2,
                   'origin_node': 1, 'total_instruction': 20, 'response_time': 55, 'phi': 0.5})
f3 = env.Function({'function_id': 2, 'required_cpu_memory': 9, 'required_gpu_memory': 6,
                   'origin_node': 1, 'total_instruction': 300, 'response_time': 200, 'phi': 0.5})

## Define the node instances with random attributes
nodes = []
for i in range(20):
    total_cpu_cores = random.randint(8, 32)
    total_cpu_memory = total_cpu_cores * \
        random.randint(3, 4)  # e.g., 3-4 GB per core
    total_gpu_cores = random.choice([0, random.randint(5, 15)])
    total_gpu_memory = total_gpu_cores if total_gpu_cores > 0 else 0
    processing_power = random.randint(5, 15)

    node_attributes = {
        'node_id': i,
        'total_cpu_cores': total_cpu_cores,
        'total_cpu_memory': total_cpu_memory,
        'total_gpu_cores': total_gpu_cores,
        'total_gpu_memory': total_gpu_memory,
        'processing_power': processing_power
    }

    nodes.append(env.Node(node_attributes))
    
## Define more environment variables
network_delays = np.random.uniform(1, 10, [len(nodes), len(nodes)])
functions = [f1, f2, f3]
metric = env.Metrics(nodes, functions, network_delays)
community_env = env.Community(functions, nodes, metric)

## Create a baseline environment with random actions
baseline_nodes = copy.deepcopy(nodes)
baseline_functions = copy.deepcopy(functions)
baseline_network_delays = copy.deepcopy(network_delays)
baseline_metric = env.Metrics(baseline_nodes, baseline_functions, baseline_network_delays)
baseline_env = env.Community(baseline_functions, baseline_nodes, baseline_metric)

## Define the action and state space
state_shape = spaces.utils.flatdim(community_env.observation_space)
action_shape = spaces.utils.flatdim(community_env.action_space)

## Define the PPO model and train it
model = PPO("MultiInputPolicy", community_env, verbose=1)
model.learn(total_timesteps=50000)
model.save("ppo_function_placement")

del model ## Delete the model to free up memory

## Load the trained model and test it
model = PPO.load("ppo_function_placement")
done = False
obs, _ = community_env.reset()
reward = 0
while not done:
    action, _states = model.predict(obs)
    obs, rewards, done, info, _ = community_env.step(action)
    reward += rewards

## Test the baseline environment
for _ in range(len(community_env.delay_history)):
    
    obs,_ = baseline_env.reset()
    done = False
    reward = 0
    while not done:
        #generate 20 length random boolean array
        action = np.random.randint(0, 2, size=20)
        obs, rewards, done, info, _ = baseline_env.step(action)
        reward += rewards

## Plot the results with a moving average window
window = 10

##plot reward
plt.figure(figsize=(12, 8))
plt.plot(np.convolve(community_env.reward_history,
                        np.ones(window), 'valid') / window, label="RL Agent")
plt.xlabel("Time")
plt.ylabel("Reward")
plt.title("Reward over time")
plt.legend()
plt.savefig("reward.png")
plt.show()

##plot metrics
plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
plt.plot(np.convolve(community_env.delay_history,
         np.ones(window), 'valid') / window, label="RL Agent")
plt.plot(np.convolve(baseline_env.delay_history,
         np.ones(window), 'valid') / window,label='Random Agent')  # Superimpose baseline delay history
plt.xlabel("Time")
plt.ylabel("Delay")
plt.title("Delay over time")
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(np.convolve(community_env.cost_history,
                     np.ones(window), 'valid') / window, label="RL Agent")
plt.plot(np.convolve(baseline_env.cost_history,
                     np.ones(window), 'valid') / window,label='Random Agent')  # Superimpose baseline cost history
plt.xlabel("Time")
plt.ylabel("Cost")
plt.title("Cost over time")
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(np.convolve(community_env.disruptions_history,
                     np.ones(window), 'valid') / window, label="RL Agent")
plt.plot(np.convolve(baseline_env.disruptions_history,
                     np.ones(window), 'valid') / window,label = 'Random Agent')  # Superimpose baseline disruptions history
plt.xlabel("Time")
plt.ylabel("Disruptions")
plt.title("Disruptions over time")
plt.legend()

plt.tight_layout()
plt.savefig("metrics.png")
plt.show()

print("Testing done!")
print("Total reward:", reward)
