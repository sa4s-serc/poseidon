import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap
from core import data_to_solver_input
from core.solvers import *
import pickle
import os

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')


def plot_average_delay_bar(poseidon_results, neptune_results, alpha_values):
    average_delays = {'Poseidon': [], 'Neptune': []}
    
    for alpha in alpha_values:
        poseidon_delay = np.mean(poseidon_results[alpha][1])  # Delay is the second metric
        neptune_delay = np.mean(neptune_results[alpha][1])
        
        average_delays['Poseidon'].append(poseidon_delay)
        average_delays['Neptune'].append(neptune_delay)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    width = 0.35  # Width of the bars
    x = np.arange(len(alpha_values))  # The label locations
    
    rects1 = ax.bar(x - width/2, average_delays['Poseidon'], width, label='Poseidon')
    rects2 = ax.bar(x + width/2, average_delays['Neptune'], width, label='Neptune')
    
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('Alpha Values')
    ax.set_ylabel('Average Delay')
    ax.set_title('Average Delay Comparison between Poseidon and Neptune')
    ax.set_xticks(x)
    ax.set_xticklabels([f'α={alpha}' for alpha in alpha_values])
    ax.legend()
    
    fig.tight_layout()
    
    plt.savefig("average_delay_comparison.png", dpi=300)
    plt.close()

def plot_delay_inference_reward(poseidon_results, neptune_results, alpha_values):
    fig, ax = plt.subplots(1, 1, figsize=(15,5))
    window_size = 5

    metrics = ['Delay']

    # for i, metric in enumerate(metrics):

    for alpha in alpha_values:
        smoothed_poseidon = moving_average(
            poseidon_results[alpha][1], window_size)
        ax.plot(smoothed_poseidon, label=f"Poseidon (α={alpha})")
        smoothed_neptune = moving_average(neptune_results[alpha][1], window_size)
        ax.plot(smoothed_neptune, label=f"Neptune (α={alpha})", linestyle='--', linewidth=2)

    ax.set_title("Comparison of Poseidon and Neptune - Delay")
    ax.set_xlabel("Workload iterations")
    ax.set_ylabel("Delay")
    ax.legend()
    sns.despine()

    plt.tight_layout()
    plt.savefig("delay_inference_reward_comparison.png", dpi=300)
    plt.close()

def plot_cost(poseidon_results, neptune_results, alpha_values):
    fig, ax = plt.subplots(1, 1, figsize=(15,5))
    window_size = 5

    metrics = ['Delay']

    # for i, metric in enumerate(metrics):

    for alpha in alpha_values:
        smoothed_poseidon = moving_average(
            poseidon_results[alpha][4], window_size)
        ax.plot(smoothed_poseidon, label=f"Poseidon (α={alpha})")
        smoothed_neptune = moving_average(neptune_results[alpha][4], window_size)
        ax.plot(smoothed_neptune, label=f"Neptune (α={alpha})", linestyle='--', linewidth=2)

    ax.set_title("Comparison of Poseidon and Neptune - Cost")
    ax.set_xlabel("Workload iterations")
    ax.set_ylabel("Cost")
    ax.legend()
    sns.despine()

    plt.tight_layout()
    plt.savefig("cost_comparison.png", dpi=300)
    plt.close()
    
def plot_inference(poseidon_results, neptune_results, alpha_values):
    fig, ax = plt.subplots(1, 1, figsize=(15,5))
    window_size = 5

    metrics = ['Delay']

    # for i, metric in enumerate(metrics):

    for alpha in alpha_values:
        smoothed_poseidon = moving_average(
            poseidon_results[alpha][2], window_size)
        ax.plot(smoothed_poseidon, label=f"Poseidon (α={alpha})")
        smoothed_neptune = moving_average(neptune_results[alpha][2], window_size)
        ax.plot(smoothed_neptune, label=f"Neptune (α={alpha})", linestyle='--', linewidth=2)

    ax.set_title("Comparison of Poseidon and Neptune - Inference Time")
    ax.set_xlabel("Workload iterations")
    ax.set_ylabel("Inference Time")
    ax.legend()
    sns.despine()

    plt.tight_layout()
    plt.savefig("inference_comparison_two.png", dpi=300)
    plt.close()


if input("Load?") == 'y':

    with open("Results/run_test_highinference/poseidon_results.pkl", "rb") as f:
            poseidon_results = pickle.load(f)

    with open("Results/run_test_highinference/neptune_results.pkl", "rb") as f:
            neptune_results = pickle.load(f)
            
else:
    
    
            
plot_average_delay_bar(poseidon_results, neptune_results, [1,0.5,0])
plot_delay_inference_reward(poseidon_results, neptune_results, [1,0.5,0])
plot_cost(poseidon_results, neptune_results, [1,0.5,0])
plot_inference(poseidon_results, neptune_results, [1,0.5,0])