import numpy as np
import matplotlib.pyplot as plt


class metrics_monitor:
    def __init__(self):
        self.total_delay_history = []
        self.total_cost_history = []
        self.total_disruption_history = []
        self.total_reward_history = []
        # self.total_energy_history = [] #TODO: add energy monitoring

    def __str__(self):
        return f"Total Delay History: {self.total_delay_history}, Total Cost History: {self.total_cost_history}, Total Disruption History: {self.total_disruption_history}, Total Reward History: {self.total_reward_history}"

    def record_metrics(self, total_delay, total_cost, total_disruption, total_reward):
        self.total_delay_history.append(total_delay)
        self.total_cost_history.append(total_cost)
        self.total_disruption_history.append(total_disruption)
        self.total_reward_history.append(total_reward)
        # self.total_energy_history.append(total_energy)
        return self

    def get_metrics(self):
        return self.total_delay_history, self.total_cost_history, self.total_disruption_history, self.total_reward_history

    def set_scores(self):
        # return a normalized score between 0 and 10 for each metric
        # the score reflects the performance of the system
        
        # TODO: add a proper scoring mechanism    
        return 0
    
    def plot_metrics(self, save_plots=True, window_size=10, save_dir="./"):
        # plot all the metrics in seperate subplots and save the plot
        # also plot a rolling average of the metrics along with the actual metrics

        fig, axs = plt.subplots(2, 2, figsize=(15, 15))
        fig.suptitle('Metrics Monitoring')

        rolling_window = np.ones(window_size)/window_size

        # plot total delay
        axs[0, 0].plot(self.total_delay_history, label='Total Delay')
        axs[0, 0].plot(np.convolve(self.total_delay_history, rolling_window,
                       mode='valid'), label='Total Delay Rolling Average')
        axs[0, 0].set_title('Total Delay')
        axs[0, 0].set_xlabel('Epochs')
        axs[0, 0].set_ylabel('Total Delay')
        axs[0, 0].legend()

        # plot total cost
        axs[0, 1].plot(self.total_cost_history, label='Total Cost')
        axs[0, 1].plot(np.convolve(self.total_cost_history, rolling_window,
                       mode='valid'), label='Total Cost Rolling Average')
        axs[0, 1].set_title('Total Cost')
        axs[0, 1].set_xlabel('Epochs')
        axs[0, 1].set_ylabel('Total Cost')
        axs[0, 1].legend()

        # plot total disruption
        axs[1, 0].plot(self.total_disruption_history, label='Total Disruption')
        axs[1, 0].plot(np.convolve(self.total_disruption_history, rolling_window,
                       mode='valid'), label='Total Disruption Rolling Average')
        axs[1, 0].set_title('Total Disruption')
        axs[1, 0].set_xlabel('Epochs')
        axs[1, 0].set_ylabel('Total Disruption')
        axs[1, 0].legend()

        # plot total reward
        axs[1, 1].plot(self.total_reward_history, label='Total Reward')
        axs[1, 1].plot(np.convolve(self.total_reward_history, rolling_window,
                       mode='valid'), label='Total Reward Rolling Average')
        axs[1, 1].set_title('Total Reward')
        axs[1, 1].set_xlabel('Epochs')
        axs[1, 1].set_ylabel('Total Reward')
        axs[1, 1].legend()

        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.show()  
        
        if save_plots:
            plt.savefig(save_dir)
