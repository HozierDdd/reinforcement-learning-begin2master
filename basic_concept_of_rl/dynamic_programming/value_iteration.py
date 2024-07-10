import numpy as np
import matplotlib.pyplot as plt

from environments.maze import Maze
from utils import Utils


class ValueIteration():

    def __init__(self):
        self.env = Maze()
        self.utils = Utils(self.env)
        self.policy_probs = np.zeros((5, 5, 4))
        self.state_values = np.zeros((5, 5))
        self.theta = 1e-6
        self.gamma = 0.99
        self.random_policy_probs = np.full((5, 5, 4), 0.25)  # maze environment has 5 rows, 5 columns, so the number
        # of total state is 25. Each state has 4 actions can choose.

    def show_init_environment(self):
        """show the initialized maze environment"""
        frame = self.env.render(mode='rgb_array')
        plt.figure(figsize=(6, 6))
        plt.axis('off')
        plt.imshow(frame)
        plt.show()
        print(f"Observation space shape: {self.env.observation_space.nvec}")
        print(f"Number of actions: {self.env.action_space.n}")

    def random_policy(self, state):
        return self.random_policy_probs[state]

    def policy(self, state):
        return self.policy_probs[state]

    def value_iteration(self):
        delta = float('inf')  # delta is measure the difference between the current state values and the updated
        # state values in each iteration

        while delta > self.theta:
            delta = 0
            for row in range(5):
                for col in range(5):
                    old_value = self.state_values[(row, col)]
                    action_probs = None
                    max_qsa = float('-inf')

                    for action in range(4):
                        next_state, reward, _, _ = self.env.simulate_step((row, col), action)
                        qsa = reward + self.gamma * self.state_values[next_state]
                        if qsa > max_qsa:
                            max_qsa = qsa
                            action_probs = np.zeros(4)
                            action_probs[action] = 1.

                    self.state_values[(row, col)] = max_qsa
                    self.policy_probs[(row, col)] = action_probs

                    delta = max(delta, abs(max_qsa - old_value))
        frame = self.env.render(mode='rgb_array')
        self.utils.plot_values(self.state_values, frame)
        self.utils.plot_policy(self.policy_probs, frame)
