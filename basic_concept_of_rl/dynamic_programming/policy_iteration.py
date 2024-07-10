import numpy as np

from environments.maze import Maze
from utils import Utils


class PolicyIterator():
    def __init__(self):
        self.env = Maze()
        self.utils = Utils(self.env)
        self.policy_probs = np.full(shape=(5, 5, 4), fill_value=0.25)
        self.state_values = np.zeros((5, 5))
        self.theta = 1e-6
        self.gamma = 0.99

    def policy(self, state):
        return self.policy_probs[state]

    def policy_evaluation(self):
        """

        :return:
        """
        delta = float('inf')
        frame = self.env.render(mode='rgb_array')
        while delta > self.theta:
            delta = 0
            for row in range(5):
                for col in range(5):
                    old_value = self.state_values[(row, col)]
                    new_value = 0
                    action_probabilities = self.policy_probs[(row, col)]

                    for action, prob in enumerate(action_probabilities):
                        next_state, reward, _, _ = self.env.simulate_step((row, col), action)
                        new_value += prob * (reward + self.gamma * self.state_values[next_state])

                    self.state_values[(row, col)] = new_value

                    delta = max(delta, abs(old_value - new_value))
            # self.utils.plot_state_values(self.state_values, frame)
            # self.utils.plot_policy(self.policy_probs, frame)

    def policy_improvement(self):

        policy_stable = True
        for row in range(5):
            for col in range(5):
                old_action = self.policy_probs[(row, col)].argmax()

                new_action = None
                max_qsa = float("-inf")

                for action in range(4):
                    next_state, reward, _, _ = self.env.simulate_step((row, col), action)
                    qsa = reward + self.gamma * self.state_values[next_state]
                    if qsa > max_qsa:
                        max_qsa = qsa
                        new_action = action

                action_probs = np.zeros(4)
                action_probs[new_action] = 1.
                self.policy_probs[(row, col)] = action_probs

                if new_action != old_action:
                    policy_stable = False

        return policy_stable

    def policy_iteration(self):
        policy_stable = False

        while not policy_stable:
            self.policy_evaluation()

            policy_stable = self.policy_improvement()
        frame = pi.env.render(mode='rgb_array')
        self.utils.plot_state_values(self.state_values, frame)
        self.utils.plot_policy(self.policy_probs, frame)


if __name__ == '__main__':
    pi = PolicyIterator()
    pi.policy_iteration()
    pi.utils.test_agent(pi.policy)
