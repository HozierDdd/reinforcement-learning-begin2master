import numpy as np
import matplotlib.pyplot as plt
from environments.maze import Maze
from setup import test_agent


class ValueIteration():

    def __init__(self):
        self.policy_probs = np.zeros((5, 5, 4))
        self.state_values = np.zeros((5, 5))
        self.theta = 1e-6
        self.gamma = 0.99

    @staticmethod
    def show_init_environment():
        """show the initialized maze environment"""
        env = Maze()
        frame = env.render(mode='rgb_array')
        plt.figure(figsize=(6, 6))
        plt.axis('off')
        plt.imshow(frame)
        plt.show()
        print(f"Observation space shape: {env.observation_space.nvec}")
        print(f"Number of actions: {env.action_space.n}")

    @staticmethod
    def random_policy(state):
        policy_probs = np.full((5, 5, 4), 0.25)  # maze environment has 5 rows, 5 columns, so the number of total
        # state is 25. Each state has 4 actions can choose.
        return policy_probs[state]

    def value_iteration(self, policy_probs, state_values):
        delta = float('inf')

        while delta > self.theta:
            delta = 0
            for row in range(5):
                for col in range(5):
                    old_value = state_values[(row, col)]
                    action_probs = None
                    max_qsa = float('-inf')

                    for action in range(4):
                        next_state, reward, _, _ = env.simulate_step((row, col), action)
                        qsa = reward + self.gamma * state_values[next_state]
                        if qsa > max_qsa:
                            max_qsa = qsa
                            action_probs = np.zeros(4)
                            action_probs[action] = 1.

                    state_values[(row, col)] = max_qsa
                    policy_probs[(row, col)] = action_probs

                    delta = max(delta, abs(max_qsa - old_value))


# value_iteration(policy_probs, state_values)


if __name__ == '__main__':
    """Initialize the environment"""
    env = Maze()
    vi = ValueIteration()
    """show the initial environment"""
    # vi.show_init_environment()
    """test the random policy"""
    # action_probabilities = vi.random_policy((0, 0))
    # for action, prob in zip(range(4), action_probabilities):
    #     print(f"Probability of taking action {action}: {prob}")
    """apply the random policy in maze"""
    test_agent(env, vi.random_policy, episodes=1)
