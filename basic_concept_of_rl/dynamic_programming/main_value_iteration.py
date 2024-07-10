import numpy as np
from value_iteration import ValueIteration
from utils import Utils

if __name__ == '__main__':
    """Initialize the environment"""
    vi = ValueIteration()
    """show the initial environment"""
    # vi.show_init_environment()
    """test the random policy"""
    # action_probabilities = vi.random_policy((0, 0))
    # for action, prob in zip(range(4), action_probabilities):
    #     print(f"Probability of taking action {action}: {prob}")
    """apply the random policy in maze"""
    # vi.test_agent(vi.random_policy, episodes=1)
    # frame = vi.env.render(mode='rgb_array')
    # policy_probs = np.full((5, 5, 4), 0.25)
    # Utils.plot_policy(policy_probs, frame)
    # state_values = np.zeros(shape=(5, 5))
    # vi.utils.plot_values(state_values, frame)
    """value iteration function"""
    vi.value_iteration()
    vi.utils.test_agent(policy=vi.policy)
