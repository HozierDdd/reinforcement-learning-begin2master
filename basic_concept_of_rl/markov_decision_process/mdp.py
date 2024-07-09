import numpy as np
from matplotlib import pyplot as plt
from setup import Maze, display_video
from setup import show_frame


def initialize_environment():
    """Initialize the environment Maze"""
    env = Maze()
    initial_state = env.reset()
    return env, initial_state


def show_trajectory():
    env, state = initialize_environment()
    trajectory = []
    show_frame(env=env, state=state)
    for _ in range(3):
        action = env.action_space.sample()
        next_state, reward, done, extra_info = env.step(action)
        trajectory.append([state, action, reward, done, next_state])
        state = next_state
        show_frame(env=env, state=state)
    env.close()

    print(f"Congrats! You just generated your first trajectory:\n{trajectory}")


def show_episode():
    env, state = initialize_environment()
    episode = []
    done = False
    show_frame(env=env, state=state)
    while not done:
        action = env.action_space.sample()
        next_state, reward, done, extra_info = env.step(action)
        episode.append([state, action, reward, done, next_state])
        state = next_state
        show_frame(env=env, state=state)
    env.close()

    print(f"Congrats! You just generated your first episode:\n{episode}")


def show_total_reward():
    env, state = initialize_environment()
    done = False
    gamma = 0.99
    G_0 = 0
    t = 0
    while not done:
        action = env.action_space.sample()
        _, reward, done, _ = env.step(action)
        G_0 += gamma ** t * reward
        t += 1
    env.close()
    print(
        f"""It took us {t} moves to find the exit,
        and each reward r(s,a)=-1, so the return amounts to {G_0}""")


def random_policy(state):
    """Using numpy arrays when mathematical operations is needed(sum, average, array multiplication, etc)
    Using list when iterate in 'items' (strings, files, etc) is needed."""
    return np.array([0.25] * 4)


def random_policy_episode():
    env, state = initialize_environment()
    action_probabilities = random_policy(state)
    objects = ('Up', 'Right', 'Down', 'Left')
    y_pos = np.arange(len(objects))

    plt.bar(y_pos, action_probabilities, alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('P(a|s)')
    plt.title('Random Policy')
    plt.tight_layout()

    plt.show()


def test_agent(policy):
    frames = []
    env, state = initialize_environment()
    done = False
    frames.append(env.render(mode="rgb_array"))
    while not done:
        action_probabilities = policy(state)
        action = np.random.choice(range(4), 1, p=action_probabilities)
        next_state, reward, done, extra_info = env.step(action)
        img = env.render(mode="rgb_array")
        frames.append(img)
        state = next_state

    return display_video(frames)


if __name__ == '__main__':
    # env, initial_state = initialize_environment()
    # print(f"For example, the initial state is: {env.reset()}")
    # print(f"The space state is of type: {env.observation_space}")
    # print(f"An example of a valid action is: {env.action_space.sample()}")
    # print(f"The action state is of type: {env.action_space}")
    # show_trajectory()
    # show_episode()
    # show_total_reward()
    random_policy_episode()
    # test_agent(random_policy)