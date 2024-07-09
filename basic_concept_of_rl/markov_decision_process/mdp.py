import gym
import numpy as np
from IPython import display
from matplotlib import pyplot as plt
from setup import Maze


# %matplotlib inline

# class Mdp():
#     def __init__(self):
#         return
#
#
# def reset():
#     return 0  # Start in the top-left corner of the maze

def show_frame(state):
    frame = env.render(mode='rgb_array')
    plt.axis('off')
    plt.title(f"State: {state}")
    plt.imshow(frame)
    plt.show()


env = Maze()  # initialize the environment with Maze environment
initial_state = env.reset()
print(f"The new episode will start in state: {initial_state}")

show_frame(initial_state)

action = 2
next_state, reward, done, info = env.step(action)
print(f"After moving down 1 row, the agent is in state: {next_state}")
print(f"After moving down 1 row, we got a reward of: {reward}")
print("After moving down 1 row, the task is", "" if done else "not", "finished")

show_frame(next_state)

env.close()