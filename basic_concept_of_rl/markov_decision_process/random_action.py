import random

import gym
import numpy as np
from IPython import display
from matplotlib import pyplot as plt

from setup import Maze
from setup import show_frame


env = Maze()  # initialize the environment with Maze environment
initial_state = env.reset()
print(f"The new episode will start in state: {initial_state}")
show_frame(env, initial_state)

done = False
while not done:

    action = random.choice([0, 1, 2, 3])
    next_state, reward, done, info = env.step(action)
    print(f"After moving down 1 row, the agent is in state: {next_state}")
    print(f"After moving down 1 row, we got a reward of: {reward}")
    print("After moving down 1 row, the task is", "" if done else "not", "finished")
    show_frame(env, next_state)

    state = next_state



env.close()