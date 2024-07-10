from typing import Tuple, Dict, Optional, Iterable, Callable

import numpy as np
import seaborn as sns
import matplotlib
from matplotlib import animation
from matplotlib import pyplot as plt

from IPython.display import HTML

import gym
from gym import spaces
from gym.error import DependencyNotInstalled

import pygame
from pygame import gfxdraw


def plot_policy(probs_or_qvals, frame, action_meanings=None):
    if action_meanings is None:
        action_meanings = {0: 'U', 1: 'R', 2: 'D', 3: 'L'}
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    max_prob_actions = probs_or_qvals.argmax(axis=-1)
    probs_copy = max_prob_actions.copy().astype(object)
    for key in action_meanings:
        probs_copy[probs_copy == key] = action_meanings[key]
    sns.heatmap(max_prob_actions, annot=probs_copy, fmt='', cbar=False, cmap='coolwarm',
                annot_kws={'weight': 'bold', 'size': 12}, linewidths=2, ax=axes[0])
    axes[1].imshow(frame)
    axes[0].axis('off')
    axes[1].axis('off')
    plt.suptitle("Policy", size=18)
    plt.tight_layout()


def plot_values(state_values, frame):
    f, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.heatmap(state_values, annot=True, fmt=".2f", cmap='coolwarm',
                annot_kws={'weight': 'bold', 'size': 12}, linewidths=2, ax=axes[0])
    axes[1].imshow(frame)
    axes[0].axis('off')
    axes[1].axis('off')
    plt.tight_layout()


# def display_video(frames):
#     # Copied from: https://colab.research.google.com/github/deepmind/dm_control/blob/master/tutorial.ipynb
#     orig_backend = matplotlib.get_backend()
#     matplotlib.use('Agg')
#     fig, ax = plt.subplots(1, 1, figsize=(5, 5))
#     matplotlib.use(orig_backend)
#     ax.set_axis_off()
#     ax.set_aspect('equal')
#     ax.set_position([0, 0, 1, 1])
#     im = ax.imshow(frames[0])
#
#     def update(frame):
#         im.set_data(frame)
#         return [im]
#
#     anim = animation.FuncAnimation(fig=fig, func=update, frames=frames,
#                                    interval=50, blit=True, repeat=False)
#     return HTML(anim.to_html5_video())


def display_video(frames):
    orig_backend = matplotlib.get_backend()
    matplotlib.use('Agg')
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    matplotlib.use(orig_backend)
    ax.set_axis_off()
    ax.set_aspect('equal')
    ax.set_position([0, 0, 1, 1])
    im = ax.imshow(frames[0])

    def update(frame):
        im.set_data(frame)
        return [im]

    anim = animation.FuncAnimation(fig, update, frames=frames, interval=50, blit=True)
    plt.show()


# def display_video(frames):
#     """display video directly"""
#     # plt.rcParams['animation.ffmpeg_path'] = r'C:\ffmpeg-7.0.1-full_build\bin\ffmpeg.exe'
#     matplotlib.use('TkAgg')  # Use TkAgg for interactive plots
#     fig, ax = plt.subplots()
#     im = ax.imshow(frames[0])
#
#     def update(frame):
#         im.set_data(frame)
#         return [im]
#
#     anim = animation.FuncAnimation(fig, update, frames=frames, interval=50, blit=True)
#     plt.show()


def test_agent(env, policy, episodes=10):
    frames = []
    for episode in range(episodes):
        state = env.reset()
        done = False
        frames.append(env.render(mode="rgb_array"))

        while not done:
            p = policy(state)
            if isinstance(p, np.ndarray):
                action = np.random.choice(4, p=p)
            else:
                action = p
            next_state, reward, done, extra_info = env.step(action)
            img = env.render(mode="rgb_array")
            frames.append(img)
            state = next_state

    return display_video(frames)
