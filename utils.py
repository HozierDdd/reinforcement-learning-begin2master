from typing import Tuple, Dict, Optional, Iterable

import numpy as np
import matplotlib
from matplotlib import animation
from matplotlib import pyplot as plt
import seaborn as sns

from IPython.display import HTML

import gym
from gym import spaces
from gym.error import DependencyNotInstalled

import pygame
from pygame import gfxdraw


class Utils():
    def __init__(self, environment):
        self.env = environment

    @staticmethod
    def save_video(frames):
        """save video to file"""
        # plt.rcParams['animation.ffmpeg_path'] = r'C:\ffmpeg-7.0.1-full_build\bin\ffmpeg.exe'
        fig, ax = plt.subplots()
        ax.set_axis_off()
        im = ax.imshow(frames[0])

        def update(frame):
            im.set_data(frame)
            return [im]

        anim = animation.FuncAnimation(fig, update, frames=frames, interval=50, blit=True)
        anim.save('output_video.mp4', writer='ffmpeg')  # Save the animation to a file
        plt.close(fig)

    @staticmethod
    def display_video(frames):
        """display video directly"""
        # plt.rcParams['animation.ffmpeg_path'] = r'C:\ffmpeg-7.0.1-full_build\bin\ffmpeg.exe'
        matplotlib.use('TkAgg')  # Use TkAgg for interactive plots
        fig, ax = plt.subplots()
        ax.set_axis_off()
        ax.set_aspect('equal')
        ax.set_position([0, 0, 1, 1])
        im = ax.imshow(frames[0])

        def update(frame):
            im.set_data(frame)
            return [im]

        anim = animation.FuncAnimation(fig, update, frames=frames, interval=50, blit=True, repeat=False)
        plt.show()

    def show_frame(self, state):
        """show a frame in the given environment"""
        frame = self.env.render(mode='rgb_array')
        plt.axis('off')
        plt.title(f"State: {state}")
        plt.imshow(frame)
        plt.show()

    @staticmethod
    def plot_policy(probs_or_qvals, frame, action_meanings=None):
        """
        plot policy on every state of the given frame in maze
        :param probs_or_qvals: policy on every state of the given frame,
        or let's say, the probability of choosing each action in every state
        :param frame:
        :param action_meanings:
        :return: None
        """
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
        plt.show()

    @staticmethod
    def plot_values(state_values, frame):
        """
        visualize the state value
        :param state_values:
        :param frame:
        :return:
        """
        f, axes = plt.subplots(1, 2, figsize=(10, 4))
        sns.heatmap(state_values, annot=True, fmt=".2f", cmap='coolwarm',
                    annot_kws={'weight': 'bold', 'size': 12}, linewidths=2, ax=axes[0])
        axes[1].imshow(frame)
        axes[0].axis('off')
        axes[1].axis('off')
        plt.tight_layout()
        plt.show()

    def test_agent(self, policy, episodes=10):
        """
        Test the policy in the environment
        :param policy:
        :param episodes:
        :return:
        """
        frames = []
        for episode in range(episodes):
            state = self.env.reset()
            done = False
            frames.append(self.env.render(mode="rgb_array"))

            while not done:
                p = policy(state)
                if isinstance(p, np.ndarray):  # IF the policy is np.ndarray
                    action = np.random.choice(4, p=p)  # Choose an action based on the probability distribution
                else:
                    action = p  # Assume the policy is a deterministic function
                next_state, reward, done, extra_info = self.env.step(action)
                img = self.env.render(mode="rgb_array")
                frames.append(img)
                state = next_state
                action_meaning = self.env.action_space.action_meanings[action]
                print(f"action:{action_meaning}, reward:{reward}, {'End' if done else 'Continue'}, state:{state}")

        return Utils.display_video(frames)
