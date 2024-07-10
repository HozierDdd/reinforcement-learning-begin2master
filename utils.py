from typing import Tuple, Dict, Optional, Iterable

import numpy as np
import matplotlib
from matplotlib import animation
from matplotlib import pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
import torch

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
    def plot_state_values(state_values, frame):
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

    def seed_everything(self, seed: int = 42) -> None:
        self.env.seed(seed)
        self.env.action_space.seed(seed)
        self.env.observation_space.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)

    @staticmethod
    def plot_stats(stats):
        rows = len(stats)
        cols = 1

        fig, ax = plt.subplots(rows, cols, figsize=(12, 6))

        for i, key in enumerate(stats):
            vals = stats[key]
            vals = [np.mean(vals[i - 10:i + 10]) for i in range(10, len(vals) - 10)]
            if len(stats) > 1:
                ax[i].plot(range(len(vals)), vals)
                ax[i].set_title(key, size=18)
            else:
                ax.plot(range(len(vals)), vals)
                ax.set_title(key, size=18)
        plt.tight_layout()
        plt.show()

    def plot_cost_to_go(self, q_network, xlabel=None, ylabel=None):
        highx, highy = self.env.observation_space.high
        lowx, lowy = self.env.observation_space.low
        X = torch.linspace(lowx, highx, 100)
        Y = torch.linspace(lowy, highy, 100)
        X, Y = torch.meshgrid(X, Y)

        q_net_input = torch.stack([X.flatten(), Y.flatten()], dim=-1)
        Z = - q_network(q_net_input).max(dim=-1, keepdim=True)[0]
        Z = Z.reshape(100, 100).detach().numpy()
        X = X.numpy()
        Y = Y.numpy()

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z, cmap='jet', linewidth=0, antialiased=False)
        fig.colorbar(surf, shrink=0.5, aspect=5)
        ax.set_xlabel(xlabel, size=14)
        ax.set_ylabel(ylabel, size=14)
        ax.set_title("Estimated cost-to-go", size=18)
        plt.tight_layout()
        plt.show()

    def plot_max_q(self, q_network, xlabel=None, ylabel=None, action_labels=[]):
        highx, highy = self.env.observation_space.high
        lowx, lowy = self.env.observation_space.low
        X = torch.linspace(lowx, highx, 100)
        Y = torch.linspace(lowy, highy, 100)
        X, Y = torch.meshgrid(X, Y)
        q_net_input = torch.stack([X.flatten(), Y.flatten()], dim=-1)
        Z = q_network(q_net_input).argmax(dim=-1, keepdim=True)
        Z = Z.reshape(100, 100).T.detach().numpy()
        values = np.unique(Z.ravel())
        values.sort()

        plt.figure(figsize=(5, 5))
        plt.xlabel(xlabel, size=14)
        plt.ylabel(ylabel, size=14)
        plt.title("Optimal action", size=18)

        im = plt.imshow(Z, cmap='jet')
        colors = [im.cmap(im.norm(value)) for value in values]
        patches = [mpatches.Patch(color=color, label=label) for color, label in zip(colors, action_labels)]
        plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.tight_layout()
