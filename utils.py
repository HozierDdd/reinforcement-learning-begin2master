from typing import Tuple, Dict, Optional, Iterable

import numpy as np
import matplotlib
from matplotlib import animation
from matplotlib import pyplot as plt
from IPython.display import HTML

import gym
from gym import spaces
from gym.error import DependencyNotInstalled

import pygame
from pygame import gfxdraw


class Utils():
    def __init__(self):
        return

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
        im = ax.imshow(frames[0])

        def update(frame):
            im.set_data(frame)
            return [im]

        anim = animation.FuncAnimation(fig, update, frames=frames, interval=50, blit=True)
        plt.show()
