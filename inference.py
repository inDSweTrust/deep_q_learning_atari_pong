import os
import time 
import collections
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

import gymnasium as gym
from gymnasium import wrappers
from gymnasium.utils.save_video import save_video

from agent import DQNAgent
from memory import ReplayBuffer
from model import DeepQNetwork
from preprocess import RepeatActionAndMaxFrame, PreprocessFrame, StackFrames, make_env

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

if __name__ == "__main__":
    env_to_wrap = make_env("ALE/Pong-v5")
    env = wrappers.RecordVideo(env_to_wrap, "./videos")


    load_checkpoint = True
    num_episodes = 100


    agent = DQNAgent(
        gamma=0.99, 
        epsilon=0, 
        lr=0.00025,          
        input_dims=(env.observation_space.shape), 
        n_actions=env.action_space.n,
        mem_size=1,
        eps_min=0,
        batch_size=64,
        replace=1000, 
        eps_dec=0.999991,
        chkpt_dir='./model_weights',
        algo='DQNAgent',
        env_name='ALE/Pong-v5',
        device=device
    )

    agent.load_models()
    
    all_rewards = []
    pbar = tqdm(range(num_episodes))
    
    steps_per_game_min = 1e6
    steps_per_game_max = -1
    
    min_score = 100
    max_score = -100
    
    ep = 1
    
    for i in pbar:
        done = False
        observation = env.reset()

        rewards = 0
        steps_per_game = 0

        # View environment
        env.render()
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, _, _= env.step(action)

            rewards += reward

            observation = observation_
            steps_per_game += 1
            ep += 1
            
        # Record game steps 
        steps_per_game_min = min(steps_per_game_min, steps_per_game)
        steps_per_game_max = max(steps_per_game_max, steps_per_game)
        # Record scores 
        min_score = min(min_score, rewards)
        max_score = max(max_score, rewards)
        all_rewards.append(rewards)

        avg_score = np.mean(all_rewards[-100:])
        pbar.set_description("Game Score %.2f, Avg Score %.2f, steps, %s, ep: %.4f" % (rewards, avg_score, steps_per_game, agent.epsilon))
    
    print("100 games played!")
    print(f"Min steps to win game: {steps_per_game_min}")
    print(f"Max steps to win game: {steps_per_game_max}")
    print(f"Min score: {min_score}")
    print(f"Max score: {max_score}")

    env.close()
    env_to_wrap.close()
