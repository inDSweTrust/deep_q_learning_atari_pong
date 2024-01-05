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

from agent import DQNAgent
from memory import ReplayBuffer
from model import DeepQNetwork
from preprocess import RepeatActionAndMaxFrame, PreprocessFrame, StackFrames, make_env

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def plot_learning_curve(x, scores, epsilons, filename, lines=None):
    fig=plt.figure()
    ax=fig.add_subplot(111, label="1")
    ax2=fig.add_subplot(111, label="2", frame_on=False)

    ax.plot(x, epsilons, color="C0")
    ax.set_xlabel("Training Steps", color="C0")
    ax.set_ylabel("Epsilon", color="C0")
    ax.tick_params(axis='x', colors="C0")
    ax.tick_params(axis='y', colors="C0")

    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
	    running_avg[t] = np.mean(scores[max(0, t-20):(t+1)])

    ax2.scatter(x, running_avg, color="C1")
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    ax2.set_ylabel('Score', color="C1")
    ax2.yaxis.set_label_position('right')
    ax2.tick_params(axis='y', colors="C1")

    if lines is not None:
        for line in lines:
            plt.axvline(x=line)

    plt.savefig(filename)

    
class Timer:
    global_times = collections.defaultdict(list)
    def __init__(self, name):
        self.name = name
        self.start = time.time()
    def done(self):
        end = time.time()
        Timer.global_times[self.name].append(end - self.start)
        
    
if __name__ == "__main__":
    all_rewards = []
    eps_history = []
    exp_history = []
    steps_array = []
    
    env = make_env("ALE/Pong-v5")

    best_score = -np.inf
    load_checkpoint = False
    num_episodes = 100000
    max_steps = 100000

    learn_every_n = 4
    learn_steps = 1

    agent = DQNAgent(
        gamma=0.97, 
        epsilon=1.0, 
        lr=0.00025,          
        input_dims=(env.observation_space.shape), 
        n_actions=env.action_space.n,
        mem_size=30000,
        eps_min=0.001,
        batch_size=64,
        eps_dec=0.9999,
        chkpt_dir='./output',
        algo='DQNAgent',
        env_name='ALE/Pong-v5',
        device=device
    )

    if load_checkpoint:
        agent.load_models()

    n_steps = 0

    steps_per_game_list = []

    pbar = tqdm(range(num_episodes))
    for i in pbar:
        done = False
        observation = env.reset()

        rewards = 0
        sum_rewards = 0

        total_timer = Timer('total')

        steps_per_game = 0 
        while not done and steps_per_game < max_steps:
            timer = Timer('action_choice')
            action = agent.choose_action(observation)
            timer.done()

            timer = Timer('step')
            observation_, reward, done, _, _= env.step(action)
            timer.done()

            rewards += reward

            if not load_checkpoint:
                timer = Timer('store')
                agent.store_transition(observation, action, reward, observation_, done)
                timer.done()

                # Learn every n steps 
                if steps_per_game % learn_every_n == 0 or done:
                    timer = Timer('learn')
                    agent.learn()
                    timer.done()

            observation = observation_
            n_steps += 1
            steps_per_game += 1


        steps_per_game_list.append(steps_per_game)
        all_rewards.append(rewards)
        steps_array.append(n_steps)

        avg_score = np.mean(all_rewards[-100:])
        pbar.set_description("Avg Score %.2f, Curr Score: %.2f, ep: %.4f" % (avg_score, rewards, agent.epsilon))

        if avg_score > best_score:
            if not load_checkpoint:
                timer = Timer('save')
                agent.save_models()
                timer.done()
            best_score = avg_score

        total_timer.done()

        eps_history.append(agent.epsilon)
        exp_history.append((agent.explores, agent.exploits))
    
    # Plot sum of rewards and epsilon 
    x = [i+1 for i in range(len(all_rewards))]
    plot_learning_curve(x, all_rewards, eps_history, "temp")