import torch
import numpy as np
import random

from memory import ReplayBuffer
from model import DeepQNetwork

class DQNAgent(object):
    def __init__(self, gamma, epsilon, lr, n_actions, input_dims,
                 mem_size, batch_size, eps_min=0.01, eps_dec=5e-7,
                 replace=1000, algo=None, env_name=None, chkpt_dir='tmp/dqn', device=None):
        
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.replace_target_cnt = replace
        self.algo = algo
        self.env_name = env_name
        self.chkpt_dir = chkpt_dir
        self.action_space = [i for i in range(n_actions)]
        self.learn_step_counter = 0
        self.indices = np.arange(self.batch_size)
        
        # Initialize replay buffer (mem)
        self.memory = ReplayBuffer(mem_size, input_dims, n_actions, device=device)
        
        # Initialize main netowrk for evaluation 
        self.eval_network = DeepQNetwork(self.lr, self.n_actions,
                                    input_dims=self.input_dims,
                                    name=self.algo + '_eval_network',
                                    chkpt_dir=self.chkpt_dir, device=device)
        
        # Initialize target network for action selection 
        self.target_network = DeepQNetwork(self.lr, self.n_actions,
                                    input_dims=self.input_dims,
                                    name=self.algo + '_target_network',
                                    chkpt_dir=self.chkpt_dir, device=device)
        self.explores = 0
        self.exploits = 0
        
        

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            with torch.inference_mode():
                state = torch.tensor(observation.reshape((1,) + observation.shape),dtype=torch.float, device=self.eval_network.device)
                actions = self.eval_network.forward(state)
                action = torch.argmax(actions).item()
            self.exploits += 1
        else:
            action = random.choices(self.action_space)[0]
            self.explores += 1

        return action

    def store_transition(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def sample_memory(self):
        # Returns transition samples from memory as torch tensors 
        states, actions, rewards, next_states, dones = \
                                self.memory.sample_buffer(self.batch_size)

        return states, actions, rewards, next_states, dones

    def update_target_network(self):
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.target_network.load_state_dict(self.eval_network.state_dict())

    def decay_epsilon(self):
        self.epsilon = self.epsilon * self.eps_dec \
                           if self.epsilon > self.eps_min else self.eps_min

    def save_models(self):
        self.eval_network.save_checkpoint()
        self.target_network.save_checkpoint()

    def load_models(self):
        self.eval_network.load_checkpoint()
        self.target_network.load_checkpoint()

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
           
        self.eval_network.optimizer.zero_grad()

        self.update_target_network()

        states, actions, rewards, states_, dones = self.sample_memory()

        with torch.no_grad():
            q_next = self.target_network.forward(states_).detach().max(dim=1)[0]

        q_next[dones] = 0.0

        q_target = rewards + self.gamma*q_next
        
        q_pred = self.eval_network.forward(states)[self.indices, actions]

        loss = self.eval_network.loss(q_target, q_pred)
        loss.backward()
        self.eval_network.optimizer.step()
        self.learn_step_counter += 1

        self.decay_epsilon()