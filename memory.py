import torch 
import numpy as np
import random

class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, n_actions, device):
        self.mem_size = max_size
        self.mem_cntr = 0
        
        self.device = device
        
        self.state_memory = torch.empty((self.mem_size, *input_shape), dtype=torch.float, device=self.device)
        self.new_state_memory = torch.empty((self.mem_size, *input_shape), dtype=torch.float, device=self.device)
        self.action_memory = torch.empty(self.mem_size, dtype=torch.int, device=self.device)
        self.reward_memory = torch.empty(self.mem_size, dtype=torch.float, device=self.device)
        self.terminal_memory = torch.empty(self.mem_size, dtype=torch.bool, device=self.device)
        
        
    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        
        self.state_memory[index] = torch.tensor(state, device=self.device)
        self.new_state_memory[index] = torch.tensor(state_, device=self.device)
        self.action_memory[index] = torch.tensor(action, device=self.device)
        self.reward_memory[index] = torch.tensor(reward, device=self.device)
        self.terminal_memory[index] = torch.tensor(done, device=self.device)
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.array(random.sample(range(max_mem), batch_size))

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]
        
        return states, actions, rewards, states_, terminal
    
    def is_full(self):
        return self.mem_cntr > self.mem_size

    