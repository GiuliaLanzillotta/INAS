""" The controller module will do what the old REINFORCE file did.
    It has to implement the following methods:
    - get_action(state)
    - update_policy(episode)
"""
import tensorflow as tf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

#constants
GAMMA = 0.9

class controller(nn.Module):

    def __init__(self, max_layers): # x is a state
        #TODO: create controller architecture
        super(controller,self).__init__()
        
        cells = []
        for layer in range(max_layers):
            cell1 = nn.LSTM(input_size = 1, hidden_size=3, num_layers=1)
            cell2 = nn.LSTM(input_size = 1, hidden_size=3, num_layers=1)
            cell3 = nn.LSTM(input_size = 1, hidden_size=3, num_layers=1)
            cell4 = nn.LSTM(input_size = 1, hidden_size=3, num_layers=1)
            cell5 = nn.LSTM(input_size = 1, hidden_size=3, num_layers=1)
            cells.append(cell1)
            cells.append(cell2)
            cells.append(cell3)
            cells.append(cell4)
            cells.append(cell5)
        self.cells = cells
        self.num_layers = 5*max_layers
        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)
            
    def forward(self, state):
        logits = []
        softmax = nn.Softmax(1)
        output, hidden_states = cells[0](state[0])
        output = output.reshape(1,3) # this is the logit
        logit = softmax(output)
        logits.append(logit)        
        for i,cell in enumerate(cells[1,-1]):
            output, hidden_states = cells[i](state[i],hidden_states[cell])
            output = output.reshape(1,3) # this is the logit
            logit = softmax(output)
            logits.append(logit)
        return logits

    def get_action(self, state): # state = sequence of length 5 times number of layers
        logits = forward(state)
        actions = [torch.argmax(logit) for logit in logits]      
        return actions,logits 
    
    # REINFORCE
    def update_policy(self, rewards, logits):
    discounted_rewards = []

    for t in range(len(rewards)):
        Gt = 0 
        pw = 0
        for r in rewards[t:]:
            Gt = Gt + GAMMA**pw * r
            pw = pw + 1
        discounted_rewards.append(Gt)
        
    discounted_rewards = torch.tensor(discounted_rewards)
    discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9) # normalize discounted rewards

    policy_gradient = []
    for logit, Gt in zip(logits, discounted_rewards):
        policy_gradient.append(-logit * Gt)
    
    self.optimizer.zero_grad()
    policy_gradient = torch.stack(policy_gradient).sum()
    policy_gradient.backward()
    self.optimizer.step()
        














