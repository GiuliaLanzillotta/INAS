""" The controller module will do what the old REINFORCE file did.
    It has to implement the following methods:
    - get_action(state)
    - update_policy(episode)
"""
import torch
import torch.nn as nn
import torch.optim as optim
import math
import numpy as np
import random
from torch.autograd import Variable

#constants##
GAMMA = 1

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
        self.cells = nn.ModuleList(cells) # better name: layers
        self.num_layers = 5*max_layers
        self.optimizer = optim.Adam(self.parameters(), lr=5e-4)
        self.exploration = 0.90

    def exponential_decayed_epsilon(self, step):
        # Decay every decay_steps interval
        decay_steps = 2
        decay_rate = 0.9
        return self.exploration * decay_rate ** (step / decay_steps)

    def forward(self, state):
        logits = []
        softmax = nn.Softmax(1)

        for i, cell in enumerate(self.cells):
            # state_i = torch.tensor(state[i], dtype=torch.float).view(1,1,1)
            if (i == 0):
                output, hidden_states = cell(torch.tensor(state[i], dtype=torch.float).view(1, 1, 1))
                for element in hidden_states:
                    element.requires_grad_(True)
            else:
                output, hidden_states = cell(
                    torch.tensor(state[i], dtype=torch.float).view(1, 1, 1).clone().detach().requires_grad_(True),
                    hidden_states)
                for element in hidden_states:
                    element.requires_grad_(True)
            output = output.reshape(1, 3)  # this is the logit
            logit = softmax(output)
            logits.append(logit)
        return logits

    def get_action(self, state, ep):  # state = sequence of length 5 times number of layers
        # if (np.random.random() < self.exponential_decayed_epsilon(ep)) and (ep > 0):

        logits = self.forward(state)
        exp=False
        if np.random.random() < self.exponential_decayed_epsilon(ep):
            exp = True
            actions = [torch.argmin(logit) for logit in logits]
            logits = [logit[0][torch.argmin(logit)] for logit in logits]
        else:
            actions = [torch.argmax(logit) for logit in logits]
            logits = [logit[0][torch.argmax(logit)] for logit in logits]
        return actions, logits, exp
    
    # REINFORCE
    def update_policy(self, rewards, logits):
        discounted_rewards = []

        for t in range(len(rewards)):
            Gt = 0 
            pw = 0
            for r in rewards[t:]:
                r = r**3
                Gt = Gt + GAMMA**pw * r
                pw = pw + 1
            discounted_rewards.append(Gt)
            
        discounted_rewards = torch.tensor(discounted_rewards)
        #discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-4) # normalize discounted rewards

        policy_gradient = []
        # logits = torch.tensor(logits)
        # logits = logits.flatten(1,-1)
        # logits is a list of lists where the outer contains all steps taken, the inner for a given step length  has 10 elements where each element is a tensor of length 3
        for logit, Gt in zip(logits, discounted_rewards):
            for element in logit:
                policy_gradient.append(-1.0 * torch.log(element) * Gt)
                # for index in range(3):
                #     policy_gradient.append(-1.0 * torch.log(element[0, index].type(torch.float)) * Gt)
            # policy_gradient.append(-1*torch.tensor(logit) * torch.tensor(Gt))
        
        self.optimizer.zero_grad()
        policy_gradient = torch.stack(policy_gradient).sum() * (1 / len(logits))
        policy_gradient.backward()
        self.optimizer.step()
        














