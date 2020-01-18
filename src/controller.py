""" The controller module will do what the old REINFORCE file did.
    It has to implement the following methods:
    - get_action(state)
    - update_policy(episode)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from torch.autograd import Variable

#constants##
GAMMA = 1

class controller(nn.Module):

    def __init__(self, max_layers):
        super(controller,self).__init__()
        
        cells = []
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
        cells=[cells*max_layers][0]

        self.cells = nn.ModuleList(cells) # better name: layers
        self.num_layers = 5*max_layers
        self.optimizer = optim.Adam(self.parameters(), lr=1e-4)
        self.exploration = 0.90
            
    def forward(self, state):
        logits = []
        softmax = nn.Softmax(1)
        for i,cell in enumerate(self.cells):
            # state_i = torch.tensor(state[i], dtype=torch.float).view(1,1,1)
            if(i==0):
                output, hidden_states = cell(torch.tensor(state[i], dtype=torch.float).view(1,1,1))
                for element in hidden_states:
                    element.clone().detach().requires_grad_(True)
            else:
                output, hidden_states = cell(torch.tensor(state[i], dtype=torch.float).view(1,1,1).clone().detach().requires_grad_(True), hidden_states)
                for element in hidden_states:
                    element.clone().detach().requires_grad_(True)
            output = output.reshape(1,3) # this is the logit
            logit = softmax(output)
            logits.append(logit)
        return logits
    
    def add_layer(self):
        self.cells = self.cells.append(self.cells[0])
        self.cells = self.cells.append(self.cells[1])
        self.cells = self.cells.append(self.cells[2])
        self.cells = self.cells.append(self.cells[3])
        self.cells = self.cells.append(self.cells[4])

    "Exploration Calculation"
    def exponential_decayed_epsilon(self, step):
        # Decay every decay_steps interval
        decay_steps = 2
        decay_rate = 0.9
        return self.exploration * decay_rate ** (step / decay_steps)

    "Get an action to build the new state of the CNN from the last one!"
    def get_action(self, state, ep):  # state = sequence of length 5 times number of layers

        logits = self.forward(state)
        exp = False
        # if np.random.random() < self.exponential_decayed_epsilon(ep):
        #     exp = True
        #     actions = [torch.argmin(logit) for logit in logits]
        #     logits = [logit[0][torch.argmin(logit)] for logit in logits]
        # else:
        #     actions = [torch.argmax(logit) for logit in logits]
        #     logits = [logit[0][torch.argmax(logit)] for logit in logits]
        # return actions, logits, exp

        actions = [torch.argmax(logit) for logit in logits]
        logits = [logit[0][torch.argmax(logit)] for logit in logits]
        return actions, logits, exp

    "The Reinforce Algorithm"
    def update_policy(self, rewards, logits):
        discounted_rewards = []

        for t in range(len(rewards)):
            Gt = 0
            pw = 0
            for r in rewards[t:]:
                r = r ** (np.sign(r) * 3)
                # r = np.tan(r * np.pi / 2)
                Gt = Gt + GAMMA ** pw * r
                pw = pw + 1
            discounted_rewards.append(Gt)

        discounted_rewards = torch.tensor(discounted_rewards)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (
                    discounted_rewards.std() + 1e-4)  # normalize discounted rewards

        policy_gradient = []
        for logit, Gt in zip(logits, discounted_rewards):
            for element in logit:
                policy_gradient.append(-1.0 * torch.log(element) * Gt)

        self.optimizer.zero_grad()
        policy_gradient = torch.stack(policy_gradient).sum()  # * (1 / len(logits))
        policy_gradient.backward()
        self.optimizer.step()       #Weights of the RNN are updated via the reinforce algorithm!












