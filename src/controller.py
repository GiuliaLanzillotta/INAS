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
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, ChebConv  # noqa

#constants##
GAMMA = 1
FEATURES =

class controller(nn.Module):

    def __init__(self, num_features, layers, dimension=16):
        super(Net, self).__init__()
        self.conv1 = GCNConv(num_features, dimension, cached=True,
                             normalize=True)
        """normalize (bool, optional) – Whether to add self-loops and 
        apply symmetric normalization. (default: True)"""
        self.conv2 = GCNConv(num_features, dimension, cached=True,
                             normalize=True)
        # self.conv1 = ChebConv(data.num_features, 16, K=2)
        # self.conv2 = ChebConv(16, data.num_features, K=2)
        self.layers = layers
        self.optimizer = optim.Adam(self.parameters(), lr=5e-4)
        self.exploration = 0.90
        self.reg_params = self.conv1.parameters()
        self.non_reg_params = self.conv2.parameters()


    def build_graph_from_state(self, state):
        """Building a graph from the state ( list of numbers). """
        edge_index = []
        for l in range(self.layers):
            edge_index.append([5 * l + 1, 5 * l + 2])
            edge_index.append([5 * l + 1, 5 * l + 5])
            edge_index.append([5 * l + 3, 5 * l + 5])
            edge_index.append([5 * l + 3, 5 * l + 4])
            edge_index.append([5 * l + 2, 5 * l + 1])
            edge_index.append([5 * l + 5, 5 * l + 1])
            edge_index.append([5 * l + 5, 5 * l + 3])
            edge_index.append([5 * l + 4, 5 * l + 3])
            if (l != 0):
                edge_index.append([5 * l + 3, 5 * (l-1) + 3])
                edge_index.append([5 * l + 4, 5 * (l-1) + 4])
                edge_index.append([5 * l + 5, 5 * (l-1) + 5])

        edge_index = torch.tensor(edge_index, dtype=torch.long)
        x = torch.tensor(state, dtype=torch.float)

        data = Data(x=x, edge_index=edge_index.t().contiguous())
        return data

    def forward(self, state):
        """State is a sequence of numbers. It first has to be transalted into the
        graph and packaged into a data object. Then it can be forwarded."""
        data = self.build_graph_from_state(state)
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1)

    def exponential_decayed_epsilon(self, step):
        # Decay every decay_steps interval
        decay_steps = 2
        decay_rate = 0.9
        return self.exploration * decay_rate ** (step / decay_steps)

    def get_action(self, state, ep):  # state = sequence of length 5 times number of layers
        # if (np.random.random() < self.exponential_decayed_epsilon(ep)) and (ep > 0):

        logits = self.forward(state)
        exp=False
        if np.random.random() < self.exponential_decayed_epsilon(ep):
            exp = True
            actions = [torch.argmin(logit) for logit in logits]
            logits = [logit[0][torch.argmax(logit)] for logit in logits]
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
        














