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

# This constant controls the discount on the rewards
# over time
GAMMA = 1
class Encoder(nn.Module):
    """Plain and simple Encoder network to put before the
    Encoder or the Attention module."""
    def __init__(self, input_dim, hid_dim, dropout):
        super().__init__()
        self.hid_dim = hid_dim
        self.rnn = nn.GRU(input_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, state):
        # state = [len, batch size]
        outputs, hidden = self.rnn(state)  # no cell state!
        # outputs = [src len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # outputs are always from the top hidden layer
        return hidden

class Decoder(nn.Module):
    """Plain and simple Decoder network to stack on top of the Encoder,
    alternatively to the Attention head."""
    def __init__(self, output_dim, hid_dim, dropout):
        super().__init__()
        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.rnn = nn.GRU(1 + hid_dim, hid_dim)
        self.fc_out = nn.Linear(1 + hid_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, context):
        # Recap on sizes
        # input = [batch size]
        # hidden = [n layers * n directions, batch size, hid dim]
        # context = [n layers * n directions, batch size, hid dim]
        # n layers and n directions in the decoder will both always be 1, therefore:
        # hidden = [1, batch size, hid dim]
        # context = [1, batch size, hid dim]
        # input = [1, batch size=1]
        input = torch.tensor(input, dtype=torch.float).view(1,1,1)
        input_con = torch.cat((input, context), dim=2)
        output, hidden = self.rnn(input_con, hidden)
        # output = [seq len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # seq len, n layers and n directions will always be 1 in the decoder, therefore:
        # output = [1, batch size, hid dim]
        # hidden = [1, batch size, hid dim]
        output = torch.cat((input.squeeze(0), hidden.squeeze(0), context.squeeze(0)),
                           dim=1)
        # output = [batch size, hid dim * 2]
        prediction = self.fc_out(output)
        # prediction = [batch size, output dim]
        return prediction, hidden

class AttentionDecoder(nn.Module):
    """Attention model inspired to Bahdanau et. al attention model.
    """
    def __init__(self, output_dim, hid_dim, dropout, max_length):
        super().__init__()
        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.max_length = max_length
        self.key_layer = nn.Linear(self.hid_dim , self.hid_dim, bias=False)
        self.query_layer = nn.Linear(self.hid_dim , self.hid_dim, bias=False)
        self.energy_layer = nn.Linear(self.hid_dim , 1, bias=False)
        self.rnn = nn.GRU(1 + hid_dim, hid_dim, 1,
                          batch_first=True)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_hidden):
        """We apply an MLP with tanh-activation to both the current
        decoder state si (the query) and each encoder state hj (the key),
        and then project this to a single value (i.e. a scalar)
        to get the attention energy eij"""
        query = hidden[-1]
        value = encoder_hidden
        # compute context vector using attention mechanism
        query = self.query_layer(query)
        proj_key = self.key_layer(encoder_hidden)
        scores = self.energy_layer(torch.tanh(query + proj_key))
        scores = scores.squeeze(2).unsqueeze(1)
        alphas = torch.softmax(scores, dim=-1)
        context = torch.bmm(alphas, value)
        # now that the input weigths are calculated we can pass through
        # the decoder and obtain the logits we need
        rnn_input = torch.cat([torch.tensor(input, dtype=torch.float).view(1,1,1), context], dim=2)
        rnn_output, hidden = self.rnn(rnn_input, hidden.view(1,1,-1))
        pre_output = self.dropout(rnn_output)
        prediction = self.fc_out(pre_output)
        return prediction, hidden

class controller(nn.Module):

    def __init__(self, max_layers):
        super(controller,self).__init__()

        self.num_layers = 5*max_layers
        self.encoder = Encoder(1,15,0.2)
        # Un-comment the following line to use a standard
        # decoder as head
        # self.decoder = Decoder(3,15,0.2)
        self.decoder = AttentionDecoder(3,15,0.2, self.num_layers)
        self.optimizer = optim.Adam(self.parameters(), lr=5e-4)

    def forward(self, state):
        # tensor to store decoder outputs
        logits = torch.zeros(self.num_layers, 1, 3)
        # last hidden state of the encoder is the context
        context = self.encoder(torch.tensor(state, dtype=torch.float).view(self.num_layers,1,1))
        # context also used as the initial hidden state of the decoder
        hidden = context[-1]

        for t in range(1, self.num_layers):
            input = state[t]
            # receive output tensor (predictions) and new hidden state
            output, hidden = self.decoder(input, hidden, context)
            # place predictions in a tensor holding predictions for each token
            logits[t] = output
        return logits

    def get_action(self, state):  # state = sequence of length 5 times number of layers
        logits = self.forward(state)
        actions = [torch.argmax(logit) for logit in logits]
        logits = [logit[0][torch.argmax(logit)] for logit in logits]
        return actions, logits


    def update_policy(self, rewards, logits):
        """This function implements the policy gradient update
        described in the REINFORCE algorithm"""

        # We first calculate the sum of the discounted rewards
        discounted_rewards = []
        for t in range(len(rewards)):
            Gt = 0 
            pw = 0
            for r in rewards[t:]:
                r = r**(np.sign(r)*3)
                # Un-comment the following line to use a tangent
                # reward function
                #r = np.tan(r * np.pi / 2)
                Gt = Gt + GAMMA**pw * r
                pw = pw + 1
            discounted_rewards.append(Gt)
            
        discounted_rewards = torch.tensor(discounted_rewards)
        # We normalise the sum of discounted rewards
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) \
                             / (discounted_rewards.std() + 1e-4)

        # Calculate the policy gradient
        policy_gradient = []
        # logits is a list of lists where the outer contains
        # all steps taken, the inner for a given step length
        # has 10 elements where each element is a tensor of length 3
        for logit, Gt in zip(logits, discounted_rewards):
            for element in logit:
                policy_gradient.append(-1.0 * torch.log(element) * Gt)

        self.optimizer.zero_grad()
        policy_gradient = torch.stack(policy_gradient).sum() #* (1 / len(logits))
        policy_gradient.backward()
        self.optimizer.step()
        














