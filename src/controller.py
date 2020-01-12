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
class Encoder(nn.Module):
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
    def __init__(self, output_dim, hid_dim, dropout):
        super().__init__()
        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.rnn = nn.GRU(1 + hid_dim, hid_dim)
        self.fc_out = nn.Linear(1 + hid_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, context):
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
    """Bahdanau et. al attention model.
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
        # now that the weigths are calculated we can pass through
        # the decoder
        rnn_input = torch.cat([torch.tensor(input, dtype=torch.float).view(1,1,1), context], dim=2)
        rnn_output, hidden = self.rnn(rnn_input, hidden.view(1,1,-1))
        pre_output = self.dropout(rnn_output)
        prediction = self.fc_out(pre_output)
        return prediction, hidden

class controller(nn.Module):

    def __init__(self, max_layers):
        super(controller,self).__init__()

        self.num_layers = 5*max_layers
        self.encoder = Encoder(1,10,0.2)
        self.decoder = AttentionDecoder(3,10,0.2, self.num_layers)
        self.optimizer = optim.Adam(self.parameters(), lr=5e-4)
        self.exploration = 0.90

    def exponential_decayed_epsilon(self, step):
        # Decay every decay_steps interval
        decay_steps = 2
        decay_rate = 0.9
        return self.exploration * decay_rate ** (step / decay_steps)

    def forward(self, state):
        # tensor to store decoder outputs
        logits = torch.zeros(self.num_layers, 1, 3)
        # last hidden state of the encoder is the context
        context = self.encoder(torch.tensor(state, dtype=torch.float).view(self.num_layers,1,1))
        # context also used as the initial hidden state of the decoder
        hidden = context[-1]

        for t in range(1, self.num_layers):
            #input = state[t]
            input = state[t]
            # receive output tensor (predictions) and new hidden state
            output, hidden = self.decoder(input, hidden, context)
            # place predictions in a tensor holding predictions for each token
            logits[t] = output

            # # get the highest predicted token from our predictions
            # top1 = output.argmax(1)
            #
            # # if teacher forcing, use actual next token as next input
            # # if not, use predicted token
            # input = trg[t] if teacher_force else top1

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
        














