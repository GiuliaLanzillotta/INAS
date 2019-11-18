"""
The train file has to coordinate the REINFORCE algorithm in the main function
"""
import numpy as np
import tensorflow as tf
import cnn
import controller
from child_manager import get_reward
import torch
from torch import nn
import torchvision

# This means that we're only looking at squared images for now.
image_size = 28
prev_channels = 1
num_classes = 10

def train():
    #with tf.name_scope("train"):
    num_episodes = 100
    num_steps = 10
    max_layers = 2
    data_loader = load_data()
    controller1 = controller(max_layers)
    for ep in num_episodes:
        cnn1 = cnn(max_layers, image_size, prev_channels, num_classes)
        initial_state = cnn1.state
        rewards = []
        logits = []
        for step in num_steps:
            action, logit = controller1.get_action(initial_state) # what state?
            new_state = cnn1.build_child_arc(action, initial_state)
            reward = cnn1.get_reward(data_loader) #already have new_state updated
            logits.append(logit)
            rewards.append(reward)
        controller.update_policy(rewards,logits)    
        
    return new_state

def load_data(batch_size = 16):
    #TODO
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=0)
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=False, num_workers=0)
    return trainloader, testloader
if __name__ == '__main__':
  train()

