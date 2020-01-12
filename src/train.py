"""
The train file has to coordinate the REINFORCE algorithm in the main function
"""
from src.cnn import cnn
from src.controller import controller
import torch
import torchvision
import torchvision.transforms as transforms
import pandas as pd
import datetime

# This means that we're only looking at squared images for now.
image_size = 32
prev_channels = 3
num_classes = 10
from time import time

if torch.cuda.device_count() > 0:
    print("Let's use GPU computing")
else:
    print('No GPU available')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# def print_action(action, layers):
#     for i in range(layers):
#         try:
#             print([a.item() for a in action[i*5:(i+1)*5]])
#         except Exception as e:
#             print([a for a in action[i * 5:(i + 1) * 5]])

def print_state(action, layers):
    for i in range(layers):
        print([a.item() for a in action[i*5:(i+1)*5]])

def train():
    #with tf.name_scope("train"):
    num_episodes = 10
    num_steps = 10
    max_layers = 15

    data_loader = load_data_CIFAR()
    controller1 = controller(max_layers)
    t1 = time()
    save_time = datetime.datetime.now()

    rewards_history = pd.DataFrame()
    states_history = pd.DataFrame()
    exploration_history = pd.DataFrame()
    for ep in range(num_episodes):
        print("-----------------------------------------------")
        print("Episode ", ep)
        cnn1 = cnn(max_layers, image_size, prev_channels, num_classes)
        state = cnn1.state
        rewards = []
        logits = []
        exps = []
        reward = 0
        for step in range(num_steps):
            action, logit, exploration = controller1.get_action(state, ep) # what state?
            #print("Action: ")
            #print_action(action, max_layers)
            new_state = cnn1.build_child_arch(action)
            print("New state: ", new_state)
            ## Trying to set the reward to the delta
            ## in this way we encourage the controller to
            ## always improve
            ##reward = cnn1.get_reward(data_loader) - reward #already have new_state updated
            reward = cnn1.get_reward(data_loader)
            state = new_state
            logits.append(logit)
            rewards.append(reward)
            exps.append(exploration)
            states_history = states_history.append([new_state])
            states_history.to_csv("Attention_states.csv")
            print("****************")
            print("Step",ep,":",step)
            print("Reward: ", reward)
            print("****************")

        exploration_history = exploration_history.append(exps)
        rewards_history = rewards_history.append(rewards)
        controller1.update_policy(rewards, logits)
        t2 = time()
        rewards_history.to_csv("Attention_rewards.csv")
        exploration_history.to_csv("Attention_Exploration.csv")
        print("Elapsed time: ", t2-t1)

def load_data_CIFAR(batch_size = 4):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                              shuffle=True, num_workers=0)
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                             shuffle=False, num_workers=0)
    return trainloader, testloader


def load_data_MNIST(batch_size=16):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.1307,), (0.3081,))])
    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=0)

    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=False, num_workers=0)
    return trainloader, testloader

if __name__ == '__main__':
  train()

