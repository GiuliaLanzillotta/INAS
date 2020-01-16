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
layers_limit = 15
from time import time

if torch.cuda.device_count() > 0:
    print("Let's use GPU computing")
else:
    print('No GPU available')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


LOAD_MODEL = False
CONTROLLER_NAME = "RecurrentController"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def print_state(state, layers):
    for i in range(layers):
        print([s.item() for s in state[i*5:(i+1)*5]])

def save_model(model,ep):
    path = './'+CONTROLLER_NAME+'.pt'
    optimizer = model.optimizer
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'episode':ep
    }, path)

def load_model(model):
    path = './'+CONTROLLER_NAME+'.pt'
    optimizer = model.optimizer
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    episode = checkpoint['episode']
    return model,episode

def train():

    num_episodes = 10
    num_steps = 5
    max_layers = 5

    data_loader = load_data_CIFAR()
    controller1 = controller(max_layers)
    t1 = time()

    starting_episode = 1

    if (LOAD_MODEL):
        print("Ripristinating the controller...")
        controller1, starting_episode = load_model(controller1)

    rewards_history = pd.DataFrame()
    states_history = pd.DataFrame()
    exploration_history = pd.DataFrame()
    try:
        for ep in range(num_episodes):
            print("-----------------------------------------------")
            print("Episode ", ep)
            cnn1 = cnn(max_layers, image_size, prev_channels, num_classes)
            state = cnn1.state
            rewards = []
            rewards_diffs = []
            logits = []
            exps = []
            max_reward = 0
            increase = False
            controller1 = controller(max_layers)
            t1 = time()
            for step in range(num_steps):
                action, logit, exploration = controller1.get_action(state, ep)

                new_state = cnn1.build_child_arch(action)
                print("New state: ", new_state)

                reward = cnn1.get_reward(data_loader)
                if reward > max_reward:
                    max_reward = reward
                else:
                    increase = True
                # reward_diff = reward - max_reward
                # if(reward>max_reward):
                #     max_reward = reward
                state = new_state
                logits.append(logit)
                # rewards_diffs.append(reward_diff)
                rewards.append(reward)
                exps.append(exploration)
                states_history = states_history.append([new_state])
                states_history.to_csv("Recurrent_states.csv")
                print("****************")
                print("Step", ep, ":", step)
                print("Reward: ", reward)
                print("****************")

            exploration_history = exploration_history.append(exps)
            rewards_history = rewards_history.append(rewards)
            controller1.update_policy(rewards, logits)
            if ep % 3 == 1:
                if (increase):
                    controller1.add_layer()
                    max_layers = min(layers_limit, max_layers + 1)
                    print("Adding one layer : ", max_layers)
            t2 = time()
            rewards_history.to_csv("Recurrent_rewards.csv")
            exploration_history.to_csv("Recurrent_Exploration.csv")
            print("Elapsed time: ", t2 - t1)

    except Exception as e:
        print(e)

    print("Saving the controller...")
    save_model(controller1, ep + starting_episode)

def load_data_CIFAR(batch_size=4):
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

