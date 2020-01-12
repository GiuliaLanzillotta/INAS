"""
The test files tests the final architecture
"""
from src.cnn import cnn
from src.controller import controller
import torch
import torchvision
import torchvision.transforms as transforms
import pandas as pd
import datetime
from time import time

image_size = 32
prev_channels = 3
num_classes = 10

LOAD_MODEL = True
CONTROLLER_PATH = "./controller.pt"

def load_model(model, path):
    optimizer = model.optimizer
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    episode = checkpoint['episode']
    return model,episode

def test():
    num_episodes = 1
    max_layers = 15
    controller1 = controller(max_layers)
    test = True

    if (LOAD_MODEL):
        print("Ripristinating the controller...")
        controller1, starting_episode = load_model(controller1, CONTROLLER_PATH)

    data_loader = load_data_CIFAR()
    best_cnn = cnn(max_layers, image_size, prev_channels, num_classes,train = False, epochs=100)
    state = best_cnn.state
    action, logit = controller1.get_action(state, 1, test)
    new_state = best_cnn.build_child_arch(action)
    print("New state: ", new_state)
    reward = best_cnn.get_reward(data_loader)
    print("****************")
    print("Reward: ", reward)
    print("****************")


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
  test()
