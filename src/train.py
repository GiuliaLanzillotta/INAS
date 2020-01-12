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
from time import time

# This means that we're only looking at squared images for now.
image_size = 32
prev_channels = 3
num_classes = 10


if torch.cuda.device_count() > 0:
    print("Let's use GPU computing")
else:
    print('No GPU available')

LOAD_MODEL = False
CONTROLLER_PATH = "./controller.pt"

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

def save_model(model,path):
    optimizer = model.optimizer
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)

def load_model(model, path):
    optimizer = model.optimizer
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return model

def train():
    #with tf.name_scope("train"):
    num_episodes = 15
    num_steps = 15
    max_layers = 15
    train = True

    data_loader = load_data_CIFAR()
    controller1 = controller(max_layers)
    t1 = time()
    save_time = datetime.datetime.now()

    rewards_history = pd.DataFrame()
    states_history = pd.DataFrame()
    exploration_history = pd.DataFrame()

    best_reward = 0
    best_state = []
    best_action = []
    for ep in range(num_episodes):
        print("-----------------------------------------------")
        print("Episode ", ep)
        cnn1 = cnn(max_layers, image_size, prev_channels, num_classes, train = True)
        state = cnn1.state
        rewards = []
        logits = []
        exps = []
        for step in range(num_steps):
            action, logit, exploration = controller1.get_action(state, ep, train) # what state?
            #print("Action: ")
            #print_action(action, max_layers)
            new_state = cnn1.build_child_arch(action)
            print("New state: ", new_state)
            reward = cnn1.get_reward(data_loader) #already have new_state updated

            if reward > best_reward:
                best_reward = reward
                best_state = state
                best_action = action
                print("Best Architecture Updated")
                print("Best State:", state)

            state = new_state
            logits.append(logit)
            rewards.append(reward)
            exps.append(exploration)
            states_history = states_history.append([new_state])
            states_history.to_csv("states_{}.csv".format(t1))
            print("****************")
            print("Step",ep,":",step)
            print("Reward: ", reward)
            print("****************")

        exploration_history = exploration_history.append(exps)
        rewards_history = rewards_history.append(rewards)
        controller1.update_policy(rewards, logits)
        t2 = time()
        rewards_history.to_csv("rewards_{}.csv".format(t1))
        rewards_history.to_csv("rewards_{}.csv".format(t1))
        exploration_history.to_csv("Exploration_{}.csv".format(t1))
        print("Elapsed time: ", t2-t1)

    print("Saving the controller...")
    save_model(controller1, CONTROLLER_PATH)

    # "The Best Architecture sampled"
    # print("Training NAS finished")
    # best_cnn = cnn(max_layers, image_size, prev_channels, num_classes,epochs=100)
    # useless_state = best_cnn.build_child_arch(best_action)
    # reward = best_cnn.get_reward(data_loader)
    # print("The Final CIFAR-10 Accuracy is {}".format(reward*100))
    # print("The Final Architecture is {}".format(best_state))




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

