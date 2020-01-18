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

"Define some image related parameters"
image_size = 32
prev_channels = 3
num_classes = 10

"Use GPU computing if available"
if torch.cuda.device_count() > 0:
    print("Let's use GPU computing")
else:
    print('No GPU available')

if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LOAD_MODEL = True
SAVE_CONTROLLER_PATH = "./controller_stacked.pt"
LOAD_CONTROLLER_PATH = "./controller1.pt"

"save_model saves the weights and the optimizer to enable more training"
def save_model(model,path):

    optimizer = model.optimizer
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)

"load_model from a specific point"
def load_model(model, path):
    optimizer = model.optimizer
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return model

"Main Training Loop"
def train():

    "Define training parameters"
    num_episodes = 5
    num_steps = 10
    max_layers = 15
    train = True

    "Load Data and Controller"
    data_loader = load_data_CIFAR()
    controller1 = controller(max_layers)
    t1 = time()
    save_time = datetime.datetime.now()
    if (LOAD_MODEL):
        print("Loading the Controller")
        controller1 = load_model(controller1, path=LOAD_CONTROLLER_PATH)

    "Save Files Defined"
    rewards_history = pd.DataFrame()
    states_history = pd.DataFrame()
    exploration_history = pd.DataFrame()

    "Best Sampled Architecture"
    best_reward = 0
    best_state = []
    for ep in range(num_episodes):
        print("-----------------------------------------------")
        print("Episode ", ep)

        """Initialise the CNN architcture and store the initial state"""
        cnn1 = cnn(max_layers, image_size, prev_channels, num_classes, train = True)
        state = cnn1.state

        "Define Rewards, Logits and Exploration(if active)"
        rewards = []
        logits = []
        exps = []

        for step in range(num_steps):
            """1. Get the action from a current state: this corresponds to  
             a forward pass through the controller, which returns the action
             and take the corresponding logit"""
            action, logit, exploration = controller1.get_action(state, ep, train)

            """2. Update the state with the new action and translate 
            the new state into an actual CNN architecture. Note: the architecture 
            is never exchanged between modules. Only the state is communicated."""
            new_state = cnn1.build_child_arch(action)

            print("New state: ", new_state)

            """3. Train the new child architecture and get a reward corresponding
            "to the test accuracy"""
            reward = cnn1.get_reward(data_loader)

            "Store the best generated architecture"
            if reward > best_reward:
                best_reward = reward
                best_state = state
                print("Best Architecture Updated")
                print("Best State:", state)

            "Store variables"
            state = new_state
            logits.append(logit)
            rewards.append(reward)
            exps.append(exploration)
            states_history = states_history.append([new_state])
            states_history.to_csv("states_{}.csv".format(save_time))    # Save_time only works on linux, remove for windows and rename save file.
            print("****************")                                   # If you get OS ERROR 22, change this variable to t1, or anything else!
            print("Step", ep,":", step)
            print("Reward: ", reward)
            print("****************")

        exploration_history = exploration_history.append(exps)
        rewards_history = rewards_history.append(rewards)
        rewards_history.to_csv("rewards_{}.csv".format(save_time))     # Save_time - Linux, t1 - Windows
        exploration_history.to_csv("Exploration_{}.csv".format(save_time))  #Save_time - Linux, t1 - Windows

        """At the end of each episode the policy gradient is back-propagated
        though the controller to update its parameters"""
        controller1.update_policy(rewards, logits)
        t2 = time()
        print("Elapsed time: ", t2-t1)

    print("Saving the controller...")
    save_model(controller1, SAVE_CONTROLLER_PATH)
    print("Training NAS finished")

    "The Best Architecture sampled training"
    print("Now training the best Architecture sampled")
    best_cnn = cnn(max_layers, image_size, prev_channels, num_classes, train=False, epochs=100)
    best_cnn.bestCNN(best_state)
    reward = best_cnn.get_reward(data_loader)
    print("The Final CIFAR-10 Accuracy is {}".format(reward*100))
    print("The Final Architecture is {}".format(best_state))

"Load Dataset Functions"
def load_data_CIFAR(batch_size = 4):
    "Normalisation values taken from PyTorch Website"
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                              shuffle=True, num_workers=0)
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                             shuffle=True, num_workers=0)
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

