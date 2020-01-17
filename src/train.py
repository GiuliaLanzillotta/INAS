"""
The train file has to coordinate the REINFORCE algorithm in the main function
"""
from src.cnn import cnn
from src.controller import controller
import torch
import torchvision
import torchvision.transforms as transforms
import pandas as pd
from time import time


## DATASETS MANAGEMENT
# Data sets parameters
image_size = 32
prev_channels = 3
num_classes = 10
def load_data_CIFAR(batch_size=64):
    # Applying normalisation to images
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=0)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=0)
    return trainloader, testloader
def load_data_MNIST(batch_size=4):
    # Applying normalisation to data set images
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.1307,), (0.3081,))])
    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                          download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=0)

    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                         download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=0)
    return trainloader, testloader

## UTILS
# parameters
LOAD_MODEL = False
CONTROLLER_NAME = "Attention"
num_episodes = 15
num_steps = 5
max_layers = 15
starting_episode = 0

## Helper functions
def print_state(state, layers):
    for i in range(layers):
        print([s.item() for s in state[i*5:(i+1)*5]])

def save_model(model,ep):
    optimizer = model.optimizer
    path = './'+CONTROLLER_NAME+'.pt'
    # Saving the episode so that the training can
    # continue from where it has left
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'episode':ep
    }, path)

def load_model(model):
    optimizer = model.optimizer
    path = './'+CONTROLLER_NAME+'.pt'
    checkpoint = torch.load(path)
    # Retrieving the episode so that the training can
    # continue from where it has left
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    episode = checkpoint['episode']
    return model,episode


if torch.cuda.device_count() > 0:
    print("Let's use GPU computing")
else:
    print('No GPU available')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train():
    # INITIALISATION
    data_loader = load_data_CIFAR()
    # The controller is created only once, at the
    # beginning of the training
    controller = controller(max_layers)
    if (LOAD_MODEL):
        print("Ripristinating the controller...")
        controller, starting_episode = load_model(controller)
    rewards_history = pd.DataFrame()
    states_history = pd.DataFrame()
    t1 = time()
    # Inserting the main loop in a try catch so that
    # the model can be saved even in case of failures
    # In this way, we avoid re-doing computation
    try:
        for ep in range(num_episodes):
            print("-----------------------------------------------")
            print("Episode number ", ep)
            # A new CNN is created at the beginning of each episode
            # The initial configuration is summarised in the state
            # variable of CNN
            # During the episode this class will keep track of the
            # state and will build the child architectures
            cnn = cnn(max_layers, image_size, prev_channels, num_classes)
            state = cnn.state
            # Initialise structures to collect rewards and logits
            rewards = []
            logits = []
            # Un-comment this to use delta rewards
            # rewards_diffs = []
            # max_reward = 0
            for step in range(num_steps):
                # 1. Get the action from a current state: this correspond to
                # a forward pass through the controller, which returns the action
                # take and the corresponding logit
                action, logit, = controller.get_action(state)
                # 2. Update the state with the new action and translate
                # the new state into an actual CNN archtiecture. Note: the
                # architecture is never exchanged between modules. Only the
                # state is communicated.
                new_state = cnn.build_child_arch(action)
                state = new_state
                print("New state: ", new_state)
                # 3. Train the new child architecture just built and
                # get a reward corresponding to the test accuracy
                reward = cnn.get_reward(data_loader)
                logits.append(logit)
                rewards.append(reward)
                # Uncomment this to use delta rewards
                # reward_diff = reward - max_reward
                # if(reward>max_reward):
                #     max_reward = reward
                # rewards_diffs.append(reward_diff)
                states_history = states_history.append([new_state])
                states_history.to_csv("5Att_states.csv")
                print("Step ",step, " of episode ",ep)
                print("Reward: ", reward)
                print("******************************************")

            rewards_history = rewards_history.append(rewards)
            rewards_history.to_csv("5Att_rewards.csv")
            # At the end of each episode the policy gradient is
            # back-propagated through the controller to update
            # its parameters
            controller.update_policy(rewards, logits)
            t2 = time()
            print("Elapsed time: ", t2-t1)

    except Exception as e:
        print(e)
    print("Saving the controller...")
    save_model(controller, ep + starting_episode)


if __name__ == '__main__':
  train()

