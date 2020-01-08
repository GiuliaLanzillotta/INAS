"""
The train file has to coordinate the REINFORCE algorithm in the main function
"""
from src.cnn import cnn
from src.ontroller import controller
import torch
import torchvision
import torchvision.transforms as transforms
import pandas as pd

# This means that we're only looking at squared images for now.
image_size = 32
prev_channels = 3
num_classes = 10
layers_limit = 20
from time import time

LOAD_MODEL = False
CONTROLLER_PATH = "./controller.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def print_action(action, layers):
    for i in range(layers):
        try:
            print([a.item() for a in action[i*5:(i+1)*5]])
        except Exception as e:
            print([a for a in action[i * 5:(i + 1) * 5]])

def print_state(action, layers):
    for i in range(layers):
        print([a.item() for a in action[i*5:(i+1)*5]])

def save_rewards():
    return

def save_model(model,path,ep):
    optimizer = model.optimizer
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'episode':ep
    }, path)

def load_model(model, path):
    optimizer = model.optimizer
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    episode = checkpoint['episode']
    return model,episode

def train():
    #with tf.name_scope("train"):
    best_reward, best_architecture = 0, None
    num_episodes = 100
    num_steps = 10
    max_layers = 6
    data_loader = load_data()
    max_ep_reward = 0
    starting_episode = 0
    controller1 = controller(max_layers)

    if (LOAD_MODEL):
        print("Ripristinating the controller...")
        controller1, starting_episode = load_model(controller1, CONTROLLER_PATH)

    t1 = time()
    rewards_averaged_per_ep = []
    rewards_history = pd.DataFrame()
    states_history = pd.DataFrame()
    for ep in range(num_episodes):
        print("/////////// episode ", ep + starting_episode, " ///////////")
        cnn1 = cnn(max_layers, image_size, prev_channels, num_classes, num_steps)
        state = cnn1.state
        rewards = []
        logits = []

        for step in range(num_steps):
            action, logit = controller1.get_action(state, ep + starting_episode) # what state?
            new_state = cnn1.build_child_arch(action)
            states_history = states_history.append([new_state])
            states_history.to_csv("states.csv")
            reward, net = cnn1.get_reward(data_loader, ep + starting_episode, rewards_averaged_per_ep) #already have new_state updated
            state = new_state
            logits.append(logit)
            rewards.append(reward)
            print("Step",ep + starting_episode,":",step)
            print("Reward: ", reward)
            print("State: ", new_state)

        rewards_history = rewards_history.append([rewards])
        controller1.update_policy(rewards, logits)
        rewards_averaged_per_ep.append(sum(rewards)/len(rewards))
        rewards_history.to_csv("rewards.csv")

        rewards = [x ** 3.0 for x in rewards]
        controller1.update_policy(rewards, logits)
        episode_reward = rewards[0]

        save_model(controller1, CONTROLLER_PATH, ep + starting_episode)

        # if ep % 3 == 1:
        #     if (episode_reward < max_ep_reward):
        #         controller1.add_layer()
        #         print("Adding one layer")
        #         max_layers = max_layers+1
        #     else:
        #         max_ep_reward = episode_reward

        if (ep % 3 == 1) and (max_layers <= layers_limit):
            controller1.add_layer()
            max_layers = max_layers + 1
        t2 = time()
        print("Elapsed time: ", t2-t1)

def load_data(batch_size = 32):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                              shuffle=True, num_workers=0)
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                             shuffle=False, num_workers=0)
    return trainloader, testloader

if __name__ == '__main__':
  train()

