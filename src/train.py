"""
The train file has to coordinate the REINFORCE algorithm in the main function
"""
from cnn import cnn
from controller import controller
import torch
import torchvision
import torchvision.transforms as transforms

# This means that we're only looking at squared images for now.
image_size = 32
prev_channels = 3
num_classes = 10
layers_limit = 15
from time import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train():
    #with tf.name_scope("train"):
    num_episodes = 100
    num_steps = 2
    max_layers = 3
    data_loader = load_data()
    max_ep_reward = 0
    controller1 = controller(max_layers)
    t1 = time()
    rewards_averaged_per_ep = []
    for ep in range(num_episodes):
        print("episode ", ep)
        cnn1 = cnn(max_layers, image_size, prev_channels, num_classes, num_steps)
        state = cnn1.state
        rewards = []
        logits = []

        for step in range(num_steps):
            action, logit = controller1.get_action(state, ep) # what state?
            new_state = cnn1.build_child_arch(action)
            reward = cnn1.get_reward(data_loader, ep, rewards_averaged_per_ep) #already have new_state updated
            state = new_state
            logits.append(logit)
            rewards.append(reward)
            print("Step",ep,":",step)
            print("Reward: ", reward)
            print("State: ", new_state)

        rewards_averaged_per_ep.append(sum(rewards)/len(rewards))
        controller1.update_policy(rewards, logits)
        episode_reward = rewards[0]
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

def load_data(batch_size = 16):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(root='../data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=0)
    
    testset = torchvision.datasets.CIFAR10(root='../data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=False, num_workers=0)
    return trainloader, testloader

if __name__ == '__main__':
  train()

