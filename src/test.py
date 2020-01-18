from src.cnn import cnn
from src.controller import controller
from src.conv_net import conv_net
import torch
import torchvision
import torchvision.transforms as transforms
import torch
from torch import nn
import torch.optim as optim
from src.conv_net import conv_net
import argparse
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
CONTROLLER_NAME = "AttentionController"

def load_model(model):
    optimizer = model.optimizer
    path = './'+CONTROLLER_NAME+'.pt'
    checkpoint = torch.load(path)
    # Retrieving the episode so that the training can
    # continue from where it has left
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return model

def load_data_CIFAR(batch_size=64):
    # Applying normalisation to images
    image_size = 32
    prev_channels = 3
    num_classes = 10

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
    return trainloader, testloader, image_size, prev_channels, num_classes

def load_data_MNIST(batch_size=4):
    # Applying normalisation to data set images

    image_size = 28
    prev_channels = 1
    num_classes = 10

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
    return trainloader, testloader, image_size, prev_channels, num_classes

def train_CNN(net, data_loader):

    net = net.to(device)
    data_loader_train, data_loader_test = data_loader
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.005,
                          weight_decay=0.0005,
                          momentum=0.9,
                          nesterov=True)
    schedular = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)
    print('Started Training')
    for epoch in range(100):  # loop over the dataset multiple times
        print("EPOCH ", epoch)
        running_loss = 0.0
        for i, data in enumerate(data_loader_train, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() / 10
            if i % 10 == 0:
                print("LOSS ", running_loss)
        schedular.step()
    print('Finished Training')

    correct = 0
    total = 0
    print('Started Testing')
    with torch.no_grad():
        for data in data_loader_test:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if i % 10 == 0:
                print("ACCURACY ", (correct / total))
    reward = correct / total
    print(reward)

def test(data, num_steps):

    layers = 15

    load_data = {
        "CIFAR":load_data_CIFAR(),
        "MNIST":load_data_MNIST(),
    }
    train_dataloader, test_dataloader, image_size, prev_channels, num_classes = load_data[data]
    data_loader = train_dataloader, test_dataloader
    # Let the controller search in the space of possible solutions
    _controller = controller(layers)
    print("Loading the controller...")
    _controller = load_model(_controller)

    _cnn = cnn(layers, image_size, prev_channels, num_classes)
    state = _cnn.state
    best_state = state
    best_net = _cnn.net
    max_reward = 0
    t1 = time()
    for step in range(num_steps):
        action, logit, = _controller.get_action(state)
        new_state = _cnn.build_child_arch(action)
        state = new_state
        print("New state: ", new_state)
        reward = _cnn.get_reward(data_loader)
        if reward > max_reward:
            best_net = _cnn.net
        print("Step ", step)
        print("Reward: ", reward)
    print("Best architecture found at ",max_reward)
    print("Training the CNN to convergence.")
    net = best_net
    train_CNN(net, data_loader)


if __name__ == '__main__':
    # Parsing the dataset and loading it
    num_steps = 10
    data = "CIFAR"
    parser = argparse.ArgumentParser(description='INAS testing')
    parser.add_argument('--data', choices=['MNIST', 'CIFAR'], default='CIFAR', type=str)
    parser.add_argument('--steps', help='number of search steps', type=int)
    args = parser.parse_args()
    data = args.data
    num_steps = args.steps
    test(data, num_steps)