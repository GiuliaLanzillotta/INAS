""" This module has to build the child architecture
and save the current architecture
    It has to implement:
    - build_child_arch(action, previous_state)
    - check_states
    -update_size
    -get_reward
"""
import torch
from torch import nn
import torch.optim as optim
from src.conv_net import conv_net
import numpy as np

max_layers = 15

"Set device to gpu if available otherwise cpu"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class cnn():

    def __init__(self, max_layers, image_size, prev_channels, num_classes, train, epochs=10):

        initial_state = [3, 1, 32, 2, 2,
                         3, 1, 32, 2, 2,
                         3, 1, 64, 2, 2,
                         3, 1, 64, 2, 2,
                         3, 1, 64, 0, 2,
                         3, 1, 128, 2, 2,
                         3, 1, 128, 2, 2,
                         3, 1, 128, 2, 2,
                         3, 1, 128, 0, 2,
                         3, 1, 128, 2, 2,
                         3, 1, 128, 0, 2,
                         3, 1, 128, 2, 2,
                         3, 1, 128, 0, 2,
                         3, 1, 128, 2, 2,
                         3, 1, 128, 2, 2]
        self.state = initial_state
        self.train = train
        self.image_size = image_size
        self.original_image_size = image_size
        self.prev_channels = prev_channels
        self.num_classes = num_classes
        self.max_layers= max_layers
        self.op_add = [lambda x: x+1 , lambda x: x, lambda x: x-1]
        self.op_mul = [lambda x: x*2, lambda x: x, lambda x: x/2]
        self.epochs = epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return

    """Get the padding"""
    @staticmethod
    def get_padding(image_size, kernel_size, stride):
        return np.ceil(((kernel_size - 1) * image_size - stride + kernel_size) / 2)

    """Update_size fixes the image size in order to carry out the relevent check for each layer!"""
    @staticmethod
    def update_size(image_size, kernel_size, stride, padding):
        return int((image_size - kernel_size + 2 * padding) / stride + 1)

    """The build_child_arch generates the CNN by taking the action and applying it to the state, the state contains
    the hyperparameters of the CNN architecture. """
    def build_child_arch(self, action):
        state = []
        self.image_size=self.original_image_size
        for layer in range(self.max_layers):
            action0 = action[0+layer*5]
            action1 = action[1+layer*5]
            action2 = action[2+layer*5]
            action3 = action[3+layer*5]
            action4 = action[4+layer*5]
            state0 = self.op_add[action0](self.state[0+layer*5])   # Filter Size, +1, 0, -1
            state1 = self.op_add[action1](self.state[1+layer*5])   # Stride, +1, 0, -1
            state2 = self.op_mul[action2](self.state[2+layer*5])   # Number of Channels, *2, 0, /2
            state3 = action3                                       # Pooling, Yes or No?
            state4 = self.op_add[action4](self.state[4+layer*5])   # Pooling Size, +1, 0, -1
            layer_state = [state0,state1,state2,state3,state4]
            layer_state = self.check_state(layer_state, layer)
            state.extend(layer_state)
            padding = self.get_padding(self.image_size, state[0], state[1])
            self.image_size = self.update_size(self.image_size, state[0], state[1], padding)
            if state[3] == 2:
                self.image_size = self.update_size(self.image_size, state[4], 1, 0)

        self.net = conv_net(state, input_size=self.original_image_size, prev_channels = self.prev_channels, n_class=self.num_classes,device = self.device)
        self.net = self.net.to(device)

        self.state = state
        return self.state

    """The check_state function ensures the that the state does not violate certain bounds
       which would not make sense, i.e. stride = 0, more explanation in section 3.3 of the report"""
    def check_state(self, state, layer):
        padding = np.ceil(((state[1]-1)*self.image_size - state[1] + state[0])/2)

        "0:size of filter, 1:stride, 2:channels, 3:maxpool(boolean), 4:max_pool_size"

        if (state[0]<1 or state[0]>self.image_size):
            state[0] = self.state[0+layer*5]

        if (state[1]<1 or state[1]>self.image_size + padding - state[0]):
            state[1] = self.state[1+layer*5]

        if (state[2]<1 or state[2] > 128):
            state[2] = self.state[2+layer*5]

        if (state[4]<1 or state[4] >= self.image_size):
            state[4] = self.state[4+layer*5]

        return state


    """The get_reward function trains the CNN architecture sampled, 
      and produces a reward for the REINFORCE Algorithm"""
    def get_reward(self, data_loader):
        data_loader_train, data_loader_test = data_loader
        "Define Optimizer, Schedular, and Loss Function"
        criterion = nn.CrossEntropyLoss()
        if self.train == True:
            optimizer = optim.SGD(self.net.parameters(), lr=0.005, momentum=0.9, nesterov=True)
            schedular = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.85)
        else:
            optimizer = optim.SGD(self.net.parameters(), lr=0.005, momentum=0.9, nesterov=True)
            schedular = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.8)

        "Main Training Loop"
        for epoch in range(self.epochs):
            "Reset Loss for every epoch"
            running_loss = 0.0

            for i, data in enumerate(data_loader_train,0):
                "Configure Inputs and sent to GPU if available"
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
        
                "Optimizer Gradient Reset"
                optimizer.zero_grad()
        
                "forward + loss + backward + optimize"
                outputs = self.net(inputs)

                loss = criterion(outputs, labels)

                loss.backward()

                optimizer.step()

                "Print Statistics"
                running_loss += loss.item()
                if i % 300 == 299:
                    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 300))
                    running_loss = 0.0
                "Train on 51,200 samples then break"
                if i % 800 == 799:
                    break
            schedular.step()

        print('Finished Training CNN')
        "Calculate Accuracy of CNN on Validation Set"
        correct = 0
        total = 0
        with torch.no_grad():
            for data in data_loader_test:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = self.net(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        "Compute Reward: Reward is the validation accuracy cubed, reward is cubed in the controller update policy!"
        reward = correct / total
        return reward

