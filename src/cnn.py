""" This module has to build the child architecture
and save the current architecture
    It has to implement:
    - build_child_arch(action, previous_state)
"""
import torch
from torch import nn
import torch.optim as optim
from conv_net import conv_net
import numpy as np

max_layers = 2
class cnn():

    def __init__(self, max_layers, image_size, prev_channels, num_classes, epochs=3):
        #TODO
        # size of filter, stride, channels, maxpool(boolean), max_pool_size
        # Droput? Use same padding for now. 
        # (We may have to change the image_size if we use same)
        initial_state = [3,1,32,0,2,3,1,64,0,2]#*max_layers #0 means yes to max_pool
        self.state = initial_state
        self.image_size = image_size
        self.prev_channels = prev_channels
        self.num_classes = num_classes
        self.max_layers= max_layers
        self.op_add = [lambda x: x+1, lambda x: x, lambda x: x-1]
        self.op_mul = [lambda x: x*2, lambda x: x, lambda x: x/2]
        self.epochs = epochs
        return
    
    def update_image_size(self, state):
        n = self.image_size
        k = state[0]  # filter_size
        s = 1                   # stride
        return (n-k)/s + 1
    

    def build_child_arch(self, action):
        #TODO
        #max_pool, cnn or avg_pool
        state = []
        for layer in range(self.max_layers):
            action0 = action[0+layer*5]
            action1 = action[1+layer*5]
            action2 = action[2+layer*5]
            action3 = action[3+layer*5]
            action4 = action[4+layer*5]
            state0  = self.op_add[action0](self.state[0+layer*5])
            state1  = self.op_add[action1](self.state[1+layer*5])
            state2  = self.op_mul[action2](self.state[2+layer*5])
            state3  = action3
            state4  = self.op_add[action4](self.state[4+layer*5])
            layer_state = [state0,state1,state2,state3,state4]
            layer_state, _ = self.check_state(layer_state, layer)
            state.extend(layer_state)
            self.image_size = self.update_image_size(layer_state)
 
        self.state=state
        self.net = conv_net(state, input_size=self.image_size , prev_channels = self.prev_channels, n_class=self.num_classes)
        return
    
    def check_state(self, state, layer):
        count = 0
        padding = np.ceil(((state[1]-1)*self.image_size - state[1] + state[0])/2)
        # 0:size of filter, 1:stride, 2:channels, 3:maxpool(boolean), 4:max_pool_size
        # We must be careful about everything except 3: maxpool(boolean)
        if (state[0]<=0 or state[0]>self.image_size):
            state[0] = self.state[0+layer*5]
            count = count+1
        if (state[1]<=0 or state[1]>self.image_size + padding - state[0]): # add later 
            state[1] = self.state[1+layer*5]
            count = count+1
        if (state[2]<=0 or state[2] > 256): # later, penalty for the running time
            state[2] = self.state[2+layer*5]
            count = count+1
        if (state[4]<=0 or state[4] >= self.image_size):
            state[4] = self.state[4+layer*5]
            count = count+1
        
        return state, count
    
    
        
        
    
    def get_reward(self, data_loader):
        data_loader_train, data_loader_test = data_loader
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.net.parameters(), lr=0.001, momentum=0.9)
        for epoch in range(self.epochs):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, data in enumerate(data_loader_train, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
        
                # zero the parameter gradients
                optimizer.zero_grad()
        
                # forward + backward + optimize
                outputs = self.net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
        
                # print statistics
                running_loss += loss.item()
                if i % 2000 == 1999:    # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0

        print('Finished Training')
        
        class_correct = list(0. for i in range(self.num_classes))
        class_total = list(0. for i in range(self.num_classes))
        with torch.no_grad():
            for data in data_loader_test:
                images, labels = data
                outputs = self.net(images)
                _, predicted = torch.max(outputs, 1)
                c = (predicted == labels).squeeze()
                for i in range(4):
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1
        
        
        for i in range(10):
            print('Accuracy of %5s : %2d %%' % (
                100 * class_correct[i] / class_total[i]))
            
        reward = sum(class_correct)/sum(class_total)
        
        return reward