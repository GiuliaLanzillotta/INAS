from torch import nn
import numpy as np
import torch

class conv_net(nn.Module):
    """ This class represents the child architecture and is therefore respondible
    for the training and testing."""

    def __init__(self, conv_layers, input_size, prev_channels, n_class, device = 'cuda'):
        super(conv_net, self).__init__()

        self.input_size = input_size
        self.n_class = n_class
        self.device = device
        self.prev_channels = prev_channels

        # We build the child layer by layer using a ModuleList
        layers = []
        # Initialise image dimension as the input size. It is important
        # to keep track of the image dimension while building the architecture
        img_dim = input_size
        # We first iterate over the state and translate each hyper-parameter
        prev_channels = self.prev_channels
        # in the state into an actual hyper-parameter of the child architecture
        for kernel_size, stride, n_channels, pooling, pooling_size in \
                [conv_layers[x:x+5] for x in range(0, len(conv_layers)-1, 5)]:
            n = img_dim
            # calculate the padding with the SAME rule
            p = int(np.ceil(((n-1)*stride - n + kernel_size)/2))
            # For each convolutional layer we add :
            # - convolution
            # - activation
            # - batch normalisation
            layers += [
                nn.Conv2d(int(prev_channels), int(n_channels), int(kernel_size), stride=int(stride), padding=p),
                nn.ELU(),
                nn.BatchNorm2d(int(n_channels))
            ]
            # update the image dimension after convolution
            img_dim = self.update_size(img_dim, int(kernel_size), int(stride), p)
            # Adding pooling layer (if specified)
            # pooling =0 is max_poos, 1 is avg_pool and 2 is no_pool
            if pooling==0:
                layers += [
                    nn.MaxPool2d(kernel_size=pooling_size, stride=1, padding=0),
                    nn.Dropout(0.2)
                ]
                img_dim = self.update_size(img_dim, pooling_size, 1, 0)
            if pooling==1:
                layers += [
                        nn.AvgPool2d(kernel_size = pooling_size, stride=1,padding=0)
                ]
                img_dim = self.update_size(img_dim, pooling_size, 1, 0)

            prev_channels = n_channels
        self.prev_fc_size = int(int(prev_channels) * img_dim * img_dim)

        # Add the head to the CNN , which consists of 2 linear layers
        layers += [nn.Dropout(0.2),
                   nn.Linear(self.prev_fc_size, 128),
                   nn.ELU(),
                   nn.Linear(128, n_class)
                   ]
        self.layers = layers
        self.layers = nn.ModuleList(layers)

    def update_size(self, image_size, kernel_size, stride, padding):
        return int((image_size - kernel_size + 2*padding)/stride + 1)

    def forward(self, x):
        x = x.to(self.device)
        # Forwarding layer by layer
        for i,layer in enumerate(self.layers):
            if(i==len(self.layers)-4):
                x = x.flatten(1,-1)
            x = layer(x)
        return x

