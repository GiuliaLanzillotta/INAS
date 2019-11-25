from torch import nn
import numpy as np

class conv_net(nn.Module):

    def __init__(self, conv_layers, input_size=32, prev_channels=3, n_class=10, device='cpu'):
        super(conv_net, self).__init__()

        self.input_size = input_size
        self.n_class = n_class

        layers = []
        img_dim = input_size

        # We don't seem to have pooling
        print(conv_layers)

        for kernel_size, stride, n_channels, pooling, pooling_size in [conv_layers[x:x+5] for x in range(0, len(conv_layers)-1, 5)]:
            n = img_dim
            p = int(np.ceil(((n-1)*stride - n + kernel_size)/2))
            layers += [
                nn.Conv2d(int(prev_channels), int(n_channels), int(kernel_size), stride=int(stride), padding=p),
                nn.ReLU()
            ]
            img_dim = self.update_size(img_dim, int(kernel_size), int(stride), p)

            # pooling =0 is max_poos, 1 is avg_pool and 2 is no_pool
            if pooling==0:
                layers += [
                    nn.MaxPool2d(kernel_size=pooling_size, stride=1, padding=0)
                ]
                img_dim = self.update_size(img_dim, pooling_size, 1, 0)
            if pooling==1:
                layers += [
                        nn.AvgPool2d(kernel_size = pooling_size, stride=1,padding=0)
                ]
                img_dim = self.update_size(img_dim, pooling_size, 1, 0)

            prev_channels = n_channels
        #layers += [nn.Flatten(1,-1)]

        self.prev_fc_size = int(int(prev_channels) * img_dim * img_dim)
        
        layers += [nn.Linear(self.prev_fc_size, n_class)]
        self.layers = layers
        self.layers = nn.ModuleList(layers)
        #self.layers = nn.Sequential(*layers)
        
    def update_size(self, image_size, kernel_size, stride, padding):
        return int((image_size - kernel_size + 2*padding)/stride + 1)

    def forward(self, x):
        for i,layer in enumerate(self.layers):
            if(i==len(self.layers)-1):
                x = x.flatten(1,-1)
            x = layer(x)
        return x

