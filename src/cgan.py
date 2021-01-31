import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Generator(nn.Module):
    def __init__(self, classes, channels, img_size, latent_dim):
        super(Generator, self).__init__() 
        self.classes = classes
        self.channels = channels
        self.img_size = img_size
        self.latent_dim = latent_dim
        self.img_shape = (
            self.channels, 
            self.img_size, 
            self.img_size
        )
        self.label_embedding = nn.Embedding(self.classes, self.classes)
        '''
        The generator network:
        - consists of 5 linear layers
        - 3 of which are connected to batch-norm layer
        - The first 4 linear-layer have 'LeakyReLU' activation functions while the last one use `Tanh` activation function
        '''
        self.model = nn.Sequential(
            *self._create_linear_layer(self.latent_dim + self.classes, 128, normalize = False),
            *self._create_linear_layer(128, 256),
            *self._create_linear_layer(256, 512),
            *self._create_linear_layer(512, 1024),
            nn.Linear(1024, int(torch.prod(self.img_shape))),
            nn.Tanh()
        )

    self._create_linear_layer(self, input_size, output_size, normalize = True):
        layers = [nn.linear(input_size, output_size)]
        if normalize:
            layers.append(nn.BatchNorm1d(output_size))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return layers
    
    def forward(self, noise, labels):
        z = torch.cat(
            (self.label_embedding(labels), noise),
            -1
        )
        x = self.model(z)
        x = x.view(x.size(0), *self.img_shape) # x.shape = [batch_size, *image_shape]
        return x