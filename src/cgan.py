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

    def _create_linear_layer(self, input_size, output_size, normalize = True):
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

class Discriminator(nn.Module):
    def __init__(self, classes, channels, img_size, latent_dim):
        super(Discriminator, self).__init__()
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
        self.adversarial_loss = torch.nn.BCELoss()
        '''
        The discriminator network:
        - consists of 5 linear layers
        - 2 of which are connected to Dropout layer(for enhance generlization capability)
        - In the last, using a sigmoid function to make sure the output value is lie within the range [0,1]
        '''
        self.model = nn.Sequential(
            *self._create_linear_layer(
                input_size = self.classes + int(torch.prod(self.img_shape))
                output_size = 1024,
                drop_out = False,
                activation_func = True
            ),
            *self._create_linear_layer(
                input_size = 1024
                output_size = 512,
                drop_out = True,
                activation_func = True
            ),
            *self._create_linear_layer(
                input_size = 512
                output_size = 256,
                drop_out = True,
                activation_func = True
            ),
            *self._create_linear_layer(
                input_size = 256
                output_size = 128,
                drop_out = False,
                activation_func = False
            ),
            *self._create_linear_layer(
                input_size = 128
                output_size = 1,
                drop_out = False,
                activation_func = False
            ),
            nn.Sigmoid()
        )
    def _create_linear_layer(self, input_size, output_size, drop_out = True, activation_func = True):
        layers = [nn.linear(input_size, output_size)]
        if drop_out:
            layers.append(nn.Dropout(0.4))
        if activation_func:
            layers.append(nn.LeakyReLU(0.2, inplace = True))
        return layers

    def forward(self, output, label):
        '''
        include the label information in input by concatenation the embedding of labels with image vector
        '''
        x = torch.cat(
            (image.view(image.size(0), -1), self.label_embedding(labels)),
            -1
        )
        return self.model(x)

    def loss(self, output, label):
        return self.adversarial_loss(output, label)