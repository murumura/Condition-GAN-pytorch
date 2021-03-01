import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Upsample(nn.Module):
    def __init__(self, scale_factor):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
    def forward(self, x):
        x = F.interpolate(
            x, 
            scale_factor=self.scale_factor,
            mode='bilinear',
            align_corners=False
        )
        return x

class Generator(nn.Module):
    def __init__(self, classes, channels, img_size, latent_dim, code_dim):
        super(Generator, self).__init__()
        self.classes = classes
        self.channels = channels # image channels
        self.img_size = img_size
        self.img_init_size = self.img_size // 4 # Initial size before upsampling
        self.latent_dim = latent_dim
        self.code_dim = code_dim
        self.img_init_shape = (
            128, 
            self.img_init_size, 
            self.img_init_size
        )
        self.img_shape = (
            self.channels, 
            self.img_size, 
            self.img_size
        )
        '''
        the first hidden layer that transforms the input tensor with a length of 
        (latent_dim + classes + code_dim) into tensor with shape (128 * img_init_size * img_init_size)
        '''
        self.stem_linear = nn.Sequential(
            nn.Linear(self.latent_dim + self.classes + self.code_dim, int(np.prod(self.img_init_shape)))
        )
        '''
        the output tensor of first hidden layer with shape  (128 * img_init_size * img_init_size) will 
        be first feed into the custom upsample layer to up-scaled its feature map
        '''
        self.model = nn.Sequential(
            nn.BatchNorm2d(128),
            *self._create_deconv_layer(
                input_size=128,
                output_size=128,
                upsample= True
            ),
            *self._create_deconv_layer(
                input_size=128,
                output_size=64,
                upsample= True
            ),
            *self._create_deconv_layer(
                input_size=64,
                output_size=self.channels,
                upsample=False,
                normalize=False
            ),
            nn.Tanh()
        )
    def _create_deconv_layer(self, input_size, output_size, upsample=True, normalize=True):
        layers = []
        if upsample:
            '''
            using the custom 'Upsample' layer to increse the size of the feature map
            '''
            layers.append(
                Upsample(scale_factor = 2)
            )
        layers.append(
            nn.Conv2d(input_size, output_size, kernel_size=3, stride=1, padding=1) 
        )
        if normalize:
            layers.append(nn.BatchNorm2d(output_size, momentum=0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        return layers

    def forward(self, noise, labels, code):
        z = torch.cat((noise, labels, code), -1)
        z_vec = self.stem_linear(z)
        z_img = z_vec.view(z_vec.shape[0], *self.img_init_shape) # z_img.shape = [batch_size, *image_shape]
        x = self.model(z_img)
        return x

class Discriminator(nn.Module):
    def __init__(self, classes, channels, img_size, latent_dim, code_dim):
        super(Discriminator, self).__init__()
        self.classes = classes
        self.channels = channels
        self.img_size = img_size
        self.latent_dim = latent_dim
        self.code_dim = code_dim
        self.img_shape = (
            self.channels,
            self.img_size,
            self.img_size
        )
        self.model = nn.Sequential(
            *self._create_conv_layer(
                input_size=self.channels,
                output_size=16,
                drop_out=True, 
                normalize=False
            ),
            *self._create_conv_layer(
                input_size=16,
                output_size=32,
                drop_out=True, 
                normalize=True
            ),
            *self._create_conv_layer(
                input_size=32,
                output_size=64,
                drop_out=True, 
                normalize=True
            ),
            *self._create_conv_layer(
                input_size=64,
                output_size=128,
                drop_out=True, 
                normalize=True
            ),
        )
        out_linear_dim = 128 * (self.img_size // 16) * (self.img_size // 16)
        # output layers
        # real/fake
        self.adv_linear = nn.Linear(out_linear_dim, 1)
        # auxiliary vector output: include class fidelity and latent code fidelity 
        self.class_linear = nn.Sequential(
            nn.Linear(out_linear_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, self.classes)
        )
        self.code_linear = nn.Sequential(
            nn.Linear(out_linear_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, self.code_dim)
        )
        self.adv_loss = torch.nn.MSELoss()
        self.class_loss = torch.nn.CrossEntropyLoss()
        self.continuous_loss = torch.nn.MSELoss() # style fidelity 

    def _create_conv_layer(self, input_size, output_size, drop_out=True, normalize=True):
        layers = [nn.Conv2d(in_channels=input_size, out_channels=output_size, kernel_size=3, stride=2, padding=1)]
        if drop_out:
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Dropout(0.4))
        if normalize:
            layers.append(nn.BatchNorm2d(output_size, 0.8))
        return layers
    
    def forward(self, image):
        y_img = self.model(image)
        y_vec = y_img.view(y_img.shape[0], -1)
        validity = self.adv_linear(y_vec) # output real/fake
        label = F.softmax(self.class_linear(y_vec), dim=1)
        latent_code = self.code_linear(y_vec)
        return validity, label, latent_code