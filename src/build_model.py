import itertools
import os
import time
from datetime import datetime
import numpy as np
import torch
import torchvision.utils as vutils
import utils
from cgan import (Generator, Discriminator)

class ConditionModel(object):
    def __init__(
        self,
        name,
        device,
        data_loader,
        classes,
        channels,
        img_size,
        latent_dim
    ):
    self.name = name
    self.device = device
    self.data_loader = data_loader
    self.classes = classes
    self.channels = channels
    self.img_size = img_size
    self.latent_dim = latent_dim
    self.NetG = None
    self.NetD = None
    self.optim_G = None
    self.optim_D = None
    if self.name == 'cgan':
        self.NetG = Generator(
            self.classes, 
            self.channels, 
            self.img_size, 
            self.latent_dim
        )
        self.NetD = Discriminator(
            self.classes, 
            self.channels, 
            self.img_size, 
            self.latent_dim
        )
        self.NetG.to(self.device)
        self.NetD.to(self.device)
    assert (self.NetG != None and self.NetD != None) \
                    f'Both generator and discriminator cant be none' 