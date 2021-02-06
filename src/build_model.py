import itertools
import os
import time
from datetime import datetime
from tqdm import tqdm
import logging
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
        assert (self.NetG != None and self.NetD != None),f'Both generator and discriminator cant be none'
    def __repr__(self):
        if self.name == 'cgan':
            return str(repr(self.NetD) + '\n' + repr(self.NetG))

    @property
    def generator(self):
        return self.NetG

    @property
    def discriminator(self):
        return self.NetD
    
    def create_optim(self, lr, alpha=0.5, beta=0.999):
        self.optim_G = torch.optim.Adam(
            params = filter(lambda p: p.requires_grad, self.NetG.parameters()),# grad all Tensor whose `requires_grad` flag is set to True
            lr = lr,
            betas = (alpha, beta)
        )
        self.optim_D = torch.optim.Adam(
            params = filter(lambda p: p.requires_grad, self.NetD.parameters()),
            lr = lr,
            betas = (alpha, beta)
        )
        self.learning_rate = lr # record it merely for later logging

    def train(
        self,
        epochs,
        log_interval=100,
        output_dir='',
        verbose=True,
        save_checkpoints=True
    ):
        # Sets the module in training mode.
        self.NetG.train()
        self.NetD.train()
        viz_noise = torch.randn(
            self.data_loader.batch_size, 
            self.latent_dim, 
            device = self.device
        )
        nrows = self.data_loader.batch_size // 8
        print(f'nrows:{nrows}')
        viz_label = torch.LongTensor(
            np.array([num for _ in range(nrows) for num in range(8)])
        ).to(self.device)

        #####  Set up training related parameters  #####
        n_train = len(self.data_loader.dataset)
        total_time = time.time()

        ##### logging training information #####
        logging.info(f'''Start training:
            Epochs:          {epochs}
            Batch size:      {self.data_loader.batch_size}
            Learning rate:   {self.learning_rate}
            Training size:   {n_train}
            Checkpoints:     {save_checkpoints}
            Device:          {self.device.type}
            Verbose:         {verbose}
        ''')

        #####  Core optimization loop  #####
        for epoch in range(epochs):
            batch_time = time.time()
            with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
                for batch_idx, (data, target) in enumerate(self.data_loader):
                    data, target = data.to(self.device), target.to(self.device)
                    batch_size = data.size(0)
                    real_label = torch.full((batch_size, 1), 1., device=self.device)
                    fake_label = torch.full((batch_size, 1), 0., device=self.device)
                    
                    # Train Generator
                    self.NetG.zero_grad()
                    z_noise = torch.randn(batch_size, self.latent_dim, device=self.device)
                    x_fake_labels = torch.randint(0, self.classes, (batch_size,), device=self.device)
                    x_fake = self.NetG(z_noise, x_fake_labels)
                    y_fake_g = self.NetD(x_fake, x_fake_labels)
                    g_loss = self.NetD.loss(y_fake_g, real_label)

                    g_loss.backward()
                    self.optim_G.step()
                    # Train Discriminator
                    self.NetD.zero_grad()
                    y_real = self.NetD(data, target)
                    d_real_loss = self.NetD.loss(y_real, real_label)

                    y_fake_d = self.NetD(x_fake.detach(), x_fake_labels)
                    d_fake_loss = self.NetD.loss(y_fake_d, fake_label)
                    d_loss = (d_real_loss + d_fake_loss) / 2

                    pbar.set_postfix(
                        **{
                        'Generator loss (batch)': g_loss.item(),
                        'Discriminator loss (batch)': d_loss.item()
                        }
                    )

                    d_loss.backward()
                    self.optim_D.step()
                    pbar.update(batch_size)

                    if verbose and batch_idx % (log_interval) == 0 and batch_idx > 0:
                        if self.name == 'cgan':
                            print('\nEpoch {} [{}/{}] loss_D: {:.4f} loss_G: {:.4f} time: {:.2f}'.format(
                                epoch, batch_idx, len(self.data_loader),
                                d_loss.mean().item(),
                                g_loss.mean().item(),
                                time.time() - batch_time)
                            )
                        vutils.save_image(data, os.path.join(output_dir, 'real_samples.png'), normalize=True)

                    with torch.no_grad():
                        viz_sample = self.NetG(viz_noise, viz_label)
                        vutils.save_image(viz_sample, os.path.join(output_dir, 'fake_samples_{}.png'.format(epoch)), nrow=8, normalize=True)
                    batch_time = time.time()
                if save_checkpoints:
                    self.save_to(path=output_dir, name=self.name, verbose=True)
        if verbose:
            logging.info('Total train time: {:.2f}'.format(time.time() - total_time))

    def eval(
        self,
        mode=None,
        batch_size=None
    ):
        # Sets the module in evaluation mode.
        self.NetG.eval()
        self.NetD.eval()

    def save_to(
        self,
        path='',
        name=None,
        verbose=True
    ):
        if name is None:
            name = self.name
        if verbose:
            logging.info('\nSaving models to {}_G.pt and {}_D.pt ...'.format(name, name))
        torch.save(self.NetG.state_dict(), os.path.join(path, '{}_G.pt'.format(name)))
        torch.save(self.NetD.state_dict(), os.path.join(path, '{}_D.pt'.format(name)))