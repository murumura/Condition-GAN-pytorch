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
from infogan import Generator as infoganG, Discriminator as infoganD
from utils import (plot_cgan_loss, create_gif)


def _weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

class ConditionModel(object):
    def __init__(
        self, 
        name, 
        device, 
        data_loader,
        classes, 
        channels, 
        img_size, 
        latent_dim,
        code_dim=2
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
        self.optim_info = None
        self.code_dim = None
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
        elif self.name == 'infogan':
            self.code_dim = code_dim
            self.NetG = infoganG(
                self.classes, 
                self.channels, 
                self.img_size, 
                self.latent_dim,
                code_dim=self.code_dim
            )
            self.NetG.apply(_weights_init_normal)
            self.NetD = infoganD(
                self.classes, 
                self.channels, 
                self.img_size, 
                self.latent_dim,
                code_dim=self.code_dim
            )
            self.NetD.apply(_weights_init_normal)
            self.NetG.to(self.device)
            self.NetD.to(self.device)
        assert (self.NetG != None and self.NetD != None),f'Both generator and discriminator cant be none'
    def __repr__(self):
        if self.name == 'cgan' or self.name == 'infogan':
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
        if self.name == 'infogan':
            self.optim_info = torch.optim.Adam(
                itertools.chain(self.NetG.parameters(), self.NetD.parameters()),
                lr=lr, 
                betas=(alpha, beta)
            )
        self.learning_rate = lr # record it merely for later logging

    
    def _to_onehot(self, var, dim):
        '''
        Transforming the label values to one-hot coding
        '''
        res = torch.zeros((var.shape[0], dim), device=self.device)
        res[range(var.shape[0]), var] = 1.
        return res

    def train(
        self,
        epoches,
        log_interval=100,
        output_dir='',
        verbose=True,
        save_checkpoints=True
    ):
        # Sets the module in training mode.
        self.NetG.train()
        self.NetD.train()
        viz_z = torch.zeros(
            (self.data_loader.batch_size, self.latent_dim), 
            device=self.device
        )
        viz_noise = torch.randn(
            self.data_loader.batch_size, 
            self.latent_dim, 
            device = self.device
        )
        nrows = self.data_loader.batch_size // 8
        print(f'nrows:{nrows}, batch_size:{self.data_loader.batch_size}')
        # generate visible data for image synsethesis
        viz_label = torch.LongTensor(
            np.array([num for _ in range(nrows) for num in range(8)])
        ).to(self.device)
        # transform label into one-hot vector
        viz_onehot = self._to_onehot(
            viz_label, 
            dim=self.classes
        )
        viz_code = torch.zeros(
            (self.data_loader.batch_size, self.code_dim), 
            device=self.device
        )
        #####  Set up training related parameters  #####
        n_train = len(self.data_loader.dataset)
        total_time = time.time()

        ##### logging training information #####
        logging.info(f'''Start training:
            epoches:          {epoches}
            Batch size:      {self.data_loader.batch_size}
            Learning rate:   {self.learning_rate}
            Training size:   {n_train}
            Checkpoints:     {save_checkpoints}
            Device:          {self.device.type}
            Verbose:         {verbose}
        ''')
        #####  Set up training loss record parameters #####
        train_hist = {}
        if self.name == 'cgan':
            train_hist['D_losses'] = []
            train_hist['G_losses'] = []
        elif self.name == 'infogan':
            train_hist['D_losses'] = []
            train_hist['G_losses'] = []
            train_hist['Info_losses'] = []
        #####  Core optimization loop  #####
        for epoch in range(epoches):
            batch_time = time.time()
            if self.name == 'cgan':
                D_losses = []
                G_losses = []
            elif self.name == 'infogan':
                D_losses = []
                G_losses = []
                Info_losses = []
            with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epoches}', unit='img') as pbar:
                for batch_idx, (data, target) in enumerate(self.data_loader):
                    data, target = data.to(self.device), target.to(self.device)
                    batch_size = data.size(0)
                    real_label = torch.full((batch_size, 1), 1., device=self.device)
                    fake_label = torch.full((batch_size, 1), 0., device=self.device)
                    
                    # Train Generator
                    self.NetG.zero_grad()
                    z_noise = torch.randn(batch_size, self.latent_dim, device=self.device)
                    '''
                    we sample a single feature c from a uniform distribution and convert it to a 1-hot vector.
                    then the generator uses this vector and z to generate an image.
                    '''
                    x_fake_labels = torch.randint(0, self.classes, (batch_size,), device=self.device)
                    if self.name == 'cgan':
                        x_fake = self.NetG(z_noise, x_fake_labels)
                        y_fake_g = self.NetD(x_fake, x_fake_labels)
                        g_loss = self.NetD.loss(y_fake_g, real_label)
                    elif self.name == 'infogan':
                        labels_onehot = self._to_onehot(x_fake_labels, dim=self.classes)
                        z_code = torch.zeros((batch_size, self.code_dim), device=self.device).normal_()
                        
                        x_fake = self.NetG(z_noise, labels_onehot, z_code)
                        y_fake_g, _, _ = self.NetD(x_fake)
                        g_loss = self.NetD.adv_loss(y_fake_g, real_label)

                    G_losses.append(g_loss.item()) # record generator loss
                    g_loss.backward()
                    self.optim_G.step()

                    # Train Discriminator
                    self.NetD.zero_grad()
                    if self.name == 'cgan':
                        y_real = self.NetD(data, target)
                        d_real_loss = self.NetD.loss(y_real, real_label)

                        y_fake_d = self.NetD(x_fake.detach(), x_fake_labels)
                        d_fake_loss = self.NetD.loss(y_fake_d, fake_label)
                    elif self.name == 'infogan':
                        y_real, _ , _= self.NetD(data) # unsupervised 
                        d_real_loss = self.NetD.adv_loss(y_real, real_label)

                        y_fake_d, _, _ = self.NetD(x_fake.detach())
                        d_fake_loss = self.NetD.adv_loss(y_fake_d, fake_label)
                        
                    d_loss = (d_real_loss + d_fake_loss) / 2
                    D_losses.append(d_loss.item()) # record discriminator loss
                    pbar.set_postfix(
                        **{
                        'Generator loss (batch)': g_loss.item(),
                        'Discriminator loss (batch)': d_loss.item()
                        }
                    )
                    d_loss.backward()
                    self.optim_D.step()
                    pbar.update(batch_size)
                    # update infogan's mutual information
                    if self.name == 'infogan':
                        self.optim_info.zero_grad()
                        z_noise.normal_()
                        x_fake_labels = torch.randint(0, self.classes, (batch_size,), device=self.device)
                        labels_onehot = self._to_onehot(x_fake_labels, dim=self.classes)
                        z_code.normal_()
                        x_fake = self.NetG(z_noise, labels_onehot, z_code)
                        _, label_fake, code_fake = self.NetD(x_fake)
                        info_loss = self.NetD.class_loss(label_fake, x_fake_labels) +\
                                self.NetD.continuous_loss(code_fake, z_code)
                        info_loss.backward()
                        self.optim_info.step()

                    if verbose and batch_idx % (log_interval) == 0 and batch_idx > 0:
                        if self.name == 'cgan':
                            print('\nEpoch {} [{}/{}] loss_D: {:.4f} loss_G: {:.4f} time: {:.2f}'.format(
                                epoch, batch_idx, len(self.data_loader),
                                d_loss.mean().item(),
                                g_loss.mean().item(),
                                time.time() - batch_time)
                            )
                        elif self.name == 'infogan':
                            print('\nEpoch {} [{}/{}] loss_D: {:.4f} loss_G: {:.4f} loss_I: {:.4f} time: {:.2f}'.format(
                                epoch, batch_idx, len(self.data_loader),
                                d_loss.mean().item(),
                                g_loss.mean().item(),
                                info_loss.mean().item(),
                                time.time() - batch_time)
                            )
                        vutils.save_image(data, os.path.join(output_dir, 'real_samples.png'), normalize=True)
                    with torch.no_grad():
                        if self.name == 'cgan':
                            viz_sample = self.NetG(viz_noise, viz_label)
                            vutils.save_image(viz_sample, os.path.join(output_dir, 'fake_samples_{}.png'.format(epoch+1)), nrow=8, normalize=True)
                        elif self.name == 'infogan':
                            viz_sample = self.NetG(viz_noise, viz_onehot, viz_code)
                            vutils.save_image(viz_sample, os.path.join(output_dir, 'fake_samples_{}.png'.format(epoch+1)), nrow=8, normalize=True)
                    batch_time = time.time()
                if save_checkpoints:
                    self.save_to(path=output_dir, name=self.name, verbose=True)
            # record batch training loss
            if self.name == 'cgan':
                train_hist['D_losses'].append(torch.mean(torch.FloatTensor(D_losses)))
                train_hist['G_losses'].append(torch.mean(torch.FloatTensor(G_losses)))
                plot_cgan_loss(
                    d_loss=train_hist['D_losses'], 
                    g_loss=train_hist['G_losses'], 
                    num_epoch=epoch + 1, 
                    epoches=epoches, 
                    save_dir=output_dir
                )
        # plot git of training loss and synsethesis images
        create_gif(
            epoches=epoches, 
            save_dir=output_dir,
            gan_name_prefix=self.name
        )
        if verbose:
            logging.info('Total train time: {:.2f}'.format(time.time() - total_time))

    def eval(
        self,
        mode=None,
        batch_size=None,
        output_dir=None
    ):
        # Sets the module in evaluation mode.
        self.NetG.eval()
        self.NetD.eval()
        if batch_size is None:
            batch_size = self.data_loader.batch_size
        nrows = batch_size // 8
        viz_labels = np.array([num for _ in range(nrows) for num in range(8)])
        viz_labels = torch.LongTensor(viz_labels).to(self.device)
        with torch.no_grad():
            if self.name == 'cgan':
                viz_tensor = torch.randn(batch_size, self.latent_dim, device=self.device)
                viz_sample = self.NetG(viz_tensor, viz_labels) # generated image from random noise
            elif self.name == 'infogan':
                viz_tensor = torch.randn(batch_size, self.latent_dim, device=self.device)
                labels_onehot = self._to_onehot(viz_labels, dim=self.classes)
                z_code = torch.zeros((batch_size, self.code_dim), device=self.device)
                if mode is not None:
                    for i in range(batch_size):
                        z_code[i, mode] = 4. * i / batch_size - 2.
                viz_sample = self.NetG(viz_tensor, labels_onehot, z_code)

            viz_vector = utils.to_np(viz_tensor).reshape(batch_size, self.latent_dim)
            cur_time = datetime.now().strftime("%Y%m%d-%H%M%S")
            np.savetxt(os.path.join(output_dir, 'vec_{}.txt'.format(cur_time)), viz_vector)
            vutils.save_image(viz_sample, os.path.join(output_dir, 'img_{}.png'.format(cur_time)), nrow=8, normalize=True)
            logging.info(f'\nSaving evaluation image to {output_dir}...')

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

    def load_state_from(
        self,
        path='',
        name=None,
        verbose=True
    ):
        if name is None:
            name = self.name
        if verbose:
            logging.info('\nLoading models from {}_G.pt and {}_D.pt ...'.format(name, name))
        ckpt_G = torch.load(os.path.join(path, '{}_G.pt'.format(name)))
        if isinstance(ckpt_G, dict) and 'state_dict' in ckpt_G:
            self.NetG.load_state_dict(ckpt_G['state_dict'], strict=True)
        else:
            self.NetG.load_state_dict(ckpt_G, strict=True)
        ckpt_D = torch.load(os.path.join(path, '{}_D.pt'.format(name)))
        if isinstance(ckpt_D, dict) and 'state_dict' in ckpt_D:
            self.NetD.load_state_dict(ckpt_D['state_dict'], strict=True)
        else:
            self.NetD.load_state_dict(ckpt_D, strict=True)