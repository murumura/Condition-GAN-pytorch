import errno
import os
import shutil
import sys
import torch
import imageio
#To plot pretty picture
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.switch_backend("agg") #release "RuntimeError: main thread is not in main loop" error
import numpy as np
def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

def to_np(var: torch.Tensor):
    """Exports torch.Tensor to Numpy array.
    """
    return var.detach().cpu().numpy()

def create_folder(folder_path):
    """Create a folder if it does not exist.
    """
    try:
        os.makedirs(folder_path)
    except OSError as _e:
        if _e.errno != errno.EEXIST:
            raise

def clear_and_create_folder(folder_path):
    """Clear all contents recursively if the folder exists.
    Create the folder if it has been accidently deleted.
    """
    create_folder(folder_path)
    for the_file in os.listdir(folder_path):
        _file_path = os.path.join(folder_path, the_file)
        try:
            if os.path.isfile(_file_path):
                os.unlink(_file_path)
            elif os.path.isdir(_file_path):
                shutil.rmtree(_file_path)
        except OSError as _e:
            print(_e)

def plot_cgan_loss(d_loss, g_loss, num_epoch, epoches, save_dir):
    
    fig, ax = plt.subplots()
    ax.set_xlim(0,epoches + 1)
    ax.set_ylim(0, max(np.max(g_loss), np.max(d_loss)) * 1.1)
    plt.xlabel('Epoch {}'.format(num_epoch))
    plt.ylabel('Loss')
    
    plt.plot([i for i in range(1, num_epoch + 1)], d_loss, label='Discriminator', color='red', linewidth=3)
    plt.plot([i for i in range(1, num_epoch + 1)], g_loss, label='Generator', color='mediumblue', linewidth=3)
    
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'cgan_loss_epoch_{}.png'.format(num_epoch)))
    plt.close()

def plot_infogan_loss(d_loss, g_loss, info_loss, num_epoch, epoches, save_dir):

    fig, ax = plt.subplots()
    ax.set_xlim(0,epoches + 1)
    ax.set_ylim(0, max(np.max(g_loss), np.max(d_loss)) * 1.1)
    plt.xlabel('Epoch {}'.format(num_epoch))
    plt.ylabel('Loss')
    
    plt.plot([i for i in range(1, num_epoch + 1)], d_loss, label='Discriminator', color='red', linewidth=3)
    plt.plot([i for i in range(1, num_epoch + 1)], g_loss, label='Generator', color='mediumblue', linewidth=3)
    plt.plot([i for i in range(1, num_epoch + 1)], info_loss, label='Info', color='coral', linewidth=3)

    plt.legend()
    plt.savefig(os.path.join(save_dir, 'infogan_loss_epoch_{}.png'.format(num_epoch)))
    plt.close()

def create_gif(epoches, save_dir, gan_name_prefix):
    """
    create gif of plotted loss and synsethesis images
    """
    images = []
    for i in range(1, epoches + 1):
        images.append(imageio.imread(os.path.join(save_dir, 'fake_samples_{}.png'.format(i))))
    imageio.mimsave(os.path.join(save_dir, 'synsethesis.gif'), images, fps=5)
    images = []
    for i in range(1, epoches + 1):
        images.append(imageio.imread(os.path.join(save_dir, str(gan_name_prefix + '_loss_epoch_{}.png').format(i))))
    imageio.mimsave(os.path.join(save_dir, 'loss.gif'), images, fps=5)

