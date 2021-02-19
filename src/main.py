import glob
import os
import sys
import time
import logging
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm import tqdm, trange
from opt import get_options
from utils import clear_and_create_folder
from build_model import ConditionModel
def train(args):
    # Create log dir and copy the config file into the log file
    log_dir = os.path.join(args.log_dir, args.exp_name)
    clear_and_create_folder(log_dir)

    # write cinfiguration arguments into the log file
    log_args_copy = os.path.join(log_dir, 'args.txt')

    with open(log_args_copy, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    logging.info(f'Copy configuration arguments to {log_args_copy}')

    if args.config is not None:
        log_config_copy = os.path.join(log_dir, 'config.txt')
        with open(log_config_copy, 'w') as file:
            file.write(open(args.config, 'r').read())
        logging.info(f'Copy configuration file to {log_config_copy}')

    # Create output dir
    out_dir = os.path.join(args.out_dir, args.exp_name)
    clear_and_create_folder(out_dir)

    # Prepare dataset and data_loader
    if args.dataset_type == 'mnist':
        dataset = datasets.MNIST(root=args.data_dir, download=True,
                             transform=transforms.Compose([
                             transforms.Resize(args.img_size),
                             transforms.ToTensor(),
                             transforms.Normalize((0.5,), (0.5,))
                             ]))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                                 shuffle=True, num_workers=4, pin_memory=True)

    assert dataset,f'Unsuccessfully loaded {args.dataset_type} dataset!'

    logging.info(f'''
        dataset: {args.dataset_type}
        all images shape: {args.img_size} x {args.img_size}
        data directory: {args.data_dir}
    ''')

    # Create gan model
    model = ConditionModel(
        name=args.model, 
        device=device, 
        data_loader=dataloader, 
        classes=args.classes, 
        channels=args.channels, 
        img_size=args.img_size, 
        latent_dim=args.latent_dim
    )
    logging.info(f'''
    Model: {args.model}\n
    Model arch:\n {model}
    ''')
    # Specify the optimizer of the model
    model.create_optim(args.lr)
    # Start Training
    logging.info(f'''
        PyTorch version: {torch.__version__}
        CUDA version: {torch.version.cuda}
    ''')
    model.train(
        args.epochs, 
        args.log_interval, 
        out_dir, 
        verbose=True, 
        save_checkpoints=True
    )
def evaluation(args):
    # Create output dir for evaluation
    eval_out_dir = os.path.join(args.eval_dir, args.exp_name)
    clear_and_create_folder(eval_out_dir)
    # Create gan model
    model = ConditionModel(
        name=args.model, 
        device=device, 
        data_loader=None, 
        classes=args.classes, 
        channels=args.channels, 
        img_size=args.img_size, 
        latent_dim=args.latent_dim
    )
    # load model state
    model.load_state_from(args.state_dir)
    model.eval(
        mode=1, 
        batch_size=args.batch_size,
        output_dir=eval_out_dir
    )

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s') 
    args = get_options().parse_args()

    if args.cuda:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logging.info(f'Using device {device}')
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if args.cuda:
            torch.cuda.manual_seed(args.seed)
        np.random.seed(args.seed)
    if args.train:
        train(args)
    elif args.eval:
        evaluation(args)