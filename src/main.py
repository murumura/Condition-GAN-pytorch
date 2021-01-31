import glob
import os
import time
import logging
import sys
import numpy as np
import torch
import torchvision
import yaml
from tqdm import tqdm, trange
from opt import get_options


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    
    args = get_options().parse_args()
    