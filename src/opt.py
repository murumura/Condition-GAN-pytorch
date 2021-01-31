import argparse
import configargparse
from utils import boolean_string

def get_options():
    parser = configargparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='cgan', 
                        help='one of `cgan` and `infogan`.')

    parser.add_argument('--cuda', type=boolean_string, default=True, 
                        help='enable CUDA.')

    parser.add_argument('--train', type=boolean_string, default=True, 
                        help='train mode or eval mode.')

    parser.add_argument('--data_dir', type=str, default='~/Data/mnist', 
                        help='Directory for dataset.')

    parser.add_argument('--out_dir', type=str, default='output', 
                        help='Directory for output.')

    parser.add_argument('--epochs', type=int, default=200, 
                        help='number of epochs')

    parser.add_argument('--batch_size', type=int, default=128, 
                        help='size of batches')

    parser.add_argument('--lr', type=float, default=0.0002, 
                        help='learning rate')

    parser.add_argument('--latent_dim', type=int, default=100, 
                        help='latent space dimension')

    parser.add_argument('--classes', type=int, default=10, 
                        help='number of classes')

    parser.add_argument('--img_size', type=int, default=64, 
                        help='size of images')

    parser.add_argument('--channels', type=int, default=1, 
                        help='number of image channels')

    parser.add_argument('--log_interval', type=int, default=100, 
                        help='interval between logging and image sampling')

    parser.add_argument('--seed', type=int, default=1, 
                        help='random seed')

    return parser