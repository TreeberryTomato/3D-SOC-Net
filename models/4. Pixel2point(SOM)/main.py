from train import train
import argparse
import test
import numpy as np
import torch
import random

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def setup_seed(seed):
   torch.manual_seed(seed)
   torch.cuda.manual_seed_all(seed)
   np.random.seed(seed)
   random.seed(seed)
   torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Model Parmeters
    parser.add_argument('--n_epochs', type=float, default=300,
                        help='max epochs')
    parser.add_argument('--batch_size', type=float, default=128,
                        help='each batch size')
    parser.add_argument('--initial_points_num', type=int, default=256,
                        help='initial points number')
    parser.add_argument('--gen_points', type=int, default=1024,
                        help='generated points number')
    parser.add_argument('--g_lr', type=float, default=5e-5,
                        help='generator learning rate')
    parser.add_argument('--beta', type=tuple, default=(0.5, 0.5),
                        help='beta for adam')

    # dir parameters
    parser.add_argument('--output_dir', type=str, default="./output",
                        help='output path')
    parser.add_argument('--pickle_dir', type=str, default='/best/',
                        help='input path')
    parser.add_argument('--data_dir', type=str, default='../../dataset_split/',
                        help='dataset load path')

    # step parameter
    parser.add_argument('--min_loss', type=float, default=1e10)

    args = parser.parse_args()

    # 设置随机数种子
    setup_seed(1)

    if 0:
        train(args)
    else:
        test.test1(args)
        test.test2(args)


