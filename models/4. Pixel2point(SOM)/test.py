import sys

import torch
from torch import optim, nn
from utils import Dataset, var_or_cuda, read_pickle, save_new_pickle
from torch.utils.data import DataLoader
from model import _G
import utils
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
import torchvision.utils as vutils

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


models_dir = "data"

def test1(args):
    G = _G(args)

    G_solver = optim.Adam(G.parameters(), lr=args.g_lr, betas=args.beta)
    if torch.cuda.is_available():
        print("using cuda")
        G.cuda()
    G.eval()
    pickle_path = args.output_dir + args.pickle_dir

    iteration = read_pickle(pickle_path, G, G_solver)

    test_dataSet = Dataset(root=models_dir, args=args, t='sample')

    test_dataLoader = DataLoader(test_dataSet, batch_size=1, shuffle=False, num_workers=0)

    utils.save_test_img(args.output_dir + "/test/iteration_" + iteration.zfill(4), iteration, G, test_dataLoader)

def test2(args):
    args.batch_size=24
    G = _G(args)

    G_solver = optim.Adam(G.parameters(), lr=args.g_lr, betas=args.beta)
    if torch.cuda.is_available():
        print("using cuda")
        G.cuda()
    G.eval()
    pickle_path = args.output_dir + args.pickle_dir

    iteration = read_pickle(pickle_path, G, G_solver)

    test_dataSet = Dataset(root=models_dir, args=args, t="test")

    test_dataLoader = DataLoader(test_dataSet, batch_size=24, shuffle=False, num_workers=0)

    utils.save_test_result(args.output_dir + "/test/", G, test_dataLoader)