import torch
from torch import optim, nn
from torch.autograd import Variable

from utils import Dataset, chamfer_distance_with_batch,var_or_cuda, read_pickle, save_new_pickle, save_test_result, validate
from torch.utils.data import DataLoader
from model import _G
import utils
import logging
from logging import handlers
import time

import threading
import GPUtil
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import numpy as np
from geomloss import SamplesLoss

from tqdm import tqdm


def train(args):
        '''
        voxel = utils.getVoxelFromMat("data/model/IKEA_chair_BERNHARD_bernhard_chair_obj0_object.mat", cube_len=args.cube_len)
        utils.plotFromVoxels(voxel)
        return
        '''
        start_time = time.time()

        logger = logging.getLogger("Train")
        logger.setLevel(logging.INFO)

        streamHandler = logging.StreamHandler()
        streamHandler.setLevel(logging.INFO)

        fileHandler = logging.FileHandler('train.log', mode="w", encoding='utf-8')
        fileHandler.setLevel(logging.INFO)

        logger.addHandler(streamHandler)
        logger.addHandler(fileHandler)

        dataset = Dataset(root=args.data_dir, args=args, t='train')

        dataLoader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0, drop_last=True,
                                      shuffle=True, pin_memory=False)


        G = _G(args=args)

        G_solver = optim.Adam(G.parameters(), lr=args.g_lr, betas=args.beta)

        emd_loss_function = SamplesLoss()

        if torch.cuda.is_available():
            print("using cuda")
            G = G.cuda()


        for epoch in tqdm(range(args.n_epochs)):
            epoch_start_time = time.time()

            for i, data in enumerate(dataLoader, 0):
                # data: [[image, model]]
                images, gt, mean_shape = data[0], data[1], data[2]

                #if len(images)<args.batch_size:
                    #break

                images = var_or_cuda(images)
                gt = var_or_cuda(gt)

                # =============== Train the generator ===============#
                G.train()
                fake = G(images)
                #g_loss = torch.add(chamfer_distance_with_batch(fake, gt, type='mean'), chamfer_distance_with_batch(gt, fake, type='mean'))
                g_loss = torch.mean(emd_loss_function(fake, gt))

                G.zero_grad()
                g_loss.backward()
                G_solver.step()

                logger.info("Epoch(%d, %d), Batch(%d, %d), Loss: %f" % (epoch+1, args.n_epochs, i+1, len(dataLoader), g_loss.item()))

            #print(G_solver.state_dict()['param_groups'][0]['params'][0])

            #print(G_solver.state_dict()['state'])

            #iteration = str(
                #G_solver.state_dict()['state'][G_solver.state_dict()['param_groups'][0]['params'][0]])
            batch_num = len(dataLoader)

            epoch_end_time = time.time()
            logger.info("Epoch(%d) takes %.2f seconds",epoch+1,round(epoch_end_time-epoch_start_time, 2))
            iteration = str(epoch+1)

            G.eval()
            dataset.switchToVal()
            validate(args, iteration, G, dataLoader, G_solver)
            dataset.switchToTrain()

        end_time = time.time()
        logger.info("The training takes %.f seconds", round(end_time-start_time, 0))


