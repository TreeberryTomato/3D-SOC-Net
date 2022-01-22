import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from utils import var_or_cuda
import open3d as o3d


class _Encoder(torch.nn.Module):
    def __init__(self, args):
        super(_Encoder, self).__init__()
        self.args = args

        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU()
        )

        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU()
        )

        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU()
        )

        self.layer4 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU()
        )

        self.layer5 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU()
        )

        self.layer6 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU()
        )

        self.layer7 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU()
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        # print(x.shape)  # torch.Size([-1, 256, 1, 1])
        x = x.view(-1, 256)
        return x


class _Decoder(torch.nn.Module):
    def __init__(self, args):
        super(_Decoder, self).__init__()
        self.args = args
        self.initial_points_num = args.initial_points_num

        self.layer = torch.nn.Sequential(
            torch.nn.Linear(self.initial_points_num*259, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, self.args.gen_points * 3),
        )

    def forward(self, x):
        x = self.layer(x)
        x = x.view(-1, self.args.gen_points, 3)
        # print(x.shape)  # torch.Size([-1, 256, 3])
        return x


class _G(torch.nn.Module):
    def __init__(self, args):
        super(_G, self).__init__()
        self.args = args

        self.Encoder = _Encoder(args)
        self.Decoder = _Decoder(args)

        self.batch_size = args.batch_size
        self.initial_points_num = args.initial_points_num

    def forward(self, x, shape):
        x = self.Encoder(x)
        x = x.view(-1, 1, 256)
        x = x.repeat([1, self.initial_points_num, 1])
        #print(x.shape) # torch.Size([-1, 64, 256])
        #print(self.sphere.shape) # torch.Size([-1, 256, 3])

        x = torch.cat((x, shape), dim=2)

        # print(x.shape) # torch.Size([-1, 256, 259])
        x = x.view(-1, self.initial_points_num * 259)
        x = self.Decoder(x)

        return x

