import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import models

from utils import var_or_cuda


class _Encoder(torch.nn.Module):
    def __init__(self, args):
        super(_Encoder, self).__init__()
        self.args = args

        self.features = models.resnet18(pretrained=True)
        self.features.fc = nn.Sequential()
        self.fc = nn.Linear(512, 128)

    def forward(self, x):
        net = self.features(x)
        out = self.fc(net)
        return out


class _Decoder(torch.nn.Module):
    def __init__(self, args):
        super(_Decoder, self).__init__()
        self.args = args

        self.layer = torch.nn.Sequential(
            torch.nn.Linear(128, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, self.args.gen_points * 3),
        )

    def forward(self, x):
        out = self.layer(x)
        out = out.view(-1, self.args.gen_points, 3)
        # print(x.shape)  # torch.Size([-1, 256, 3])
        return out


class _G(torch.nn.Module):
    def __init__(self, args):
        super(_G, self).__init__()
        self.args = args

        self.Encoder = _Encoder(args)
        self.Decoder = _Decoder(args)

        self.batch_size = args.batch_size

    def forward(self, x):
        x = self.Encoder(x)
        x = self.Decoder(x)

        return x

