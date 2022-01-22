import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from utils import var_or_cuda
import open3d as o3d
from torchvision import models

class _Encoder(torch.nn.Module):
    def __init__(self, args):
        super(_Encoder, self).__init__()
        self.args = args

        self.features = models.resnet18(pretrained=True)
        self.features.fc = nn.Sequential()
        self.fc = nn.Linear(512, 256)

    def forward(self, x):
        net = self.features(x)
        out = self.fc(net)
        return out


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

        # Create a sphere
        mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
        sphere = mesh_sphere.sample_points_uniformly(number_of_points=self.initial_points_num).points
        sphere = np.array(sphere, dtype=np.float32)
        sphere = np.expand_dims(sphere, axis=0)
        sphere = np.repeat(sphere, self.batch_size, axis=0)
        self.sphere = var_or_cuda(torch.tensor(sphere, dtype=torch.float32))

    def forward(self, x):
        x = self.Encoder(x)
        x = x.view(-1, 1, 256)
        x = x.repeat([1, self.initial_points_num, 1])
        #print(x.shape) # torch.Size([-1, 64, 256])
        #print(self.sphere.shape) # torch.Size([-1, 256, 3])

        if(x.shape[0]==1):
            x = torch.cat((x, self.sphere[0].view(1, self.initial_points_num, 3)), dim=2)
        else:
            x = torch.cat((x, self.sphere), dim=2)

        # print(x.shape) # torch.Size([-1, 256, 259])
        x = x.view(-1, self.initial_points_num * 259)
        x = self.Decoder(x)

        return x

