#PyTorch lib
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torch.nn.functional as F
import torchvision
#Tools lib
import numpy as np
import cv2
import random
import time
import os


def print_network(net):
	num_params = 0
	for param in net.parameters():
		num_params += param.numel()
	print(net)
	print('Total number of parameters: %d' % num_params)



class DRN(nn.Module):
    def __init__(self, channel=3, inter_iter=7, intra_iter=7, use_GPU=True):
        super(DRN, self).__init__()
        self.iteration = inter_iter
        self.intra_iter = intra_iter
        channel_feature = 16

        self.use_GPU = use_GPU

        self.conv0 = nn.Sequential(
            nn.Conv2d(channel*2, channel_feature, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv1 = nn.Sequential(
            nn.Conv2d(channel_feature, channel_feature, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(channel_feature, channel_feature, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv2 = nn.Sequential(
            nn.Conv2d(channel_feature, channel_feature, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(channel_feature, channel_feature, 3, 1, 1),
            nn.ReLU()
        )

        self.conv = nn.Sequential(
            nn.Conv2d(channel_feature, channel, 3, 1, 1),
            )


    def forward(self, input):
        x = input

        x_list = []
        for i in range(self.iteration):
            x = torch.cat((input, x), 1)
            x = self.conv0(x)
            resx = x
            for j in range(self.intra_iter):
                x = F.relu(self.res_conv1(x) + resx)
                resx = x
                x = F.relu(self.res_conv2(x) + resx)
                resx = x


            x = self.conv(x)
            x = input + x
            x_list.append(x)

        return x, x_list