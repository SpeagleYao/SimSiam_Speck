# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(planes)
        self.shortcut = nn.Sequential()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = out + self.shortcut(x)
        
        return out

class SimSiam_My(nn.Module):
    """
    Build a SimSiam model.
    """
    def __init__(self):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super(SimSiam_My, self).__init__()

        # create the encoder
        # num_classes is the output fc dimension, zero-initialize last BNs
        # self.encoder = base_encoder(num_classes=dim, zero_init_residual=True)

        self.encoder = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            BasicBlock(32, 32),
            # nn.Linear(64, 64, bias=False),
            # nn.BatchNorm1d(64),
            # nn.ReLU(inplace=True), # first layer
            nn.Flatten(),
            nn.Linear(512, 64, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True), # second layer
            nn.BatchNorm1d(64, affine=False)
        )
        # self.encoder[8].bias.requires_grad = False !!!
        # self.res = BasicBlock(64, 64)
        # # build a 3-layer projector
        # prev_dim = self.encoder.fc.weight.shape[1]
        # self.encoder.fc = nn.Sequential(nn.Conv1d(2, 64, kernel_size=1),
        #                                 nn.BatchNorm1d(64),
        #                                 nn.ReLU(inplace=True),
        #                                 nn.Conv1d(64, 64, kernel_size=3, padding=1),
        #                                 nn.BatchNorm1d(64),
        #                                 nn.ReLU(inplace=True),
        #                                 nn.BatchNorm1d(dim, affine=False))
        
        # self.encoder.fc[6].bias.requires_grad = False # hack: not use bias as it is followed by BN

        # build a 2-layer predictor
        self.predictor = nn.Sequential(nn.Linear(64, 64, bias=False),
                                        nn.BatchNorm1d(64),
                                        nn.ReLU(inplace=True), # hidden layer
                                        nn.Linear(64, 64)) # output layer

    def forward(self, x):
        """
        Input:
            x1: first views of images
            x2: second views of images
        Output:
            p1, p2, z1, z2: predictors and targets of the network
            See Sec. 3 of https://arxiv.org/abs/2011.10566 for detailed notations
        """

        x = x.view(x.shape[0], 4, 16)
        x1 = x[:, 0:2, :]
        x2 = x[:, 2:4, :]
        # compute features for one view
        z1 = self.encoder(x1) # NxC
        z2 = self.encoder(x2) # NxC

        p1 = self.predictor(z1) # NxC
        p2 = self.predictor(z2) # NxC

        return p1, p2, z1.detach(), z2.detach()

def SS():
    return SimSiam_My()

if __name__=='__main__':
    net = SS()
    summary(net.cuda(), (1, 64))
    x = torch.rand(13, 64).cuda()
    y = net(x) 
    # print(y.size())