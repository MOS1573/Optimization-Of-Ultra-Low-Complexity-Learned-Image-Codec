import torch
import torch.nn as nn
import torch.nn.functional as F
from compressai.layers import subpel_conv3x3
from modules.layers.conv import conv1x1, conv3x3, conv, deconv
from modules.layers.res_blk import *


class AnalysisTransform(nn.Module):
    def __init__(self, N, M):
        super().__init__()
        layers = []
        for k in range(len(N)):
            if k == 0:
                layers.append(ResidualBlockWithStride(3, N[k], stride=2))
            else:
                layers.append(ResidualBlockWithStride(N[k-1], N[k], stride=2))
            layers.append(ResidualBlock(N[k], N[k]))

        layers.append(conv3x3(N[-1], M, stride=2))
        self.analysis_transform = nn.Sequential(*layers)

    def forward(self, x):
        x = self.analysis_transform(x)

        return x


class HyperAnalysis(nn.Module):
    """
    Local reference
    """
    def __init__(self, M=192, N=192):
        super().__init__()
        self.M = M
        self.N = N
        self.reduction = nn.Sequential(
            conv3x3(M, N),
            nn.GELU(),
            conv3x3(N, N),
            nn.GELU(),
            conv3x3(N, N, stride=2),
            nn.GELU(),
            conv3x3(N, N),
            nn.GELU(),
            conv3x3(N, N, stride=2),
        )

    def forward(self, x):
        x = self.reduction(x)

        return x
