import torch
import torch.nn as nn
import torch.nn.functional as F
from compressai.layers import subpel_conv3x3, AttentionBlock
from modules.layers.conv import conv1x1, conv3x3, conv, deconv
from modules.layers.res_blk import *


class HyperSynthesis(nn.Module):
    """
    Local Reference
    """
    def __init__(self, M=192, N=192) -> None:
        super().__init__()
        self.M = M
        self.N = N

        self.increase = nn.Sequential(
            conv3x3(N, M),
            nn.GELU(),
            subpel_conv3x3(M, M, 2),
            nn.GELU(),
            conv3x3(M, M * 3 // 2),
            nn.GELU(),
            subpel_conv3x3(M * 3 // 2, M * 3 // 2, 2),
            nn.GELU(),
            conv3x3(M * 3 // 2, M * 2),
        )

    def forward(self, x):
        x = self.increase(x)

        return x


class SynthesisTransformOld(nn.Module):
    def __init__(self, N, M):
        super().__init__()
        N = N[::-1]
        self.N = N
        layers = []
        for k in range(len(N)):
            if k == 0:
                layers.append(ResidualBlock(M, N[k]))
                layers.append(ResidualBlockUpsample(N[k], N[k], 2))
            else:
                layers.append(ResidualBlock(N[k-1], N[k]))
                layers.append(ResidualBlockUpsample(N[k], N[k], 2))
        layers.append(ResidualBlock(N[-1], N[-1]))
        layers.append(subpel_conv3x3(N[-1], 3, 2))
        
        self.synthesis_transform = nn.Sequential(*layers)

    def forward(self, x):
        x = self.synthesis_transform(x)

        return x


class SynthesisTransform(nn.Module):
    def __init__(self, N, M):
        super().__init__()
        layers = []
        N = N[::-1]
        for k in range(len(N)):
            if k == 0:
                layers.append(ResidualBlock(M, M))
                layers.append(ResidualBlockUpsample(M, N[k], 2))
            else:
                layers.append(ResidualBlock(N[k-1], N[k]))
                layers.append(ResidualBlockUpsample(N[k], N[k], 2))
        layers.append(ResidualBlock(N[-1], N[-1]))
        layers.append(subpel_conv3x3(N[-1], 3, 2))

        self.synthesis_transform = nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.synthesis_transform(x)

        return x
