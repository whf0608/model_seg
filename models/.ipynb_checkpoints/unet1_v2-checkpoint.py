import torch
from torch import nn
from modules.blocks.blocks import conv3x3,conv1x1,residualBlock

class Model(nn.Module):
    def __init__(self, in_channels, out_channels, blocks=[], repeats=[], downs=[]):
        super(Model, self).__init__()
        self.in_channels = in_channels

        self.out_channels = in_channels

        self.model = nn.Sequential()
        for i, (block, repeat,down) in enumerate(zip(blocks,repeats,downs)):
            for _ in range(repeat):
                self.in_channels = self.out_channels
                self.out_channels= self.out_channels*down

                self.model.add_module('conv'+str(i)+'_'+str(_),block(self.in_channels, self.out_channels,stride=down))

    def forward(self, x):
        x = self.model(x)
        return x


def get_model(class_n=3):
    model = Model(class_n, 64, blocks=[conv3x3,residualBlock,conv3x3,residualBlock,conv3x3,residualBlock,conv3x3], repeats=[1,3,1,3,1,3,1],downs=[1,1,2,1,1,1,1])
    return model
