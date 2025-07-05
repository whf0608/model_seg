from torch import nn
import torch

class DUC(nn.Module):
    def __init__(self, in_channels, out_channles, upscale):
        super(DUC, self).__init__()
        out_channles = out_channles * (upscale ** 2)
        self.conv = nn.Conv2d(in_channels, out_channles, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channles)
        self.relu = nn.ReLU(inplace=True)
        self.pixl_shf = nn.PixelShuffle(upscale_factor=upscale)

        # initialize_weights(self)
        kernel = self.icnr(self.conv.weight, scale=upscale)
        self.conv.weight.data.copy_(kernel)

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        x = self.pixl_shf(x)
        return x

    def icnr(self, x, scale=2, init=nn.init.kaiming_normal):
        '''
        Even with pixel shuffle we still have check board artifacts,
        the solution is to initialize the d**2 feature maps with the same
        radom weights: https://arxiv.org/pdf/1707.02937.pdf
        '''
        new_shape = [int(x.shape[0] / (scale ** 2))] + list(x.shape[1:])
        subkernel = torch.zeros(new_shape)
        subkernel = init(subkernel)
        subkernel = subkernel.transpose(0, 1)
        subkernel = subkernel.contiguous().view(subkernel.shape[0],
                                                subkernel.shape[1], -1)
        kernel = subkernel.repeat(1, 1, scale ** 2)
        transposed_shape = [x.shape[1]] + [x.shape[0]] + list(x.shape[2:])
        kernel = kernel.contiguous().view(transposed_shape)
        kernel = kernel.transpose(0, 1)
        return kernel