import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.activation import \
    CELU,ELU,GELU,GLU,LeakyReLU,LogSigmoid,LogSoftmax,\
    MultiheadAttention,PReLU,ReLU,RReLU,\
    Tanh,Tanhshrink,Threshold,\
    SELU,Sigmoid,Softmax,Softmax2d,Softmin,Softplus,Softshrink,Softsign,\
    Hardshrink,Hardsigmoid,Hardswish,Hardtanh
from torch.nn.modules.activation import SiLU as TSiLU
from torch.nn.modules.activation import ReLU6 as TReLU6



# Activation functions below -------------------------------------------------------------------------------------------
class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x * torch.sigmoid(x)

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        sx = torch.sigmoid(x)  # sigmoid(ctx)
        return grad_output * (sx * (1 + x * (1 - sx)))
class MishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.mul(torch.tanh(F.softplus(x)))  # x * tanh(ln(1 + exp(x)))

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        sx = torch.sigmoid(x)
        fx = F.softplus(x).tanh()
        return grad_output * (fx + x * sx * (1 - fx * fx))

class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)
class MemoryEfficientMish(nn.Module):
    def forward(self, x):
        return MishImplementation.apply(x)

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)
class Mish(nn.Module):  # https://github.com/digantamisra98/Mish
    def forward(self, x):
        return x * F.softplus(x).tanh()
class HardSwish(nn.Module):
    def __init__(self):
        super(HardSwish, self).__init__()

    def forward(self, x):
        return x * (F.relu6(x + 3.0, inplace=True) / 6.0)
class HardSigmoid(nn.Module):
    def __init__(self):
        super(HardSigmoid, self).__init__()

    def forward(self, x):
        out = F.relu6(x + 3.0, inplace=True) / 6.0
        return out

class ReLU6(nn.Module):
    def __init__(self):
        super(ReLU6, self).__init__()

    def forward(self, x):
        return F.relu6(x, inplace=True)
# FReLU https://arxiv.org/abs/2007.11824 -------------------------------------------------------------------------------
class FReLU(nn.Module):
    def __init__(self, c1, k=3):  # ch_in, kernel
        super().__init__()
        self.conv = nn.Conv2d(c1, c1, k, 1, 1, groups=c1, bias=False)
        self.bn = nn.BatchNorm2d(c1)

    def forward(self, x):
        return torch.max(x, self.bn(self.conv(x)))

# SiLU https://arxiv.org/pdf/1606.08415.pdf ----------------------------------------------------------------------------
class SiLU(nn.Module):  # export-friendly version of nn.SiLU()
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)