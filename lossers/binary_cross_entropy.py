  #二值交叉熵，这里输入要经过sigmoid处理
import torch
import torch.nn as nn
import torch.nn.functional as F

def Binary_Cross_entropy(input,target):
    nn.BCELoss(F.sigmoid(input), target)
    #多分类交叉熵, 用这个 loss 前面不需要加 Softmax 层
    nn.CrossEntropyLoss(input, target)


class WeightedCrossEntropyLoss(torch.nn.CrossEntropyLoss):
   """
   Network has to have NO NONLINEARITY!
   """
   def __init__(self, weight=None):
       super(WeightedCrossEntropyLoss, self).__init__()
       self.weight = weight

   def forward(self, inp, target):
       target = target.long()
       num_classes = inp.size()[1]

       i0 = 1
       i1 = 2

       while i1 < len(inp.shape): # this is ugly but torch only allows to transpose two axes at once
           inp = inp.transpose(i0, i1)
           i0 += 1
           i1 += 1

       inp = inp.contiguous()
       inp = inp.view(-1, num_classes)

       target = target.view(-1,)
       wce_loss = torch.nn.CrossEntropyLoss(weight=self.weight)

       return wce_loss(inp, target)

