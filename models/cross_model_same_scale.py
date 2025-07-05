from init import *
import numpy as np
from torch import nn
import torch
from modules.blocks.yolo_blocks import C3, Conv

class Fusion2Backbone(nn.Module):
    def __init__(self, c1, ns=[], ss=[], c_size=True):
        super(Fusion2Backbone, self).__init__()

        self.seq = nn.Sequential()
        self.seq1 = nn.Sequential()
        self.seq2 = nn.Sequential()
        c = c1
        for i, (n, s) in enumerate(zip(ns, ss)):
            subseq = nn.Sequential()
            for _ in range(n):
                subseq.add_module('c3_' + str(i) + '_' + str(_), C3(c1, c, 1))

            if not c_size: c *= s
            self.seq.add_module('subseq_' + str(i), subseq)
            self.seq1.add_module('c1_' + str(i), Conv(c1, c, k=3, s=s, p=1))
            c1 = c

            self.seq2.add_module('c1_' + str(i), Conv(c1 * 2, c1, k=3, s=1, p=1))
        self.cov_act1 = Cov_Act(c1, c, 3, 1, 1)
        self.cov_act2 = Cov_Act(c1, c, 3, 1, 1)

    def forward(self, x1, x2):
        for m, m1, m2 in zip(self.seq, self.seq1, self.seq2):
            x1 = m(x1)
            y1 = m1(x1)

            x2 = m(x2)
            y2 = m1(x2)
            x1 = torch.cat([y1 * 0.8, y2 * 0.2], 1)
            x2 = torch.cat([y2 * 0.8, y1 * 0.2], 1)

            x1 = m2(x1)
            x2 = m2(x2)
            x1 = self.cov_act1(x1)
            x2 = self.cov_act2(x2)
        return x1, x2


class Cov_Act(nn.Module):
    def __init__(self, c1, c2, k, s, p):
        super(Cov_Act, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=1, bias=False)
        # self.bn = nn.BatchNorm2d(c2,eps=0.001,momentum=0.03)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)
        return x

def train_model():
    cfg['training']['batch_size'] = 16

    fb = Fusion2Backbone(3, ns=[6, 1, 1, 6, 1, 1], ss=[1, 1, 1, 1, 1, 1])
    n_claess, train_loader, val_loder = get_dataloader(cfg)
    optimizer, scheduler = get_optimizer_scheduler(fb)
    device = 'cuda:5'
    epochs = 500
    fb = fb.train()
    fb = fb.to(device).half()

    for epoch in range(epochs):
        losses = []
        for images in train_loader:
            scheduler.step()
            optimizer.zero_grad()
            t1 = images['image_mask'].to(device).half()
            t2 = images['image_mask'].to(device).half()
            t1_b = images['image'].to(device).half()
            t2_b = images['image'].to(device).half()

            r1, r2 = fb(t1, t2)
            loss = (torch.abs(r1 - t1_b).double().sum() / r1.shape.numel() + torch.abs(
                r2 - t2_b).double().sum() / r2.shape.numel()) / 1.0
            loss.backward()
            optimizer.step()

            losses.append(loss.detach().cpu().numpy())
        print('epoch' + str(epoch) + ' :', np.sum(losses) / len(losses))
