import torch
from torch import nn
from torchvision import models
from torch.utils.data import Dataset, DataLoader

import os
from os.path import join

import time
import copy
import random
from tqdm import tqdm
import numpy as np
from scipy import stats

# import sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock1x1(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock1x1, self).__init__()
        norm_layer = nn.BatchNorm2d
        self.conv1 = conv1x1(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv1x1(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class PartGroupNet (nn.Module):
    def __init__ (self, num_parts=3, num_chnals=256):
        super(PartGroupNet, self).__init__()
        self.num_parts = num_parts
        self.num_chnals = num_chnals

        self.part_centers = nn.Parameter(torch.zeros(num_parts, self.num_chnals))
        self.part_smooth_factors = nn.Parameter(torch.ones(num_parts))

        # self.featmap_norm = nn.InstanceNorm1d(self.num_chnals, affine=True)
        # self.output_norm = nn.InstanceNorm1d(self.num_chnals, affine=True)

    def init_parameters (self, init_centers=None, init_smooth_factors=None):
        if init_centers is None:
            nn.init.kaiming_normal_(self.part_centers)
            self.part_centers.data.clamp_(min=1e-5)
        else:
            # init weight based on clustering
            assert init_centers.shape == (self.num_parts, self.num_chnals)
            with torch.no_grad():
                self.part_centers.copy_(init_centers.unsqueeze(2).unsqueeze(3))

        # set smooth factor to 0 (before sigmoid)
        if init_smooth_factors is None:
            nn.init.constant_(self.part_smooth_factors, 0)
        else:
            # init smooth factor based on clustering 
            assert init_smooth_factors.shape == (self.num_parts,)
            with torch.no_grad():
                self.part_smooth_factors.copy_(init_smooth_factors)

    def forward (self, featmaps):
        # featmaps: B x C x H x W
        k = self.num_parts
        bs, ch, h, w = featmaps.shape
        assert self.num_chnals == ch 

        # 1. generate the grouping centers (c) and featmaps (x)
        c = self.part_centers.unsqueeze(0).expand(bs, k, ch)    # B x K x 256
        x = featmaps.view(bs, ch, -1).contiguous()    # B x 256 x 49

        # x = self.featmap_norm(x)    # B x 256 x 49
        # x = nn.functional.normalize(x, dim=1)   # B x 256 x 49

        # 2. compute assignment matrix
        # - d = -\|X - C\|_2 = - X^2 - C^2 + 2 * C^T X
        # C^T x X
        cx_ = torch.bmm(c, x) # B x K x 49
        cx = cx_.contiguous().view(bs, k, h, w)
        # X^2
        x_sq = x.pow(2).sum(dim=1, keepdim=True) # B x 1x49
        x_sq = x_sq.expand(-1, k, -1).view(bs, k, h, w)  # B x K x 7x7
        # C^2
        c_sq = c.pow(2).sum(2).unsqueeze(2).unsqueeze(3)  # B x K x 1x1
        c_sq = c_sq.expand(-1, -1, h, w)  # B x K x 7x7
        # Expand the smooth term
        beta = torch.sigmoid(self.part_smooth_factors)
        beta_batch = beta.unsqueeze(0).unsqueeze(2).unsqueeze(3)    # 1xKx1x1
        beta_batch = beta_batch.expand(bs, -1, h, w)     # B x 1 x 7x7
        assign = (2 * cx - x_sq - c_sq).clamp(max=0.0) / beta_batch # B x K x 7x7
        assign = nn.functional.softmax(assign, dim=1) # B x K x 7x7

        # 3. compute residual coding
        x = x.permute(0, 2, 1)  #B x 49 x 256
        assign = assign.view(bs, k, -1).contiguous()  #B x K x 49
        qx = torch.bmm(assign, x)   # B x K x 256

        sum_ass = torch.sum(assign, dim=2, keepdim=True) # B x K x 1
        sum_ass = sum_ass.expand(-1, -1, ch).clamp(min=1e-5) # B x K x 256
        qx_ = qx / sum_ass

        sigma = (beta / 2).sqrt()   # K
        out = (qx_ - c) / sigma.unsqueeze(0).unsqueeze(2)    # B x K x 256

        # 4. prepare outputs
        outputs = nn.functional.normalize(out, dim=2)    # B x K x C
        outputs = outputs.permute(0, 2, 1)    # B x C x K

        # outputs = self.output_norm(out.transpose(1,2)).transpose(1,2)   # B x K x C
        # outputs = nn.functional.relu(outputs)
        # outputs = outputs.permute(0, 2, 1)    # B x C x K

        # outputs = qx_.transpose(1,2).contiguous()   # B x C x K

        # for kidx in range(k):
        #     print(f'{kidx}: Sigma, {sigma[kidx]}')
        #     print(f'{kidx}: Beta, {beta[kidx]}')
        #     print(f'{kidx}: Assign, {assign[0,kidx].min()}, {assign[0,kidx].max()}, {assign[0,kidx].sum()}')
        #     print(f'{kidx}: QX, {qx_[0,kidx].min()}, {qx_[0,kidx].max()}, {qx_[0,kidx].norm(p=2, dim=0)}')
        #     print(f'{kidx}: C, {c[0,kidx].min()}, {c[0,kidx].max()}, {c[0,kidx].norm(p=2, dim=0)}')
        #     print(f'{kidx}: Out, {out[0,kidx].min()}, {out[0,kidx].max()}, {out[0,kidx].norm(p=2, dim=0)}')
        #     print(f'{kidx}: Output, {outputs[0,:,kidx].min()}, {outputs[0,:,kidx].max()}, {outputs[0,:,kidx].norm(p=2, dim=0)}')
        #     print()

        assign = assign.contiguous().view(bs, k, h, w)    # B x K x 7x7

        return outputs, assign

