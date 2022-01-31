"""
MIT License

Copyright (c) 2022 SLAMcore

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from .blocks import *

def intialize(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.Conv3d):
            n = m.kernel_size[0] * m.kernel_size[1]*m.kernel_size[2] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.bias.data.zero_()
    return model

class SiameseTower(nn.Module):
    def __init__(self, scale_factor):
        super(SiameseTower, self).__init__()

        self.conv1 = conv_block(nc_in=3, nc_out=32, k=3, s=1, norm=None, act=None)
        res_blocks = [ResBlock(32, 32, 3, 1, 1)] * 3
        self.res_blocks = nn.Sequential(*res_blocks)    
        convblocks = [conv_block(32, 32, k=3, s=2, norm='bn', act='lrelu')] * int(math.log2(scale_factor))
        self.conv_blocks = nn.Sequential(*convblocks)
        self.conv2 = conv_block(nc_in=32, nc_out=32, k=3, s=1, norm=None, act=None)
    
    def forward(self, x):

        out = self.conv1(x)
        out = self.res_blocks(out)
        out = self.conv_blocks(out)
        out = self.conv2(out)

        return out

class CoarseNet(nn.Module):
    def __init__(self, maxdisp, scale_factor, img_shape):
        super(CoarseNet, self).__init__()
        self.maxdisp = maxdisp
        self.scale_factor = scale_factor
        self.img_shape = img_shape

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.conv3d_1 = conv3d_block(64, 32, 3, 1, norm='bn', act='lrelu')
        self.conv3d_2 = conv3d_block(32, 32, 3, 1, norm='bn', act='lrelu')
        self.conv3d_3 = conv3d_block(32, 32, 3, 1, norm='bn', act='lrelu')
        self.conv3d_4 = conv3d_block(32, 32, 3, 1, norm='bn', act='lrelu')

        self.conv3d_5 = conv3d_block(32, 1, 3, 1, norm=None, act=None)
        self.disp_reg = DisparityRegression(self.maxdisp)

    def costVolume(self, refimg_fea, targetimg_fea, views):
        #Cost Volume
        cost = torch.zeros(refimg_fea.size()[0], refimg_fea.size()[1]*2, self.maxdisp//self.scale_factor, refimg_fea.size()[2], refimg_fea.size()[3]).cuda()
        views = views.lower()
        if views == 'left':
            for i in range(self.maxdisp//self.scale_factor):
                if i > 0:
                    cost[:, :refimg_fea.size()[1], i, :, i:] = refimg_fea[:,:,:,i:]
                    cost[:, refimg_fea.size()[1]:, i, :, i:] = targetimg_fea[:,:,:,:-i]
                else:
                    cost[:, :refimg_fea.size()[1], i, :,:] = refimg_fea
                    cost[:, refimg_fea.size()[1]:, i, :,:] = targetimg_fea
        elif views == 'right':
            for i in range(self.maxdisp // self.scale_factor):
                if i > 0:
                    cost[:, :refimg_fea.size()[1], i, :, :-i] = refimg_fea[:,:,:,i:]
                    cost[:, refimg_fea.size()[1]:, i, :, :-i] = targetimg_fea[:,:,:,:-i]
                else:
                    cost[:, :refimg_fea.size()[1], i, :,:] = refimg_fea
                    cost[:, refimg_fea.size()[1]:, i, :,:] = targetimg_fea
        return cost

    def Coarsepred(self, cost):
        cost = self.conv3d_1(cost)
        cost = self.conv3d_2(cost) + cost
        cost = self.conv3d_3(cost) + cost
        cost = self.conv3d_4(cost) + cost
        
        cost = self.conv3d_5(cost)
        cost = F.interpolate(cost, size=[self.maxdisp, self.img_shape[1], self.img_shape[0]], mode='trilinear', align_corners=False)
        pred = cost.softmax(dim=2).squeeze(dim=1)
        pred = self.disp_reg(pred)

        return pred

    def forward(self, refimg_fea, targetimg_fea, do_right=True):
        '''
        Args:
            refimg_fea: output of SiameseTower for a left image
            targetimg_fea: output of SiameseTower for the right image

        '''
        cost_left = self.costVolume(refimg_fea, targetimg_fea, 'left')
        pred_left = self.Coarsepred(cost_left)

        if do_right:
            cost_right = self.costVolume(refimg_fea, targetimg_fea, 'right')
            pred_right = self.Coarsepred(cost_right)

            return pred_left, pred_right
        else:
            return pred_left, pred_left
        
class RefineNet(nn.Module):
    def __init__(self):
        super(RefineNet, self).__init__()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # stream_1, left_img
        self.conv1_s1 = conv_block(3, 16, 3, 1)
        self.resblock1_s1 = ResBlock(16, 16, 3, 1, 1)
        self.resblock2_s1 = ResBlock(16, 16, 3, 1, 2)

        # stream_2, upsampled low_resolution disp
        self.conv1_s2 = conv_block(1, 16, 1, 1)
        self.resblock1_s2 = ResBlock(16, 16, 3, 1, 1)
        self.resblock2_s2 = ResBlock(16, 16, 3, 1, 2)

        # cat
        self.resblock3 = ResBlock(32, 32, 3, 1, 4)
        self.resblock4 = ResBlock(32, 32, 3, 1, 8)
        self.resblock5 = ResBlock(32, 32, 3, 1, 1)
        self.resblock6 = ResBlock(32, 32, 3, 1, 1)
        self.conv2 = conv_block(32, 1, 3, 1)

    def forward(self, left_img, up_disp):
        
        stream1 = self.conv1_s1(left_img)
        stream1 = self.resblock1_s1(stream1)
        stream1 = self.resblock2_s1(stream1)

        stream2 = self.conv1_s2(up_disp)
        stream2 = self.resblock1_s2(stream2)
        stream2 = self.resblock2_s2(stream2)

        out = torch.cat((stream1, stream2), 1)
        out = self.resblock3(out)
        out = self.resblock4(out)
        out = self.resblock5(out)
        out = self.resblock6(out)
        out = self.conv2(out)

        return out     


class ActiveStereoNet(nn.Module):
    def __init__(self, maxdisp, scale_factor, img_shape):
        super(ActiveStereoNet, self).__init__()
        self.maxdisp = maxdisp
        self.scale_factor = scale_factor
        self.SiameseTower = intialize(SiameseTower(scale_factor))
        self.CoarseNet = intialize(CoarseNet(maxdisp, scale_factor, img_shape))
        self.RefineNet = intialize(RefineNet())
        self.img_shpae = img_shape
      
    def forward(self, left, right, disp_left, do_right=True):
        
        flip = (np.random.rand() > 0.5)
        if flip:
            left = torch.flip(left, dims=(3,))
            right = torch.flip(right, dims=(3,))
        left_tower = self.SiameseTower(left)
        right_tower = self.SiameseTower(right)
        if flip:
            left = torch.flip(left, dims=(3,))
            right = torch.flip(right, dims=(3,))
            left_tower = torch.flip(left_tower, dims=(3,))
            right_tower = torch.flip(right_tower, dims=(3,))
        coarseup_pred_left, coarseup_pred_right = self.CoarseNet(left_tower, right_tower, do_right)

        res_disp_left = self.RefineNet(left, coarseup_pred_left)
        ref_pred_left = nn.ReLU(False)(coarseup_pred_left + res_disp_left)
        coarseup_pred_left = nn.ReLU(False)(coarseup_pred_left)

        if do_right:
            res_disp_right = self.RefineNet(right, coarseup_pred_right)
            ref_pred_right = nn.ReLU(False)(coarseup_pred_right + res_disp_right)
            coarseup_pred_right = nn.ReLU(False)(coarseup_pred_right)
            return ref_pred_left, coarseup_pred_left, ref_pred_right, coarseup_pred_right
        else:
            return ref_pred_left, coarseup_pred_left, ref_pred_left, coarseup_pred_left

