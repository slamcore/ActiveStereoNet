import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def resample(grid, dispmap_view, img_to_resample, disp_to_resample, view_type='left'):
    dispmap_view_norm = dispmap_view * 2 / img_to_resample.shape[-1]
    dispmap_view_norm = dispmap_view_norm.cuda()
    dispmap_view_norm = dispmap_view_norm.squeeze(1).unsqueeze(3)
    dispmap_view_norm = torch.cat((dispmap_view_norm, torch.zeros(dispmap_view_norm.size()).cuda()), dim=3)
    
    if view_type == 'left':    
       grid_view = grid - dispmap_view_norm
    elif view_type == 'right':    
       grid_view = grid + dispmap_view_norm        
    else:    
       raise NotImplementedError('Incorrect view type: [{:s}]'.format(view_type))
    
    recon_img = F.grid_sample(img_to_resample, grid_view)
    recon_dispmap = F.grid_sample(disp_to_resample, grid_view)

    return recon_img, recon_dispmap

class RHLoss(nn.Module):

    def __init__(self, max_disp):

        super(RHLoss, self).__init__()
        self.max_disp = max_disp
        self.crit = nn.SmoothL1Loss(reduction='mean')
    
    def forward(self, left_img, right_img, output, target, brk, weight):

        mask = target < self.max_disp
        mask.detach_()
        
        loss = self.crit(output[mask], target[mask])
        
        return weight * loss

class XTLoss(nn.Module):
    '''
    Args:
        left_img right_img: N * C * H * W,
        dispmap : N * H * W
    '''
    def __init__(self, max_disp, lcn_weight=0.5, occluded_weight=0.1):
        super(XTLoss, self).__init__()
        self.max_disp = max_disp
        self.theta = torch.Tensor(
            [[1, 0, 0], 
            [0, 1, 0]]
        )
        self.inplanes = 3
        self.outplanes = 3
        self.crossEntropy = nn.CrossEntropyLoss(reduction='mean')
        self.lcn_weight = lcn_weight
        self.occluded_weight = occluded_weight

    
    def forward(self, left_img, right_img, dispmap_left, dispmap_right, dispmap_gt, brk, weight):

        n, c, h, w = left_img.shape
        valid_gt = torch.ones(dispmap_left.shape[0], dispmap_left.shape[2], dispmap_left.shape[3]).long().cuda()
        
        theta = self.theta.repeat(left_img.size()[0], 1, 1)
                
        grid = F.affine_grid(theta, left_img.size())
        grid = grid.cuda()
        
        recon_img_right, recon_dispmap_right = resample(grid, dispmap_left, right_img, dispmap_right, view_type='left')
        recon_img_left, recon_dispmap_left = resample(grid, dispmap_right, left_img, dispmap_left, view_type='right')

        losses_left_photo = torch.abs(((left_img - recon_img_right)))
        losses_right_photo = torch.abs(((right_img - recon_img_left)))

        #""" weighted LCN 
        if self.lcn_weight > 0.0:
            recon_img_right_LCN, _, _ = self.LCN(recon_img_right, 9)
            recon_img_left_LCN, _, _ = self.LCN(recon_img_left, 9)        
            left_img_LCN, _, left_std_local = self.LCN(left_img, 9)
            right_img_LCN, _, right_std_local = self.LCN(right_img, 9)

            losses_left_LCN = torch.abs(((left_img_LCN - recon_img_right_LCN) * left_std_local))
            losses_left_LCN = self.ASW(left_img, losses_left_LCN, 12, 2)  
            losses_right_LCN = torch.abs(((right_img_LCN - recon_img_left_LCN) * right_std_local))
            losses_right_LCN = self.ASW(left_img, losses_right_LCN, 12, 2) 
        else:
            losses_left_LCN = torch.zeros_like(losses_left_photo)
            losses_right_LCN = torch.zeros_like(losses_right_photo)

        losses_left = (1 - self.lcn_weight) * losses_left_photo + self.lcn_weight * losses_left_LCN
        losses_right = (1-  self.lcn_weight) * losses_right_photo + self.lcn_weight * losses_right_LCN
        #"""

        #""" Occluded pixels
        if self.occluded_weight > 0.0:
            dispmap_left_diff = (dispmap_left - recon_dispmap_right).abs()        
            valid_pred_left_prob = torch.exp(-0.6931 * dispmap_left_diff.clamp(min=0.0)) #

            dispmap_right_diff = (dispmap_right - recon_dispmap_left).abs()        
            valid_pred_right_prob = torch.exp(-0.6931 * dispmap_right_diff.clamp(min=0.0)) #
        else:
            valid_pred_left_prob = valid_gt.unsqueeze(1).float()
            valid_pred_right_prob = valid_gt.unsqueeze(1).float()

        valid_pred_left = valid_pred_left_prob > 0.5 # same as dispmap_diff < 1
        valid_pred_left_prob = torch.cat((1.0 - valid_pred_left_prob, valid_pred_left_prob), 1)

        valid_pred_right = valid_pred_right_prob > 0.5 # same as dispmap_diff < 1
        valid_pred_right_prob = torch.cat((1.0 - valid_pred_right_prob, valid_pred_right_prob), 1)

        loss_invalid = self.crossEntropy(valid_pred_left_prob, valid_gt) + self.crossEntropy(valid_pred_right_prob, valid_gt)
        #"""    
 
        loss_valid = (losses_left[valid_pred_left.repeat(1,3,1,1)].sum() / (valid_pred_left.repeat(1,3,1,1).sum() + 1e-6)) + (losses_right[valid_pred_right.repeat(1,3,1,1)].sum() / (valid_pred_right.repeat(1,3,1,1).sum() + 1e-6))
        loss = weight * ((1 - self.occluded_weight) * loss_valid + self.occluded_weight * loss_invalid)

        assert not torch.isnan(loss)
        return loss


    def LCN(self, img, kSize):
        '''
            Args: 
                img : N * C * H * W
                kSize : 9 * 9
        '''

        w = torch.ones((self.outplanes, 1, kSize, kSize)).cuda() / (kSize * kSize)
        mean_local = F.conv2d(input=img, weight=w, padding=kSize // 2, groups = self.inplanes)
        
        mean_square_local = F.conv2d(input=img ** 2, weight=w, padding=kSize // 2, groups = self.inplanes)
        std_local = torch.sqrt(nn.ReLU(False)(mean_square_local - mean_local ** 2)) * (kSize ** 2) / (kSize ** 2 - 1)

        epsilon = 1e-5
        img_LCN = (img - mean_local) / (std_local + epsilon)
        return img_LCN, mean_local, std_local


    def ASW(self, img, Cost, kSize, sigma_omega):
        
        weightGraph = torch.zeros(img.shape, requires_grad=False).cuda()
        CostASW = torch.zeros(Cost.shape, dtype=torch.float, requires_grad=True).cuda()

        pad_len = kSize // 2
        img = F.pad(img, [pad_len] * 4)
        Cost = F.pad(Cost, [pad_len] * 4)
        n, c, h, w = img.shape
        

        
        for i in range(kSize):
            for j in range(kSize):
                tempGraph = torch.abs(img[:, :, pad_len : h - pad_len, pad_len : w - pad_len] - img[:, :, i:i + h - pad_len * 2, j:j + w - pad_len * 2])
                tempGraph = torch.exp(-tempGraph / sigma_omega)
                weightGraph += tempGraph
                CostASW += tempGraph * Cost[:, :, i:i + h - pad_len * 2, j:j + w - pad_len * 2]
    
        CostASW = CostASW / weightGraph

        return CostASW        
