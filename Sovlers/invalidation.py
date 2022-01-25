import torch
import torch.nn as nn
import torch.nn.functional as F

def resample(grid, dispmap_view, disp_to_resample, view_type='left'):
    dispmap_view_norm = dispmap_view * 2 / dispmap_view.shape[-1]
    dispmap_view_norm = dispmap_view_norm.cuda()

    dispmap_view_norm = dispmap_view_norm.squeeze(1).unsqueeze(3)
    dispmap_view_norm = torch.cat((dispmap_view_norm, torch.zeros(dispmap_view_norm.size()).cuda()), dim=3)
    
    if view_type == 'left':    
       grid_view = grid - dispmap_view_norm
    elif view_type == 'right':    
       grid_view = grid + dispmap_view_norm        
    else:    
       raise NotImplementedError('Incorrect view type: [{:s}]'.format(view_type))
    
    recon_dispmap = F.grid_sample(disp_to_resample, grid_view)

    return recon_dispmap

class LRInvalidation(nn.Module):
    '''
    Args:
        left_img right_img: N * C * H * W,
        dispmap : N * H * W
    '''
    def __init__(self):
        super(LRInvalidation, self).__init__()
        self.theta = torch.Tensor(
            [[1, 0, 0],  
            [0, 1, 0]]
        )
    
    def forward(self, dispmap_left, dispmap_right):

        n, c, h, w = dispmap_left.shape
        
        theta = self.theta.repeat(n, 1, 1)
                
        grid = F.affine_grid(theta, dispmap_left.size())
        grid = grid.cuda()
        
        recon_dispmap_right = resample(grid, dispmap_left, dispmap_right, view_type='left')
        dispmap_left_diff = (dispmap_left - recon_dispmap_right).abs()        
        valid_pred_left_prob = torch.exp(-0.6931 * dispmap_left_diff.clamp(min=0.0)) #
        invalid_mask_left = valid_pred_left_prob < 0.5

        return invalid_mask_left
