import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def epe_metric(target, output, maxdisp):
    mask = (target < maxdisp).float()
    
    target *= mask
    output *= mask
    
    return torch.abs(target - output).mean()

def epe_metric_non_zero(target, output, maxdisp):
    mask = torch.mul(target > 0.0, target < maxdisp).float()
    
    diff = torch.abs(target - output) * mask
    
    return mask.mean()

def tripe_metric(target, output, maxdisp):
    delta = torch.abs(target - output)
    correct = ((delta < 3) | torch.lt(delta, target * 0.05))
    eps = 1e-7
    return 1 - (float(torch.sum(correct))/(delta.numel() + eps))
