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
