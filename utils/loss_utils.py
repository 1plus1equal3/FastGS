#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
from lpipsPyTorch import lpips

C1 = 0.01 ** 2
C2 = 0.03 ** 2

def charbonnier_loss(network_output, gt, eps=1e-6):
    return torch.sqrt((network_output - gt) ** 2 + eps ** 2).mean()

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
    


_loss_fns = {}

def lpips_loss(network_output, gt, net_type="vgg"):
    if net_type not in _loss_fns:
        _loss_fns[net_type] = lpips
    loss_fn = _loss_fns[net_type]
    mn = min(network_output.min().item(), gt.min().item())
    mx = max(network_output.max().item(), gt.max().item())
    # [0,1] -> [-1,1]
    if 0.0 <= mn and mx <= 1.0:
        network_output = network_output * 2.0 - 1.0
        gt = gt * 2.0 - 1.0
    # Already [-1,1]
    elif -1.0 <= mn and mx <= 1.0:
        pass
    else:
        raise ValueError(
            f"LPIPS expects inputs in [0,1] or [-1,1], "
            f"but got range [{mn:.3f}, {mx:.3f}]."
        )
    return loss_fn(network_output, gt, net_type=net_type).mean()

