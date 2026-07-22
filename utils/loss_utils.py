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
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
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

def lpips_loss(network_output, gt, net_type="vgg", downsample=True):
    if net_type not in _loss_fns:
        _loss_fns[net_type] = lpips
    loss_fn = _loss_fns[net_type]
    # If already in [-1,1], leave as is.
    if network_output.min() >= -1.0 and network_output.max() <= 1.0 \
       and gt.min() >= -1.0 and gt.max() <= 1.0 \
       and (network_output.min() < 0 or gt.min() < 0):
        pass
    else:
        # Assume [0,1] with possible overshoot and clamp.
        network_output = network_output.clamp(0.0, 1.0)
        gt = gt.clamp(0.0, 1.0)
        network_output = network_output * 2.0 - 1.0
        gt = gt * 2.0 - 1.0
    # Downsample for LPIPS
    if downsample:
        network_output = F.interpolate(
            network_output,
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
        )
        gt = F.interpolate(
            gt,
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
        )
    return loss_fn(network_output, gt, net_type=net_type).mean()

class VGG19PerceptualLoss(nn.Module):
    """
    VGG-19 perceptual loss.

    resize options:
        False      -> no resize
        int > 1    -> downsample by this factor (e.g. 4)
        (H, W)     -> resize to exact size
    """

    def __init__(
        self,
        layer_weights=(1.0, 1.0, 1.0, 1.0, 1.0),
        resize=False,
    ):
        super().__init__()
        weights = models.VGG19_Weights.DEFAULT
        vgg = models.vgg19(weights=weights).features.eval()
        self.blocks = nn.ModuleList([
            vgg[0:4],    # relu1_2
            vgg[4:9],    # relu2_2
            vgg[9:18]    # relu3_4
        ])
        for block in self.blocks:
            for p in block.parameters():
                p.requires_grad = False
        self.layer_weights = layer_weights
        self.resize = resize
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
        )

    def _resize(self, img):
        if self.resize is False:
            return img
        # resize to exact resolution
        if isinstance(self.resize, (tuple, list)):
            return F.interpolate(
                img,
                size=self.resize,
                mode="bilinear",
                align_corners=False,
            )
        # downsample by scale factor
        if isinstance(self.resize, int):
            if self.resize <= 1:
                return img
            h, w = img.shape[-2:]
            new_h = max(1, h // self.resize)
            new_w = max(1, w // self.resize)
            return F.interpolate(
                img,
                size=(new_h, new_w),
                mode="bilinear",
                align_corners=False,
            )
        raise ValueError("resize must be False, int, or (H, W)")

    def forward(self, pred, target):
        if pred.dim() == 3:
            pred = pred.unsqueeze(0)
            target = target.unsqueeze(0)
        pred = self._resize(pred)
        target = self._resize(target)

        pred = (pred - self.mean) / self.std
        target = (target - self.mean) / self.std
        loss = 0.0
        x = pred
        y = target
        for weight, block in zip(self.layer_weights, self.blocks):
            x = block(x)
            y = block(y)
            loss += weight * F.l1_loss(x, y)
        return loss

