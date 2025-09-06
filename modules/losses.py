#!/usr/bin/env python

import os
import sys
from torch.functional import align_tensors
import tqdm
import pdb
import math

import numpy as np
from skimage.metrics import structural_similarity as ssim_func

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
#from torchgeometry.losses import SSIM

from PIL import Image
#from torchvision.transforms import Resize, Compose, ToTensor, Normalize
#from pytorch_wavelets import DWTForward, DWTInverse

import skimage
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import cv2

from modules import utils

class TVNorm():
    def __init__(self, mode='l1'):
        self.mode = mode
    def __call__(self, img):
        grad_x = img[..., 1:, 1:] - img[..., 1:, :-1]
        grad_y = img[..., 1:, 1:] - img[..., :-1, 1:]
        
        if self.mode == 'isotropic':
            #return torch.sqrt(grad_x.abs().pow(2) + grad_y.abs().pow(2)).mean()
            return torch.sqrt(grad_x**2 + grad_y**2).mean()
        elif self.mode == 'l1':
            return abs(grad_x).mean() + abs(grad_y).mean()
        else:
            return (grad_x.pow(2) + grad_y.pow(2)).mean()     
        
class DiagLoss():
    def __init__(self):
        pass
    def __call__(self, mat):
        gramm = torch.bmm(mat.permute(0, 2, 1), mat)
        gdiag = torch.diag(torch.diag(gramm[0, ...]))[None, ...]
        
        return ((gramm - gdiag)**2).mean()   
    
class WaveletNorm(torch.nn.Module):
    def __init__(self, J, wave='db6', mode='per'):
        super().__init__()
        
        self.dwt = DWTForward(J=J, wave=wave, mode=mode)
        
    def __call__(self, img):
        Yl, Yh = self.dwt(img)
        
        loss = torch.zeros(1).cuda()
        
        for Yh_sub in Yh:
            loss = loss + abs(Yh_sub).sum()
            
        return loss
    
class HessianNorm():
    def __init__(self):
        pass
    def __call__(self, img):
        # Compute Individual derivatives
        fxx = img[..., 1:-1, :-2] + img[..., 1:-1, 2:] - 2*img[..., 1:-1, 1:-1]
        fyy = img[..., :-2, 1:-1] + img[..., 2:, 1:-1] - 2*img[..., 1:-1, 1:-1]
        fxy = img[..., :-1, :-1] + img[..., 1:, 1:] - \
              img[..., 1:, :-1] - img[..., :-1, 1:]
          
        return torch.sqrt(fxx.abs().pow(2) +\
                          2*fxy[..., :-1, :-1].abs().pow(2) +\
                          fyy.abs().pow(2)).mean()
    
class L1Norm():
    def __init__(self):
        pass
    def __call__(self, x1, x2):
        return abs(x1 - x2).mean()        

class L2Norm():
    def __init__(self):
        pass
    def __call__(self, x1, x2):
        return ((x1-x2).pow(2)).mean()    
    
class CosineNorm():
    def __init__(self):
        pass
    def __call__(self, x1, x2):
        c11 = (x1*x1).mean()
        c22 = (x2*x2).mean()
        c12 = (x1*x2).mean()
        
        return c12/torch.sqrt(c11*c22)
    
class SSIMNorm():
    def __init__(self, imshape):
        self.ssim = SSIM(5, reduction='mean')
        self.imshape = imshape
    
    def __call__(self, x1, x2):
        nchan, _, nbatch = x1.shape
        x1_img = x1.permute(0, 2, 1).reshape(nchan, nbatch,
                                             self.imshape[0],
                                             self.imshape[1])
        x2_img = x2.permute(0, 2, 1).reshape(nchan, nbatch,
                                             self.imshape[0],
                                             self.imshape[1])
        return self.ssim(x1_img, x2_img)
class Charbonnier():
    def __init__(self):
        pass
    def __call__(self, x1, x2):
        return (((x1-x2)**2 + 1).sqrt()- 1).mean()
    

# Rounding is imlated with addition of noise
# Its pdf is cdf(x+n+0.5) - cdf(x+n-0.5) 
def self_information(weight, prob_model, is_single_model=False, is_val=False, g=None):
    weight = (weight + torch.rand(weight.shape, generator=g).to(weight)-0.5) if not is_val else torch.round(weight)
    weight_p = weight + 0.5
    weight_n = weight - 0.5
    if not is_single_model:
        prob = prob_model(weight_p) - prob_model(weight_n)
    else:
        prob = prob_model(weight_p.reshape(-1,1))-prob_model(weight_n.reshape(-1,1))
    total_bits = torch.sum(torch.clamp(-1.0 * torch.log(prob + 1e-10) / np.log(2.0), 0, 50))
    return total_bits, prob    
    
class entropy_reg(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.loss = {}
        self.cfg = cfg

    def forward(self,latents, prob_models, single_prob_model, lambda_loss):
        bits = num_elems = 0
        for latent_tag,cur_latent in latents.items():
            if "_" in latent_tag:
                try:
                    layer_no, layer_type = latent_tag.split("_")
                    cur_prob_model= prob_models[layer_no][layer_type]
                except:
                    try:
                        layer_no, layer_type,isreal = latent_tag.split("_")
                        idx= 0 if "real" in isreal else 1
                        cur_prob_model= prob_models[layer_no][layer_type][idx]
                    except:
                        layer_no,isreal = latent_tag.split("_")
                        idx= 0 if "real" in isreal else 1
                        cur_prob_model= prob_models[layer_no][idx]
                    
                    

            else:
                layer_no = latent_tag
                cur_prob_model = prob_models[layer_no]
            if torch.any(torch.isnan(cur_latent)):
                raise Exception('Weights are NaNs')
            cur_bits, prob = self_information(cur_latent,cur_prob_model, single_prob_model, is_val=False)
            bits += cur_bits
            num_elems += prob.size(0)
        self.loss = bits/num_elems*lambda_loss #{'ent_loss': bits/num_elems*lambda_loss}
        return self.loss, bits.float().item()/8

    

@torch.no_grad()
def get_metrics(gt, estim, lpip_func=None, pad=True):
    '''
        Compute SNR, PSNR, SSIM, and LPIP between two images.
        
        Inputs:
            gt: Ground truth image
            estim: Estimated image
            lpip_func: CUDA function for computing lpip value
            pad: if True, remove boundaries when computing metrics
            
        Outputs:
            metrics: dictionary with following fields:
                snrval: SNR of reconstruction
                psnrval: Peak SNR 
                ssimval: SSIM
                lpipval: VGG perceptual metrics
    '''
    if min(gt.shape) < 50:
        pad = False
    if pad:
        gt = gt[20:-20, 20:-20]
        estim = estim[20:-20, 20:-20]
        
    snrval = utils.asnr(gt, estim)
    psnrval = utils.asnr(gt, estim, compute_psnr=True)
    ssimval = ssim_func(gt, estim, multichannel=True)
    
    # Need to convert images to tensors for computing lpips
    gt_ten = torch.tensor(gt)[None, None, ...]
    estim_ten = torch.tensor(estim)[None, None, ...]
    
    # For some reason, the image values should be [-1, 1]
    if lpip_func is not None:
        lpipval = lpip_func(2*gt_ten-1, 2*estim_ten-1).item()
    else:
        lpipval = 0
    
    metrics = {'snrval': snrval,
               'psnrval': psnrval,
               'ssimval': ssimval,
               'lpipval': lpipval}
    return metrics