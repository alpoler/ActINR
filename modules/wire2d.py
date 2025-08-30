#!/usr/bin/env python

import os
import sys
import tqdm
import pdb
import utils
import importlib
import time

import numpy as np
import torch
from torch import nn
import cv2

import torch.nn.functional as F

import utils
import losses

utils = importlib.reload(utils)

class ComplexGaborLayer2D(nn.Module):
    '''
        Implicit representation with complex Gabor nonlinearity with 2D activation function
        
        Inputs;
            in_features: Input features
            out_features; Output features
            bias: if True, enable bias for the linear operation
            is_first: Legacy SIREN parameter
            omega_0: Legacy SIREN parameter
            omega0: Frequency of Gabor sinusoid term
            sigma0: Scaling of Gabor Gaussian term
            trainable: If True, omega and sigma are trainable parameters
    '''
    
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega0=10.0, sigma0=10.0,
                 trainable=False):
        super().__init__()
        self.omega_0 = omega0
        self.scale_0 = sigma0
        self.is_first = is_first
        
        self.in_features = in_features
        
        if self.is_first:
            dtype = torch.float
        else:
            dtype = torch.cfloat
            
        # Set trainable parameters if they are to be simultaneously optimized
        self.omega_0 = nn.Parameter(self.omega_0*torch.ones(1), trainable)
        self.scale_0 = nn.Parameter(self.scale_0*torch.ones(1), trainable)
        
        self.linear = nn.Linear(in_features,
                                out_features,
                                bias=bias,
                                dtype=dtype)
        
        # Second Gaussian window
        self.scale_orth = nn.Linear(in_features,
                                    out_features,
                                    bias=bias,
                                    dtype=dtype)
    
    def forward(self, input):
        lin = self.linear(input)
        
        scale_x = lin
        scale_y = self.scale_orth(input)
        
        freq_term = torch.exp(1j*self.omega_0*lin)
        
        arg = scale_x.abs().square() + scale_y.abs().square()
        gauss_term = torch.exp(-self.scale_0*self.scale_0*arg)
                
        return freq_term*gauss_term
    
class INR(nn.Module):
    def __init__(self, in_features, hidden_features, 
                 hidden_layers, 
                 out_features, outermost_linear=True,
                 first_omega_0=10, hidden_omega_0=10., scale=10.0,
                 pos_encode=False, sidelength=512, fn_samples=None,
                 use_nyquist=True):
        super().__init__()
        
        # All results in the paper were with the default complex 'gabor' nonlinearity
        self.nonlin = ComplexGaborLayer2D
        self.mode = '2d'
        
        # Since complex numbers are two real numbers, reduce the number of 
        # hidden parameters by 4
        hidden_features = int(hidden_features/2)
        dtype = torch.cfloat
        self.complex = True
        self.wavelet = 'gabor'    
        
        # Legacy parameter
        self.pos_encode = False
            
        self.net = []
        self.net.append(self.nonlin(in_features,
                                    hidden_features, 
                                    omega0=first_omega_0,
                                    sigma0=scale,
                                    is_first=True,
                                    trainable=False))

        for i in range(hidden_layers):
            self.net.append(self.nonlin(hidden_features,
                                        hidden_features, 
                                        omega0=hidden_omega_0,
                                        sigma0=scale))

        final_linear = nn.Linear(hidden_features,
                                 out_features,
                                 dtype=dtype)            
        self.net.append(final_linear)
        
        self.net = nn.Sequential(*self.net)
    
    def bias_parameters(self):
        '''
            Wrapper to get only bias parameters
        '''
        params = []
        nparams = 0
        for module in self.net:
            try:
                params.append(module.linear.bias)
                if self.mode == '2d':
                    params.append(module.scale_orth.bias)
            except AttributeError:
                params.append(module.bias)
                
            nparams += params[-1].numel()
            
        self.bias_nparams = nparams
        return params
    
    def set_weights(self, ref_model):
        '''
            Wrapper to set the weights of one model to another
        '''
        for idx in range(len(self.net)):
            try:
                self.net[idx].linear.weight = \
                    ref_model.net[idx].linear.weight
                
                if self.mode == '2d':
                    self.net[idx].scale_orth.weight = \
                        ref_model.net[idx].scale_orth.weight
            except AttributeError:
                self.net[idx].weight = \
                    ref_model.net[idx].weight
    
    def forward(self, coords):
        output = self.net(coords)
        
        if self.wavelet == 'gabor':
            return output.real
         
        return output
    
def single_inr(im_list, nscales, stopping_mse, config):
    '''
        Kilonerf training that runs single inr for multiple
        images
        Inputs:
            im_list: List of (H, W, 3) images to fit. 
            switch_mse: Blocks are terminated when this MSE is achieved
            stopping_mse: Fitting is halted if this MSE is achieved. Set to 0 to run for all iterations
                
        Outputs:
            imfit_list: List of final fitted image
            info: Dictionary with debugging information
                mse_array: MSE as a function of epochs
                time_array: Time as a function of epochs
                num_units_array: Number of active units as a function of epochs
                nparams: Total number of parameters
            model: Trained model
    '''
    
    if sys.platform == 'win32':
        visualize = True
    else:
        visualize = False

    nimg = len(im_list)
    H, W, _ = im_list[0].shape
    
    imten_list = [torch.tensor(im).permute(2, 0, 1)[None, ...] for im in im_list]
    
    imten = torch.cat(imten_list, 0).cuda()
    
    # Create model
    hidden_omega_0 = 1.0
    model = INR(
        in_features=config.in_features,
        out_features=config.out_features*nimg, 
        hidden_features=config.nfeat, 
        hidden_layers=config.nlayers,
        outermost_linear=True,
        first_omega_0=config.omega_0,
        hidden_omega_0=hidden_omega_0,
        scale=config.scale
    ).cuda()
    
    params = model.parameters()
    nparams = utils.count_parameters(model)
    
    nparams_array = np.zeros(nimg)
    nparams_array[0] = nparams
        
    optim = torch.optim.Adam(lr=config.lr, params=params)
        
    # Criteria
    criterion = losses.L2Norm()
    
    # Create inputs
    X, Y = torch.meshgrid(torch.linspace(-1, 1, W),
                          torch.linspace(-1, 1, H))
    coords = torch.hstack((X.reshape(-1, 1), Y.reshape(-1, 1)))[None, ...]
    coords = coords.cuda()
        
    mse_array = np.zeros(config.epochs)
    best_mse = float('inf')
    best_img = None
    
    tbar = tqdm.tqdm(range(config.epochs))
        
    for idx in tbar:
        tic = time.time()
        lr = config.lr*pow(0.1, idx/config.epochs)
            
        optim.param_groups[0]['lr'] = lr
        
        optim.zero_grad()
        
        im_estim = model(coords).reshape(H, W, -1).permute(2, 1, 0)
        im_estim = im_estim.reshape(nimg, -1, H, W)
                
        loss = criterion(im_estim, imten)
        loss.backward()
        optim.step()
        lossval = loss.item()
        
        im_diff = abs(imten - im_estim).mean(1)*10
        im_diff_cpu = im_diff.detach().cpu().numpy()
        
        im_estim_cpu = im_estim.detach().permute(0, 2, 3, 1).cpu().numpy()
        
        mse_array[idx] = lossval
        
        with torch.no_grad():
            mse_list = ((im_estim - imten)**2).mean([1, 2, 3])
        mse_list = tuple(mse_list.cpu().numpy().tolist())
        
        if lossval < best_mse:
            best_mse = lossval
            best_img = im_estim_cpu
            
        if lossval < stopping_mse:
            best_img = im_estim_cpu
            break
        
        if visualize:
            for idx in range(nimg):
                cv2.imshow('Estim %d'%idx, im_estim_cpu[idx, ...])
            cv2.waitKey(1)
        
        mse_minmax = (min(mse_list), max(mse_list))
        tbar.set_description(('%.2e, %.2e')%mse_minmax)
        tbar.refresh()
    
    psnr_array = -10*np.log10(mse_list)
    info = {'nparams_array': nparams_array,
            'psnr_array': psnr_array}
    
    return best_img, info, model

def multibias(im_list, nscales, stopping_mse, config):
    '''
        Kilonerf training that runs multiple INRs but with
        shared weights across INRs.
        
        Inputs:
            im_list: List of (H, W, 3) images to fit. 
            nscales: Required for compatibility with miner
            switch_mse: Blocks are terminated when this MSE is achieved
            stopping_mse: Fitting is halted if this MSE is achieved. Set to 0 to run for all iterations
                
        Outputs:
            imfit_list: List of final fitted image
            info: Dictionary with debugging information
                mse_array: MSE as a function of epochs
                time_array: Time as a function of epochs
                num_units_array: Number of active units as a function of epochs
                nparams: Total number of parameters
            model: Trained model
    '''
    
    if sys.platform == 'win32':
        visualize = True
    else:
        visualize = False

    nimg = len(im_list)
    H, W, _ = im_list[0].shape
    
    imten_list = [torch.tensor(im).permute(2, 0, 1)[None, ...] for im in im_list]
    
    imten = torch.cat(imten_list, 0).cuda()
    
    model_list = []
    params = []
    nparams_array = np.zeros(nimg)
    
    for idx in range(nimg):
        hidden_omega_0 = 1.0
        model = INR(
            in_features=config.in_features,
            out_features=config.out_features*nimg, 
            hidden_features=config.nfeat, 
            hidden_layers=config.nlayers,
            outermost_linear=True,
            first_omega_0=config.omega_0,
            hidden_omega_0=hidden_omega_0,
            scale=config.scale
        ).cuda()
    
        if idx > 0:
            model.set_weights(model_list[0])
            params += model.bias_parameters()
            nparams_array[idx] = model.bias_nparams
        else:
            params+= list(model.parameters())
            nparams_array[idx] = utils.count_parameters(model)
            
        model_list.append(model)
        
    optim = torch.optim.Adam(lr=config.lr, params=params)
        
    # Criteria
    criterion = losses.L2Norm()
    
    # Create inputs
    X, Y = torch.meshgrid(torch.linspace(-1, 1, W),
                                  torch.linspace(-1, 1, H))
    coords = torch.hstack((X.reshape(-1, 1), Y.reshape(-1, 1)))[None, ...]
    coords = coords.cuda()
        
    mse_array = np.zeros(config.epochs)
    best_mse = float('inf')
    best_img = None

    tbar = tqdm.tqdm(range(config.epochs))
    
    for idx in tbar:
        tic = time.time()
        lr = config.lr*pow(0.1, idx/config.epochs)
            
        optim.param_groups[0]['lr'] = lr
        
        optim.zero_grad()
        
        im_estim_list = []
        
        for m_idx in range(nimg):
            model = model_list[m_idx]
            im_estim_list.append(model(coords).permute(2, 1, 0))
        
        im_estim = torch.cat(im_estim_list, 0)
        
        loss = criterion(im_estim, imten)
        loss.backward()
        optim.step()
        lossval = loss.item()
        
        im_diff = abs(imten - im_estim).mean(1)*10
        im_diff_cpu = im_diff.detach().cpu().numpy()
        
        im_estim_cpu = im_estim.detach().permute(0, 2, 3, 1).cpu().numpy()
        
        mse_array[idx] = lossval
        
        with torch.no_grad():
            mse_list = ((im_estim - imten)**2).mean([1, 2, 3])
        mse_list = tuple(mse_list.cpu().numpy().tolist())
        
        if lossval < best_mse:
            best_mse = lossval
            best_img = im_estim_cpu
            
        if lossval < stopping_mse:
            best_img = im_estim_cpu
            break
        
        if visualize:
            for idx in range(nimg):
                cv2.imshow('Estim %d'%idx, im_estim_cpu[idx, ...])
            cv2.waitKey(1)
        
        mse_minmax = (min(mse_list), max(mse_list))
        tbar.set_description(('%.2e, %.2e')%mse_minmax)
        tbar.refresh()
        
    psnr_array = -10*np.log10(np.array(mse_list))
    info = {'psnr_array': psnr_array,
            'nparams_array': nparams_array}
    
    return best_img, info, model_list
