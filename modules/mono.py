#!/usr/bin/env python

import os
import sys
from numpy.lib.twodim_base import mask_indices
import tqdm
import importlib
import time
import pdb
import copy
import configparser
import argparse
import ast

import numpy as np
from scipy import io
from skimage.metrics import structural_similarity as ssim_func

import cv2
import torch
from torch.optim.lr_scheduler import ExponentialLR
from torch.nn import Parameter

import folding_utils as unfoldNd
import nvidia_smi

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
plt.gray()

import utils
import losses
import mono_models

utils = importlib.reload(utils)
losses = importlib.reload(losses)
mono_models = importlib.reload(mono_models)

def mono(im, stopping_mse, config, model=None,
         bias_only=False, warm_start=False,
         weights=None):
    '''
        Fixed Neural Implicit Representation fitting for images 
        
        
        Inputs:
            im: (H, W, 3) Image to fit. 
            switch_mse: Not applicable
            stopping_mse: Fitting is halted if this MSE is achieved. Set to 0 to run for all iterations
            model: Pre-optimized model?
            bias_only: Optimize all parameters or only biases
            warm_start: If True, start from a pre-trained model
            weights: If warm_start is True, use these weights 
                as initialization
                
        Outputs:
            imfit: Final fitted image
            info: Dictionary with debugging information
                mse_array: MSE as a function of epochs
                time_array: Time as a function of epochs
                
    '''
    
    if sys.platform == 'win32':
        visualize = True
    else:
        visualize = False

    if not hasattr(config, 'save_epochs'):
        config.save_epochs = False

    H, W, _ = im.shape
    imten = torch.tensor(im).cuda()
    
    if config.nonlin == 'wire':
        config.nonlin = 'gabor2d'
        hidden_omega = 2*config.scale
    else:
        hidden_omega = config.omega_0
    
    # Create model
    if model is None:
        model = mono_models.INR(in_features=config.in_features,
                                out_features=config.out_features,            
                                nonlinearity=config.nonlin,
                                hidden_features=config.nfeat, 
                                hidden_layers=config.nlayers,
                                outermost_linear=True,
                                first_omega_0=config.omega_0,
                                hidden_omega_0=hidden_omega,
                                scale=config.scale
                            ).cuda()
            
    if warm_start:
        model.load_state_dict(weights)
        
        # Information is encoded in weights, not biases
        model.reset_bias()
            
    if bias_only:
        params = model.bias_parameters()
        nparams = model.bias_nparams
        
        params.append(model.net[0].linear.weight)
        nparams += params[-1].numel()
        
        params.append(model.net[0].orth_scale.weight)
        nparams += params[-1].numel()             
    else:
        params = model.parameters()
        nparams = utils.count_parameters(model)
        
    optim = torch.optim.Adam(lr=config.lr, params=params)
        
    # Criteria
    criterion = losses.L2Norm()
    
    # Create inputs
        
    mse_array = np.zeros(config.epochs)
    time_array = np.zeros(config.epochs)
    best_mse = float('inf')
    best_img = None
    
    if config.save_epochs:
        imstack = np.zeros((config.epochs, H, W, 3),
                           dtype=np.uint8)
    else:
        imstack = np.zeros(1)
    
    tbar = tqdm.tqdm(range(config.epochs))
    
    if visualize:
        cv2.imshow('GT', im)
        cv2.waitKey(1)
        
    x = torch.linspace(-1, 1, W)
    y = torch.linspace(-1, 1, H)
    
    X, Y = torch.meshgrid(x, y, indexing='xy')
    coords = torch.hstack((X.reshape(-1, 1), Y.reshape(-1, 1)))[None, ...].cuda()
    
    for idx in tbar:
        tic = time.time()
        lr = config.lr*pow(0.5, idx/config.epochs)
        
        if bias_only:
            lr = lr*(1-np.exp(-idx*0.05))
            
        optim.param_groups[0]['lr'] = lr
        
        optim.zero_grad()
        
        im_estim = model(coords)
        im_estim = im_estim.reshape(H, W, 3)
        
        loss = criterion(im_estim, imten)
        loss.backward()
        optim.step()
        lossval = loss.item()
        
        epoch_time = time.time() - tic
        
        im_estim_cpu = im_estim.squeeze().detach().cpu().numpy()
        
        mse_array[idx] = lossval
        time_array[idx] = epoch_time
        
        if config.save_epochs:
            imstack[idx, ...] = (255*im_estim_cpu).astype(np.uint8)
        
        if lossval < best_mse:
            best_mse = lossval
            best_img = im_estim_cpu
            
        if lossval < stopping_mse:
            best_img = im_estim_cpu
            break
        
        if visualize:
            cv2.imshow('Estim', im_estim_cpu)
            cv2.waitKey(1)
        
        tbar.set_description('%.2e'%lossval)
        tbar.refresh()
    
    info = {'time_array': time_array,
            'mse_array': mse_array,
            'nparams': nparams,
            'imstack': imstack}
    
    return best_img, info, model
    
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
    
    # Create folders and unfolders
    unfold = torch.nn.Unfold(kernel_size=config.ksize, stride=config.stride)
    fold = torch.nn.Fold(output_size=(H, W), kernel_size=config.ksize,
                         stride=config.stride)
    
    # Find out number of chunks
    weighing = torch.ones(1, 1, H, W)
    nchunks = unfold(weighing).shape[-1]    
    
    # Create model
    if config.nonlin == 'sine':
        model = siren.AdaptiveMultiSiren(
            in_features=config.in_features,
            out_features=config.out_features*nimg, 
            n_channels=nchunks,
            hidden_features=config.nfeat, 
            hidden_layers=config.nlayers,
            outermost_linear=True,
            first_omega_0=config.omega_0,
            hidden_omega_0=config.omega_0
        ).cuda()
    elif config.nonlin == 'wire':
        hidden_omega_0 = 1.0
        
        model = wire.AdaptiveMultiWIRE(
            in_features=config.in_features,
            out_features=config.out_features*nimg, 
            n_channels=nchunks,
            hidden_features=config.nfeat, 
            hidden_layers=config.nlayers,
            outermost_linear=True,
            first_omega_0=config.omega_0,
            hidden_omega_0=hidden_omega_0,
            scale=config.scale,
            const=1.0
        ).cuda()
        
    params = model.parameters()
    nparams = utils.count_parameters(model)
    
    nparams_array = np.zeros(nimg)
    nparams_array[0] = nparams
        
    optim = torch.optim.Adam(lr=config.lr, params=params)
        
    # Criteria
    criterion = losses.L2Norm()
    
    # Create inputs
    coords_chunked = utils.get_coords([H, W],
                                      config.ksize,
                                      config.coordstype,
                                      unfold)
    coords_chunked = coords_chunked.cuda()
    
    mse_array = np.zeros(config.epochs)
    best_mse = float('inf')
    best_img = None
    
    learn_indices = torch.arange(nchunks, device='cuda')
    

    tbar = tqdm.tqdm(range(config.epochs))
        
    for idx in tbar:
        tic = time.time()
        lr = config.lr*pow(0.1, idx/config.epochs)
            
        optim.param_groups[0]['lr'] = lr
        
        optim.zero_grad()
        
        im_estim_chunked = model(coords_chunked, learn_indices).permute(2, 1, 0)
        
        im_estim = fold(im_estim_chunked).reshape(nimg, -1, H, W)
        
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
    
    if nimg == 2:
        info['im_diff'] = im_diff_cpu
    
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
    
    # Create folders and unfolders
    unfold = torch.nn.Unfold(kernel_size=config.ksize, stride=config.stride)
    fold = torch.nn.Fold(output_size=(H, W),
                         kernel_size=config.ksize,
                         stride=config.stride)
    
    # Find out number of chunks
    weighing = torch.ones(1, 1, H, W)
    nchunks = unfold(weighing).shape[-1]    
    
    model_list = []
    params = []
    nparams_array = np.zeros(nimg)
    
    for idx in range(nimg):
        # Create model
        if config.nonlin == 'sine':
            model = siren.AdaptiveMultiSiren(
                in_features=config.in_features,
                out_features=config.out_features, 
                n_channels=nchunks,
                hidden_features=config.nfeat, 
                hidden_layers=config.nlayers,
                outermost_linear=True,
                share_weights=config.share_weights,
                first_omega_0=config.omega_0,
                hidden_omega_0=config.omega_0
            ).cuda()
        elif config.nonlin == 'wire':
            hidden_omega_0 = 1.0
            
            model = wire.AdaptiveMultiWIRE(
                in_features=config.in_features,
                out_features=config.out_features, 
                n_channels=nchunks,
                hidden_features=config.nfeat, 
                hidden_layers=config.nlayers,
                outermost_linear=True,
                share_weights=config.share_weights,
                first_omega_0=config.omega_0,
                hidden_omega_0=hidden_omega_0,
                scale=config.scale,
                const=1.0
            ).cuda()
    
        if idx > 0:
            model.set_weights(model_list[0])
            params += model.bias_parameters()
            nparams_array[idx] = model.bias_nparams
        else:
            params+= list(model.parameters())
            nparams_array[idx] = utils.count_parameters(model)
            
        model_list.append(model)
    
    #model_state_dict = model_list[0].state_dict()    
    #for key in model_state_dict.keys():
    #    print(key, model_state_dict[key].dtype)
    
    optim = torch.optim.Adam(lr=config.lr, params=params)
        
    # Criteria
    criterion = losses.L2Norm()
    
    # Create inputs
    coords_chunked = utils.get_coords([H, W],
                                      config.ksize,
                                      config.coordstype,
                                      unfold)
    coords_chunked = coords_chunked.cuda()
        
    mse_array = np.zeros(config.epochs)
    best_mse = float('inf')
    best_img = None
    
    learn_indices = torch.arange(nchunks, device='cuda')

    tbar = tqdm.tqdm(range(config.epochs))
    
    for idx in tbar:
        tic = time.time()
        lr = config.lr*pow(0.1, idx/config.epochs)
            
        optim.param_groups[0]['lr'] = lr
        
        optim.zero_grad()
        
        im_estim_list = []
        
        for m_idx in range(nimg):
            model = model_list[m_idx]
            im_estim_list.append(
                model(coords_chunked, learn_indices).permute(2, 1, 0))
        
        im_estim_chunked = torch.cat(im_estim_list, 0)
        
        im_estim = fold(im_estim_chunked).reshape(nimg, -1, H, W)
        
        if config.warm_start and idx < config.warm_epochs:
            loss = criterion(im_estim[0], imten[0])
        else:
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
    
    if nimg == 2:
        info['imdiff'] = im_diff_cpu
    
    return best_img, info, model_list

def sequential_optim(im_list, nscales, target_mse, config):
    '''
        Kilonerf training that runs multiple INRs but with
        shared weights across INRs. Training of biases happens
        in a sequential manner
        
        Inputs:
            im_list: List of (H, W, 3) images to fit. 
            nscales: Required for compatibility with miner
            target_mse: Fitting is halted if this MSE is achieved. Set to 0 to run for all iterations
            config: Configuration object
                
        Outputs:
            imfit_list: List of final fitted image
            info: Dictionary with debugging information
                mse_array: MSE as a function of epochs
                time_array: Time as a function of epochs
                num_units_array: Number of active units as a function of epochs
                nparams: Total number of parameters
            model: Trained model
    '''
    
    # Train the full model on the first image
    
    best_im_list = []
    models_list = []
    
    nparams_array = np.zeros(len(im_list))
    psnr_array = np.zeros(len(im_list))
    
    best_im, info, model_base = kilonerf(im_list[0],
                                    target_mse,
                                    config,
                                    model=None,
                                    bias_only=False)

    best_im_list.append(best_im[np.newaxis, ...])
    models_list.append(model_base)
    nparams_array[0] = info['nparams']
    psnr_array[0] = utils.psnr(im_list[0], best_im)
    
    for idx in range(1, len(im_list)):
        best_im, info, model = kilonerf(im_list[idx],
                                        target_mse,
                                        config,
                                        model=model_base,
                                        bias_only=True)
        best_im_list.append(best_im[np.newaxis, ...])
        models_list.append(model)
        nparams_array[idx] = info['nparams']
        psnr_array[idx] = utils.psnr(im_list[idx], best_im)
        
    info = {
        'nparams_array': nparams_array,
        'psnr_array': psnr_array
    }
    best_im = np.concatenate(best_im_list, 0)
    
    return best_im, info, models_list

def get_atlas_image(model_dict, mode='avg'):
    '''
        Get an Atlas image from a given set of KiloNeRF
        models
        
        Inputs:
            model_dict: Dictionary with models and configuration
            mode: 'avg' or 'zero'
    '''
    config = argparse.Namespace(**model_dict['config'])
    H, W, nchan = model_dict['imsize']
    
    if config.nonlin == 'wire':
        config.nonlin = 'gabor2d'
    
    if config.nonlin == 'sine':
        model = siren.AdaptiveMultiSiren(
            in_features=config.in_features,
            out_features=config.out_features, 
            n_channels=nchunks,
            hidden_features=config.nfeat, 
            hidden_layers=config.nlayers,
            outermost_linear=True,
            first_omega_0=config.omega_0,
            hidden_omega_0=config.omega_0
        ).cuda()
    elif config.nonlin == 'wire':
        hidden_omega_0 = 1.0
        
        model = wire.AdaptiveMultiWIRE(
            in_features=config.in_features,
            out_features=config.out_features, 
            n_channels=nchunks,
            hidden_features=config.nfeat, 
            hidden_layers=config.nlayers,
            outermost_linear=True,
            first_omega_0=config.omega_0,
            hidden_omega_0=hidden_omega_0,
            scale=config.scale,
            const=1.0
        )
        
    params = model.state_dict()
    
    # Set all parameters to zero, and then average of 
    # all model weights
    
    with torch.no_grad():
        for key in params.keys():
            params[key][...] = 0
            
            mkeys = [key for key in model_dict if 'model' in key]
            N = len(mkeys)
            
            
            for mkey in mkeys:
                params[key] += model_dict[mkey][key]/N
                
            if mode == 'zero':
                if 'bias' in key:
                    params[key][...] = 0
                    
            if mode == 'diff':
                if 'bias' in key:
                    diff_param = model_dict['model0'][key] -\
                                 model_dict['model1'][key]
                    params[key][...] = diff_param
                
    model = model.cuda()
                
    # Create inputs
    coords_chunked = utils.get_coords([H, W],
                                      config.ksize,
                                      config.coordstype,
                                      unfold).cuda()
    learn_indices = torch.arange(nchunks, device='cuda')
    
    im_estim_chunked = model(coords_chunked, learn_indices).permute(2, 1, 0)
    
    im_estim = fold(im_estim_chunked)[:, 0, ...]
    
    im_estim_cpu = im_estim.detach().permute(1, 2, 0).cpu().numpy()
    
    return im_estim_cpu

def get_activations(model_dict):
    '''
        Get activation outputs from a given set of models
        
        Inputs:
            model_dict: Dictionary with models and configuration
    '''
    config = argparse.Namespace(**model_dict['config'])
    H, W, nchan = model_dict['imsize']
    
    # Generate fold and unfold operators
    unfold = torch.nn.Unfold(kernel_size=config.ksize, 
                             stride=config.stride)
    fold = torch.nn.Fold(output_size=(H, W),
                         kernel_size=config.ksize,
                         stride=config.stride)
    
    # Find out number of chunks
    weighing = torch.ones(1, 1, H, W)
    nchunks = unfold(weighing).shape[-1]
    
    if config.nonlin == 'sine':
        model = siren.AdaptiveMultiSiren(
            in_features=config.in_features,
            out_features=config.out_features, 
            n_channels=nchunks,
            hidden_features=config.nfeat, 
            hidden_layers=config.nlayers,
            outermost_linear=True,
            first_omega_0=config.omega_0,
            hidden_omega_0=config.omega_0
        ).cuda()
    elif config.nonlin == 'wire':
        hidden_omega_0 = 1.0
        
        model = wire.AdaptiveMultiWIRE(
            in_features=config.in_features,
            out_features=config.out_features, 
            n_channels=nchunks,
            hidden_features=config.nfeat, 
            hidden_layers=config.nlayers,
            outermost_linear=True,
            first_omega_0=config.omega_0,
            hidden_omega_0=hidden_omega_0,
            scale=config.scale,
            const=1.0
        )
        
    mlist = [key for key in model_dict.keys() if 'model' in key]
