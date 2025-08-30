#!/usr/bin/env python

import os
import sys
import tqdm
import importlib
import time
import pdb
import copy

import numpy as np
from scipy import io
from skimage.metrics import structural_similarity as ssim_func

import cv2
import torch
from torch.nn import Parameter

from modules import folding_utils as unfoldNd

import matplotlib.pyplot as plt
plt.gray()

from modules import utils
from modules import siren
from modules import losses
from modules import volutils
from modules import wire

utils = importlib.reload(utils)
siren = importlib.reload(siren)
losses = importlib.reload(losses)
volutils = importlib.reload(volutils)
wire = importlib.reload(wire)

def miner(im, nscales, stopping_mse, config):
    '''
        Multiscale Neural Implicit Representation (MINER) fitting for images
        
        Inputs:
            im: (H, W, 3) Image to fit. 
            nscales: Number of scales to fit
            stopping_mse: Fitting is halted if this MSE is achieved. Set to 0
                to run for all iterations
            config: Namespace with configuration variables. Check configs folder for more examples
                
        Outputs:
            imfit: Final fitted image
            info: Dictionary with debugging information
                mse_array: MSE as a function of epochs
                time_array: Time as a function of epochs
                num_units_array: Number of active units as a function of epochs
                learn_indices_list: Active indices at the starting of each
                    scale of fitting
                nparams: Total number of parameters
            models: List of models at all scales
    '''
    H, W, _ = im.shape
    # Send image to CUDA
    imten = torch.tensor(im).permute(2, 0, 1)[None, ...]
    config.nimg = 1
    config.kpix = config.ksize[0]*config.ksize[1]
    
    if sys.platform == 'win32':
        visualize = True
    else:
        visualize = False
    
    imten_gpu = imten.cuda()
    
    # Show ground truth image
    if visualize:
        cv2.imshow('GT', im)
        cv2.waitKey(1)
    
    # Parameters to save across iterations
    nparams = 0
    mse_array = np.zeros(config.epochs*(nscales+1))
    time_array = np.zeros(config.epochs*(nscales+1))
    nparams_array = np.zeros(nscales)
    
    prev_params = None
    im_estim = None
    models_list = []
        
    # Begin
    for scale_idx in tqdm.tqdm(range(nscales), dynamic_ncols=True):
        imtarget_ten = torch.nn.functional.interpolate(
            imten_gpu,
            scale_factor=pow(2.0, -nscales+scale_idx+1),
            mode='area')
        _, _, Ht, Wt = imtarget_ten.shape
        
        # Create folders and unfolders
        unfold = torch.nn.Unfold(kernel_size=config.ksize, 
                                 stride=config.stride)
        fold = torch.nn.Fold(output_size=(Ht, Wt), 
                             kernel_size=config.ksize,
                             stride=config.stride)
   
        # Find out number of chunks
        nchunks = int(Ht*Wt/(config.kpix))   
        
        # Create model
        if scale_idx == 0 and nscales > 1:
            nfeat = config.nfeat*4
            lr_div = 4
        else:
            nfeat = config.nfeat
            lr_div = 1
                            
        # Create inputs
        coords_chunked = utils.get_coords((Ht, Wt),
                                          config.ksize,
                                          config.coordstype, 
                                          unfold)
     
        criterion = losses.L2Norm()
        best_mse = float('inf')
        best_img = None
            
        tbar = tqdm.tqdm(range(config.epochs), dynamic_ncols=True)
        
        if scale_idx == 0:
            im_prev = torch.zeros_like(imtarget_ten)
        else:
            im_prev = torch.nn.functional.interpolate(
                im_estim, scale_factor=2, mode='bilinear'
            )
            
        learn_indices = torch.arange(nchunks)
        
        # Make a copy of the parameters and then truncate  
        if scale_idx < 2:      
            model, prev_params = get_model(config, nchunks, nfeat,
                                           learn_indices, None, scale_idx)
        else:
            model, prev_params = get_model(config, nchunks, nfeat,
                                           learn_indices, prev_params, scale_idx)
                    
        # Optimizer
        optim = torch.optim.Adam(lr=config.lr/lr_div,
                                 params=model.parameters())
                
        nparams += utils.count_parameters(model)
        nparams_array[scale_idx] = utils.count_parameters(model)
                
        prev_loss = 0
        
        # Move tensors to GPU
        coords_chunked = coords_chunked.cuda(non_blocking=True)
        learn_indices = learn_indices.cuda(non_blocking=True)
        
        for idx in tbar:             
            if learn_indices.numel() == 0:
                break
            lr = config.lr*pow(0.1, idx/config.epochs)
            optim.param_groups[0]['lr'] = lr
            
            optim.zero_grad()
            
            im_sub_chunked = model(coords_chunked,
                                learn_indices).permute(2, 1, 0)
            
            im_estim = fold(im_sub_chunked).permute(1, 0, 2, 3) +\
                im_prev
            loss = criterion(imtarget_ten, im_estim)
            
            loss.backward()
            optim.step()                
            lossval = loss.item()
            
            with torch.no_grad():
                im_estim_up = torch.nn.functional.interpolate(
                    im_estim, size=(H, W),mode='bilinear'
                )
                mse_full = ((im_estim_up - imten_gpu)**2).mean()
            
            if scale_idx < nscales - 1:
                if abs(lossval - prev_loss) < config.switch_thres and idx > config.epochs//4:
                    break
            
            prev_loss = lossval
            
            mse_array[scale_idx*config.epochs + idx] = mse_full
            
            if lossval < best_mse:
                best_mse = lossval
                best_img = copy.deepcopy(im_estim.detach())
                
            if mse_full < stopping_mse:
                break
                
            if visualize:
                im_estim_cpu = im_estim.detach().cpu()
                im_estim_cpu = im_estim_cpu.squeeze().permute(1, 2, 0).numpy()
                cv2.imshow('Estim', im_estim_cpu)
                cv2.waitKey(1)
            
            tbar.set_description('%.2e | %.2e'%(lossval, mse_full))
            tbar.refresh()
        
        if scale_idx < nscales - 1:
            prev_params = copy_params(model, prev_params,
                                      learn_indices, nchunks,
                                      config, [Ht, Wt])
        
        models_list.append(model)
        
        # Disconnect estimated image from graph
        im_estim = im_estim.detach()
                        
    # Remove zero valued entries
    indices, = np.where(time_array > 0)
    
    info = {'time_array': time_array[indices],
            'mse_array': mse_array[indices],
            'nparams': nparams,
            'nparams_array': nparams_array}
    
    best_img = best_img.cpu().detach().squeeze()
    
    return best_img.permute(1, 2, 0).numpy(), info, models_list

def single_inr(im_list, nscales, stopping_mse, config):
    '''
        Multiscale Neural Implicit Representation (MINER) fitting for images
        
        Inputs:
            im_list: List of (H, W, 3) images to fit. 
            nscales: Number of scales to fit
            stopping_mse: Fitting is halted if this MSE is achieved. Set to 0
                to run for all iterations
            config: Namespace with configuration variables. Check configs folder for more examples
                
        Outputs:
            imfit: Final fitted image
            info: Dictionary with debugging information
                mse_array: MSE as a function of epochs
                time_array: Time as a function of epochs
                num_units_array: Number of active units as a function of epochs
                learn_indices_list: Active indices at the starting of each
                    scale of fitting
                nparams: Total number of parameters
            models: List of models at all scales
    '''
    H, W, _ = im_list[0].shape
    
    # Send image to CUDA
    imten_list = [torch.tensor(im).permute(2, 0, 1)[None, ...] for im in im_list]
    
    imten = torch.cat(imten_list, 0).cuda()
    
    config.nimg = len(imten)
    nimg = config.nimg
    config.kpix = config.ksize[0]*config.ksize[1]
    
    if sys.platform == 'win32':
        visualize = True
    else:
        visualize = False
        
    # Parameters to save across iterations
    nparams = 0
    mse_array = np.zeros(config.epochs*(nscales+1))
    nparams_array = np.zeros((nscales, nimg))
    psnr_array = np.zeros((nscales, nimg))
    
    prev_params = None
    im_estim = None
    models_list = []
        
    # Begin
    for scale_idx in tqdm.tqdm(range(nscales), dynamic_ncols=True):
        imtarget_ten = torch.nn.functional.interpolate(
            imten,
            scale_factor=pow(2.0, -nscales+scale_idx+1),
            mode='area')
        _, _, Ht, Wt = imtarget_ten.shape
        
        # Create folders and unfolders
        unfold = torch.nn.Unfold(kernel_size=config.ksize, 
                                 stride=config.stride)
        fold = torch.nn.Fold(output_size=(Ht, Wt), 
                             kernel_size=config.ksize,
                             stride=config.stride)
   
        # Find out number of chunks
        nchunks = int(Ht*Wt/(config.kpix))   
        
        # Create model
        if scale_idx == 0 and nscales > 1:
            nfeat = config.nfeat*4
            lr_div = 4
        else:
            nfeat = config.nfeat
            lr_div = 1
                            
        # Create inputs
        coords_chunked = utils.get_coords((Ht, Wt),
                                          config.ksize,
                                          config.coordstype, 
                                          unfold)
     
        criterion = losses.L2Norm()
        best_mse = float('inf')
        best_img = None
            
        tbar = tqdm.tqdm(range(config.epochs), dynamic_ncols=True)
        
        if scale_idx == 0:
            im_prev = torch.zeros_like(imtarget_ten)
        else:
            im_prev = torch.nn.functional.interpolate(
                im_estim, scale_factor=2, mode='bilinear'
            )
            
        learn_indices = torch.arange(nchunks)
        
        # Make a copy of the parameters and then truncate  
        if scale_idx < 2:      
            model, prev_params = get_model(config, nchunks, nfeat,
                                           learn_indices, None, scale_idx)
        else:
            model, prev_params = get_model(config, nchunks, nfeat,
                                           learn_indices, prev_params, scale_idx)
                    
        # Optimizer
        optim = torch.optim.Adam(lr=config.lr/lr_div,
                                 params=model.parameters())
                
        nparams += utils.count_parameters(model)
        
        nparams_array[scale_idx, 0] = nparams
                
        prev_loss = 0
        
        # Move tensors to GPU
        coords_chunked = coords_chunked.cuda(non_blocking=True)
        learn_indices = learn_indices.cuda(non_blocking=True)
        
        for idx in tbar:             
            if learn_indices.numel() == 0:
                break
            lr = config.lr*pow(0.1, idx/config.epochs)
            optim.param_groups[0]['lr'] = lr
            
            optim.zero_grad()
            
            im_chunked = model(coords_chunked,
                                learn_indices).permute(2, 1, 0)
            
            im_estim = fold(im_chunked).reshape(nimg, -1, Ht, Wt) +\
                im_prev
            loss = criterion(imtarget_ten, im_estim)
            
            loss.backward()
            optim.step()                
            lossval = loss.item()
            
            with torch.no_grad():
                im_estim_up = torch.nn.functional.interpolate(
                    im_estim, size=(H, W),mode='bilinear'
                )
                mse_list = ((im_estim_up - imten)**2).mean([1, 2, 3])
                mse_full = mse_list.mean()
                
            mse_list = tuple(mse_list.cpu().numpy().tolist())
            
            if scale_idx < nscales - 1:
                diffval = abs(lossval - prev_loss)
                if diffval < config.switch_thres: #and \
                    #idx > config.epochs//4:
                    break
            
            prev_loss = lossval
            
            mse_array[scale_idx*config.epochs + idx] = mse_full
            
            if lossval < best_mse:
                best_mse = lossval
                best_img = copy.deepcopy(im_estim.detach())
                
            if mse_full < stopping_mse:
                break
                
            if visualize:
                im_estim_cpu = im_estim.detach().cpu()
                im_estim_cpu = im_estim_cpu.permute(0, 2, 3, 1).numpy()
                
                for idx in range(nimg):
                    cv2.imshow('Estim %d'%idx,
                               im_estim_cpu[idx, ...])
                cv2.waitKey(1)
            
            mse_minmax = (min(mse_list), max(mse_list))
            tbar.set_description(('%.2e, %.2e')%mse_minmax)
            tbar.refresh()
        
        if scale_idx < nscales - 1:
            prev_params = copy_params(model, prev_params,
                                      learn_indices, nchunks,
                                      config, [Ht, Wt])
        
        models_list.append(model)
        
        # Disconnect estimated image from graph
        im_estim = im_estim.detach()
                        
        psnr_array[scale_idx, ...] = -10*np.log10(mse_list)

    info = {'psnr_array': psnr_array,
            'nparams_array': nparams_array}
    
    best_img = best_img.cpu().detach()
    
    return best_img.permute(0, 2, 3, 1).numpy(), info, models_list

def multibias(im_list, nscales, stopping_mse, config):
    '''
        Multiscale Neural Implicit Representation (MINER) fitting for images
        
        Inputs:
            im_list: List of (H, W, 3) images to fit. 
            nscales: Number of scales to fit
            stopping_mse: Fitting is halted if this MSE is achieved. Set to 0
                to run for all iterations
            config: Namespace with configuration variables. Check configs folder for more examples
                
        Outputs:
            imfit: Final fitted image
            info: Dictionary with debugging information
                mse_array: MSE as a function of epochs
                time_array: Time as a function of epochs
                num_units_array: Number of active units as a function of epochs
                learn_indices_list: Active indices at the starting of each
                    scale of fitting
                nparams: Total number of parameters
            models: List of models at all scales
    '''
    H, W, _ = im_list[0].shape
    
    # Send image to CUDA
    imten_list = [torch.tensor(im).permute(2, 0, 1)[None, ...] for im in im_list]
    
    imten = torch.cat(imten_list, 0).cuda()
    
    config.nimg = 1
    nimg = len(imten)
    config.kpix = config.ksize[0]*config.ksize[1]
    
    if sys.platform == 'win32':
        visualize = True
    else:
        visualize = False
    
    # Parameters to save across iterations
    mse_array = np.zeros(config.epochs*(nscales+1))
    nparams_array = np.zeros((nscales, nimg))
    psnr_array = np.zeros((nscales, nimg))
    
    prev_params = None
    im_estim = None
    models_list_master = []
    prev_params_list = [None]*nimg
        
    # Begin
    for scale_idx in tqdm.tqdm(range(nscales), dynamic_ncols=True):
        imtarget_ten = torch.nn.functional.interpolate(
            imten,
            scale_factor=pow(2.0, -nscales+scale_idx+1),
            mode='area')
        _, _, Ht, Wt = imtarget_ten.shape
        
        model_list = []
        params = []
        
        # Create folders and unfolders
        unfold = torch.nn.Unfold(kernel_size=config.ksize, 
                                 stride=config.stride)
        fold = torch.nn.Fold(output_size=(Ht, Wt), 
                             kernel_size=config.ksize,
                             stride=config.stride)
   
        # Find out number of chunks
        nchunks = int(Ht*Wt/(config.kpix))   
        
        # Create model
        if scale_idx == 0 and nscales > 1:
            nfeat = config.nfeat*4
            lr_div = 4
        else:
            nfeat = config.nfeat
            lr_div = 1
                            
        # Create inputs
        coords_chunked = utils.get_coords((Ht, Wt),
                                          config.ksize,
                                          config.coordstype, 
                                          unfold)
     
        criterion = losses.L2Norm()
        best_mse = float('inf')
        best_img = None
            
        tbar = tqdm.tqdm(range(config.epochs), dynamic_ncols=True)
        
        if scale_idx == 0:
            im_prev = torch.zeros_like(imtarget_ten)
        else:
            im_prev = torch.nn.functional.interpolate(
                im_estim, scale_factor=2, mode='bilinear'
            )
            
        learn_indices = torch.arange(nchunks)
        
        for m_idx in range(nimg):
            if scale_idx < 2:   
                prev_params_mod = None
            else:
                prev_params_mod = prev_params_list[m_idx]   
            model, prev_params = get_model(config,
                                            nchunks,
                                            nfeat,
                                            learn_indices,
                                            prev_params_mod,
                                            scale_idx)
            
            if m_idx > 0:
                model.set_weights(model_list[0])
                params += model.bias_parameters()
                nparams_array[scale_idx, m_idx] += model.bias_nparams
            else:
                params += list(model.parameters())
                nparams_array[scale_idx, m_idx] += utils.count_parameters(model)
            
            prev_params_list[m_idx] = prev_params
            model_list.append(model)
                    
        # Optimizer
        optim = torch.optim.Adam(lr=config.lr/lr_div,
                                 params=params)
                
        prev_loss = 0
        
        # Move tensors to GPU
        coords_chunked = coords_chunked.cuda(non_blocking=True)
        learn_indices = learn_indices.cuda(non_blocking=True)
        
        for idx in tbar:             
            if learn_indices.numel() == 0:
                break
            lr = config.lr*pow(0.1, idx/config.epochs)
            optim.param_groups[0]['lr'] = lr
            
            optim.zero_grad()
            
            im_chunked_list = []
            for m_idx in range(nimg):
                model = model_list[m_idx]
                output = model(coords_chunked, learn_indices)
                im_chunked_list.append(output.permute(2, 1, 0))
            
            im_chunked = torch.cat(im_chunked_list, 0)
            
            im_estim = fold(im_chunked).reshape(nimg, -1, Ht, Wt) +\
                im_prev
            loss = criterion(imtarget_ten, im_estim)
            
            loss.backward()
            optim.step()                
            lossval = loss.item()
            
            with torch.no_grad():
                im_estim_up = torch.nn.functional.interpolate(
                    im_estim, size=(H, W),mode='bilinear'
                )
                mse_list = ((im_estim_up - imten)**2).mean([1, 2, 3])
                mse_full = mse_list.mean()
                
            mse_list = tuple(mse_list.cpu().numpy().tolist())
            
            if scale_idx < nscales - 1:
                diffval = abs(lossval - prev_loss)
                if diffval < config.switch_thres and \
                    idx > config.epochs//4:
                    break
            
            prev_loss = lossval
            
            mse_array[scale_idx*config.epochs + idx] = mse_full
            
            if lossval < best_mse:
                best_mse = lossval
                best_img = copy.deepcopy(im_estim.detach())
                
            if mse_full < stopping_mse:
                break
                
            if visualize:
                im_estim_cpu = im_estim.detach().cpu()
                im_estim_cpu = im_estim_cpu.permute(0, 2, 3, 1).numpy()
                
                for idx in range(nimg):
                    cv2.imshow('Estim %d'%idx, im_estim_cpu[idx, ...])
                cv2.waitKey(1)
            
            mse_minmax = (min(mse_list), max(mse_list))
            tbar.set_description(('%.2e, %.2e')%mse_minmax)
            tbar.refresh()
        
        if scale_idx < nscales - 1:
            for m_idx in range(nimg):
                prev_params_list[m_idx] = copy_params(
                    model_list[m_idx], prev_params_list[m_idx],
                    learn_indices, nchunks, config, [Ht, Wt])
        
        models_list_master.append(model_list)
        
        # Disconnect estimated image from graph
        im_estim = im_estim.detach()
                        
        psnr_array[scale_idx, :] = -10*np.log10(mse_list)
        
    
    # nparams array is not cumulative, fix it
    nparams_array  = nparams_array.cumsum(0)
    
    info = {'psnr_array': psnr_array,
            'nparams_array': nparams_array}
    
    best_img = best_img.cpu().detach()
    
    return best_img.permute(0, 2, 3, 1).numpy(), info, models_list_master

def sequential_optim(im_list, nscales, target_mse, config):
    '''
        WARNING: THIS MODULE DOES NOT WORK
        
        MINER training that runs multiple INRs but with
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
    
    best_im, info, model_base = miner(im_list[0],
                                    target_mse,
                                    config,
                                    model=None,
                                    bias_only=False)

    best_im_list.append(best_im[np.newaxis, ...])
    models_list.append(model_base)
    nparams_array[0] = info['nparams']
    psnr_array[0] = utils.psnr(im_list[0], best_im)
    
    for idx in range(1, len(im_list)):
        best_im, info, model = miner(im_list[idx],
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

@torch.no_grad()
def get_model(config, nchunks, nfeat, master_indices=None, prev_params=None,
              scale_idx=0):
    '''
        Get an adaptive kilo-siren
        
        Inputs:
            config: Configuration structure
            master_indices: Indices used for copying previous parameters
                If None, previous parameters are not copied
            prev_params: If master_indices is not None, then these parameters
                are copied to the model.
            scale_idx: Scale of fitting
    '''        
    # Create a model first
    #if config.signaltype == 'occupancy':
    if config.loss == 'logistic':
        const = 2.0
    else:
        const = pow(2, -scale_idx)
    if config.nonlin == 'wire':
        hidden_omega_0 = 1.0
        
        if scale_idx == 0:
            model = wire.AdaptiveMultiWIRE(
                in_features=config.in_features,
                out_features=config.out_features*config.nimg, 
                n_channels=nchunks,
                hidden_features=config.nfeat, 
                hidden_layers=config.nlayers,
                outermost_linear=True,
                first_omega_0=config.omega_0,
                hidden_omega_0=hidden_omega_0,
                scale=config.scale,
                const=1.0,
                mode='2d'
            ).cuda()
        else:
            model = wire.AdaptiveMultiWIRE(
                in_features=config.in_features,
                out_features=config.out_features*config.nimg, 
                n_channels=nchunks,
                hidden_features=config.nfeat, 
                hidden_layers=config.nlayers,
                outermost_linear=True,
                first_omega_0=config.omega_0/5,
                hidden_omega_0=config.omega_0/5,
                scale=config.scale,
                const=1.0,
                mode='1d'
            ).cuda()
    else:
        model = siren.AdaptiveMultiSiren(
            in_features=config.in_features,
            out_features=config.out_features*config.nimg, 
            n_channels=nchunks,
            hidden_features=nfeat, 
            hidden_layers=config.nlayers,
            outermost_linear=True,
            first_omega_0=config.omega_0,
            hidden_omega_0=config.omega_0,
            nonlin=config.nonlin,
            const=const
        ).cuda()
        
    if config.weightshare == 'noshare':
        print('Is it not better to share?')
        prev_params = copy.deepcopy(model.state_dict())
    
    # Load, save, truncate
    if prev_params is not None:
        model.load_state_dict(prev_params)
    else:
        prev_params = copy.deepcopy(model.state_dict())
        
    # Then see if we need to copy parameters
    if prev_params is not None:
        for l_idx in range(len(model.net)-1):
            # N-1 layers are with non linearity
            model.net[l_idx].linear.weight = \
                Parameter(model.net[l_idx].linear.weight[master_indices, ...])
                
            model.net[l_idx].linear.bias = \
                Parameter(model.net[l_idx].linear.bias[master_indices, ...])
        
        # Last layer is just linear
        model.net[-1].weight = \
            Parameter(model.net[-1].weight[master_indices, ...])
            
        model.net[-1].bias = \
            Parameter(model.net[-1].bias[master_indices, ...])
                
    return model, prev_params

def copy_params(model, prev_params, learn_indices, nchunks,
                config, imsize):
    '''
        Copy parameters from current model to previous set of 
        parameters
    '''
    Ht, Wt = imsize
    prev_params_sub = copy.deepcopy(model.state_dict())
            
    # First copy the new parameters
    for key in prev_params:
        prev_params[key][learn_indices, ...] = prev_params_sub[key][...]
    
    # Now we will double up all the parameters
    indices = np.arange(nchunks).reshape(Ht//config.ksize[0],
                                            Wt//config.ksize[1])
    indices = cv2.resize(indices, None, fx=2, fy=2,
                        interpolation=cv2.INTER_NEAREST).ravel()
    indices = torch.tensor(indices).cuda().long()
    
    for key in prev_params:
        prev_params[key] = prev_params[key][indices, ...]/2.0
        
    return prev_params