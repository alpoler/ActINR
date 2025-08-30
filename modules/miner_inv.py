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

import numpy as np
from scipy import io
from skimage.metrics import structural_similarity as ssim_func

import cv2
import torch
from torch.optim.lr_scheduler import ExponentialLR
from torch.nn import Parameter
#import unfoldNd
import folding_utils as unfoldNd
import nvidia_smi

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
plt.gray()

import utils
import siren
import losses
import volutils
import wire
import miner

utils = importlib.reload(utils)
siren = importlib.reload(siren)
losses = importlib.reload(losses)
volutils = importlib.reload(volutils)
wire = importlib.reload(wire)
miner = importlib.reload(miner)

def miner_imfit(meas_ten, func, nscales, switch_std,  config):
    '''
        Multiscale Neural Implicit Representation (MINER) for solving image-based
        inverse problems
        
        Inputs:
            meas_ten: Measurements tensor
            func: meas = func(im) generates measurements
            nscales: Number of scales to fit
            switch_mse: Blocks are terminated when this MSE is achieved
            stopping_mse: Fitting is halted if this MSE is achieved. Set to 0
                to run for all iterations
            config: Namespace with configuration variables. Check configs folder
                for more examples
                
        Outputs:
            imfit: Final fitted image
            info: Dictionary with debugging information
                mse_array: MSE as a function of epochs
                time_array: Time as a function of epochs
                num_units_array: Number of active units as a function of epochs
                learn_indices_list: Active indices at the starting of each
                    scale of fitting
                nparams: Total number of parameters
                
        TODO: Return all model parameters
        WARNING: Fitting beyond 4096x4096 is currently not implemented
    '''
    H, W, nchan = config.imshape
    if sys.platform == 'win32':
        visualize = True
    else:
        visualize = False
    
    # Parameters to save across iterations
    learn_indices_list = []
    nparams = 0
    mse_array = np.zeros(config.epochs*(nscales+1))
    time_array = np.zeros(config.epochs*(nscales+1))
    num_units_array = np.zeros(config.epochs*(nscales+1))
    nparams_array = np.zeros(nscales)
    memory_array = np.zeros(nscales)
    
    prev_params = None
    im_estim = None
        
    # Memory usage helpers
    if sys.platform != 'win32':
        nvidia_smi.nvmlInit()
        mem_handle = nvidia_smi.nvmlDeviceGetHandleByIndex(5)
        
    if config.nonlin == 'wire':
        param_const = 0.5
    else:
        param_const = 0.5
    
    # Begin
    for scale_idx in tqdm.tqdm(range(nscales), dynamic_ncols=True):
        Ht, Wt = H//pow(2, nscales-scale_idx-1), W//pow(2, nscales-scale_idx-1)
        
        # Create folders and unfolders
        unfold = torch.nn.Unfold(kernel_size=config.ksize, stride=config.stride)
        fold = torch.nn.Fold(output_size=(Ht, Wt), kernel_size=config.ksize,
                            stride=config.stride)
   
        # Find out number of chunks
        nchunks = int(Ht*Wt/(config.ksize**2))   
        
        # Create model
        if scale_idx == 0:
            nfeat = config.nfeat*4
            lr_div = 4
        else:
            nfeat = config.nfeat
            lr_div = 1
                            
        # Create inputs
        coords_chunked = miner.get_coords((Ht, Wt), config.ksize,
                                    config.coordstype, unfold)

        if scale_idx < nscales-1:
            criterion = losses.L1Norm()
        else:
            criterion = losses.L2Norm()
        best_mse = float('inf')
        best_img = None
            
        tbar = tqdm.tqdm(range(config.epochs), dynamic_ncols=True)
        
        ret_list = get_next_signal(scale_idx, config, unfold, nchunks,
                                   im_estim, ndim=2, nchan=nchan,
                                   switch_std=switch_std)
        [im_chunked_prev, master_indices, learn_indices] = ret_list
        
        learn_indices_list.append(master_indices)
                
        # Make a copy of the parameters and then truncate  
        if scale_idx < 2:      
            model, prev_params = get_model(config,
                                           nchunks,
                                           nfeat,
                                           master_indices,
                                           None,
                                           scale_idx)
        else:
            model, prev_params = get_model(config,
                                           nchunks,
                                           nfeat,
                                           master_indices, prev_params,
                                           scale_idx)
                                
        # Truncate coordinates and training data
        coords_chunked = coords_chunked[master_indices, ...]
                    
        # Optimizer
        optim = torch.optim.Adam(lr=config.lr/lr_div,
                                 params=model.parameters())
                
        nparams += utils.count_parameters(model)
        nparams_array[scale_idx] = utils.count_parameters(model)
                
        prev_loss = 0
        
        # Move tensors to GPU
        coords_chunked = coords_chunked.cuda(non_blocking=True)
        im_chunked_prev = im_chunked_prev.cuda(non_blocking=True)
        learn_indices = learn_indices.cuda(non_blocking=True)
        master_indices = master_indices.cuda(non_blocking=True)
        
        for idx in tbar:             
            if learn_indices.numel() == 0:
                break
            lr = config.lr*learn_indices.numel()/master_indices.numel()
            weight = pow(0.1, idx/config.epochs)
            optim.param_groups[0]['lr'] = lr*weight
            
            optim.zero_grad()
            
            im_sub_chunked = model(coords_chunked,
                                    learn_indices).permute(2, 1, 0)
        
            tmp = torch.zeros_like(im_chunked_prev)
            tmp[..., master_indices[learn_indices]] = im_sub_chunked
            
            if config.propmethod == 'coarse2fine':
                im_estim = fold(tmp +\
                    im_chunked_prev)[:, 0, ...]
            else:
                im_estim = fold(tmp)[:, 0, ...]
                      
            if scale_idx < nscales-1:      
                im_estim_up = torch.nn.functional.interpolate(
                    im_estim[None, ...],
                    size=(H, W),
                    mode='bilinear'
                )[0, ...]
            else:
                im_estim_up = im_estim
        
            meas_estim = func(im_estim_up)
            
            loss = criterion(meas_estim, meas_ten)
            
            if config.denoising:
                param_norms = [torch.linalg.norm(p.flatten()) for p in model.parameters()]
                l1_loss = sum(param_norms)/len(param_norms)
                
                loss = loss + (nscales-scale_idx)*config.l1_pen*l1_loss
            
            loss.backward()
            optim.step()                
            lossval = loss.item()
            
            if scale_idx < nscales - 1:
                if abs(lossval - prev_loss) < config.switch_thres:
                    break
            prev_loss = lossval
                
            try:
                mse_array[scale_idx*config.epochs + idx] = lossval
            except ValueError:
                pdb.set_trace()
            time_array[scale_idx*config.epochs + idx] = time.time()
            
            num_units_array[scale_idx*config.epochs + idx] = learn_indices.numel()
            
            if lossval < best_mse:
                best_mse = lossval
                best_img = copy.deepcopy(im_estim.detach())
                
            if visualize:
                im_estim_cpu = im_estim.detach().cpu().permute(1, 2, 0).numpy()
                meas_estim_cpu = meas_estim.detach().cpu().squeeze().numpy()
                cv2.imshow('Estim', im_estim_cpu)
                cv2.imshow('Meas estim', meas_estim_cpu)
                cv2.waitKey(1)
            
            # Memory usage
            if sys.platform != 'win32':
                mem_info = nvidia_smi.nvmlDeviceGetMemoryInfo(mem_handle)
                mem_usage = mem_info.used/2**30
            else:
                mem_usage = 0.0
            
            if idx == 0:
                memory_array[scale_idx] = mem_usage
            
            tbar.set_description('[%d/%d | %.2f GB| %.4e]'%(
                learn_indices.numel(),
                nchunks,
                mem_usage,
                lossval))
            tbar.refresh()
        
        if scale_idx < nscales - 1:
            # Copy the model, we will need it to propagate parameters
            prev_params_sub = copy.deepcopy(model.state_dict())
            
            # First copy the new parameters
            for key in prev_params:
                prev_params[key][master_indices, ...] = prev_params_sub[key][...]
            
            # Now we will double up all the parameters
            indices = np.arange(nchunks).reshape(Ht//config.ksize,
                                                 Wt//config.ksize)
            indices = cv2.resize(indices, None, fx=2, fy=2,
                                interpolation=cv2.INTER_NEAREST).ravel()
            indices = torch.tensor(indices).cuda().long()
            
            for key in prev_params:
                prev_params[key] = prev_params[key][indices, ...]*param_const
                                       
        # Move tensors to CPU
        im_estim = im_estim.cpu()

        im_estim = im_estim.detach()
        save_img = np.clip(im_estim.permute(1, 2, 0).numpy(), 0, 1)
        cv2.imwrite('%s/scale%d.png'%(config.savedir, scale_idx),
                   (255*save_img).astype(np.uint8))
        
        imblock = miner.drawblocks_single(master_indices.cpu(), (Ht, Wt),
                                    (config.ksize, config.ksize))
        imblock = imblock.cpu().numpy()
        save_img = np.clip(im_estim.permute(1, 2, 0).numpy(), 0, 1)
        save_img = np.clip(save_img + imblock, 0, 1)
        cv2.imwrite('%s/scale%d_ann.png'%(config.savedir, scale_idx),
                   (255*save_img).astype(np.uint8))
                        
    # Remove zero valued entries
    indices, = np.where(time_array > 0)
    
    info = {'time_array': time_array[indices],
            'mse_array': mse_array[indices],
            'num_units_array': num_units_array[indices],
            'nparams': nparams,
            'nparams_array': nparams_array,
            'memory_array': memory_array,
            'learn_indices_list': learn_indices_list}
    
    return best_img.cpu().permute(1, 2, 0).numpy(), info

def miner_volfit(im, nscales, switch_mse, stopping_mse, config):
    '''
        Multiscale Neural Implicit Representation (MINER) fitting for 3d volumes
        
        Inputs:
            im: (H, W, T) Volume to fit. 
            nscales: Number of scales to fit
            switch_mse: Blocks are terminated when this MSE is achieved
            stopping_mse: Fitting is halted if this MSE is achieved. Set to 0
                to run for all iterations
            config: Namespace with configuration variables. Check configs folder
                for more examples
                
        Outputs:
            imfit: Final fitted volume
            info: Dictionary with debugging information
                mse_array: MSE as a function of epochs
                time_array: Time as a function of epochs
                num_units_array: Number of active units as a function of epochs
                learn_indices_list: Active indices at the starting of each
                    scale of fitting
                nparams: Total number of parameters
                
        TODO: Return all model parameters
    '''
    # Memory usage helpers
    if sys.platform != 'win32':
        nvidia_smi.nvmlInit()
        mem_handle = nvidia_smi.nvmlDeviceGetHandleByIndex(4)
        
    H, W, T = im.shape
    if sys.platform == 'win32':
        visualize = True
    else:
        visualize = False

    # Send image to CUDA
    imten = torch.tensor(im)[None, ...]
    
    max_size = 256
    
    if H*W*T <= max_size**3 + 1:
        imten_gpu = imten.cuda()
        
    # Parameters to save across iterations
    learn_indices_list = []
    nparams = 0
    mse_array = np.zeros(config.epochs*(nscales+1))
    time_array = np.zeros(config.epochs*(nscales+1))
    num_units_array = np.zeros(config.epochs*(nscales+1))
    
    nparams_array = np.zeros(nscales)
    mse_epoch_array = np.zeros(nscales)
    time_epoch_array = np.zeros(nscales)
    mem_array = np.zeros(nscales)
    
    prev_params = None
    im_estim = None
    
    # Begin
    mse_full = 0
    init_epoch_time = time.time()
    save_mesh_every = 1000
    global_time = 0
    last_save_time = 0
    for scale_idx in tqdm.tqdm(range(nscales), dynamic_ncols=True):
        tic = time.time()
        imtarget_ten = torch.nn.functional.interpolate(imten[None, ...],
                                    scale_factor=pow(2, -nscales+scale_idx+1),
                                    mode='area')
        _, _, Ht, Wt, Tt = imtarget_ten.shape
        
        #if Ht*Wt*Tt > 256**3 + 1:
        #    config.maxchunks = 512
        
        # Create folders and unfolders
        unfold = unfoldNd.UnfoldNd(kernel_size=config.ksize,
                                   stride=config.stride)
        fold = unfoldNd.FoldNd(output_size=(Ht, Wt, Tt),
                               kernel_size=config.ksize,
                               stride=config.stride)
   
        # Find out number of chunks
        nchunks = int(Ht*Wt*Tt/(config.ksize**3))   
        
        # Create model
        if scale_idx == 0:
            if config.propmethod == 'coarse2fine':
                nfeat = config.nfeat*4
                lr_div = 4
                start_carry = 2
            else:
                nfeat = config.nfeat*4
                lr_div = 4
                start_carry = 2
        else:
            nfeat = config.nfeat
            lr_div = 1
                            
        # Create inputs
        coords_chunked = miner.get_coords((Ht, Wt, Tt), config.ksize,
                                    config.coordstype, unfold)
                            
        imten_chunked = unfold(imtarget_ten)
        
        #if config.signaltype == 'occupancy':
        if config.loss == 'logistic':
            criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
        else:
            criterion = losses.L2Norm()
        best_mse = float('inf')
        best_img = None
            
        tbar = tqdm.tqdm(range(config.epochs), dynamic_ncols=True)
        
        try:
            ret_list = get_next_signal(imtarget_ten, scale_idx, config, unfold, 
                                    switch_mse, nchunks, im_estim, ndim=3)
        except RuntimeError:
            pdb.set_trace()
        [imten_chunked, im_chunked_prev, im_chunked_res_frozen, 
                master_indices, learn_indices, loss_frozen] = ret_list
                
        learn_indices_list.append(master_indices)
                                
        # Make a copy of the parameters and then truncate              
        if scale_idx < start_carry:      
            model, prev_params = get_model(config, nchunks, nfeat,
                                           master_indices, None, scale_idx)
        else:
            model, prev_params = get_model(config, nchunks, nfeat,
                                           master_indices, prev_params,
                                           scale_idx)
                                
        # Truncate coordinates and training data
        coords_chunked = coords_chunked[master_indices, ...]
        imten_chunked = imten_chunked[..., master_indices]
                    
        # Optimizer
        optim = torch.optim.Adam(lr=config.lr/lr_div,
                                 params=model.parameters())
                
        nparams += utils.count_parameters(model)
        nparams_array[scale_idx] = utils.count_parameters(model)
                
        prev_loss = 0
        
        if config.loss == 'logistic':
            signal_norm = torch.ones(imten_chunked.shape[-1], device='cuda')
        else:
            signal_norm = (imten_chunked**2).mean(0, keepdim=True)
            signal_norm = signal_norm.mean(1, keepdim=True).flatten()
        
        # Move tensors to GPU
        coords_chunked = coords_chunked.cuda(non_blocking=True)
        imten_chunked = imten_chunked.cuda(non_blocking=True)
        
        if Ht*Wt*Tt < max_size**3 + 1:
            im_chunked_prev = im_chunked_prev.cuda(non_blocking=True)
            im_chunked_res_frozen = im_chunked_res_frozen.cuda(non_blocking=True)
        learn_indices = learn_indices.cuda(non_blocking=True)
        master_indices = master_indices.cuda(non_blocking=True)
        signal_norm = signal_norm.cuda(non_blocking=True)
        
        init_time = time.time() - tic
        
        for idx in tbar: 
            tic = time.time()            
            if learn_indices.numel() == 0:
                break
            lr = config.lr*learn_indices.numel()/master_indices.numel()
            
            if scale_idx == 0:
                weight = pow(0.999, idx)
            else:
                weight = pow(0.999, idx)/pow(1.2, nscales)
                
            if config.loss == 'logistic':
                term1 = (1 - 0.9*np.exp(-idx/pow(1.2, scale_idx)))
                if Ht*Wt*Tt > 256**3 + 1:
                    term2 = pow(2, scale_idx)
                else:
                    term2 = pow(1.5, scale_idx)
                term3 = pow(0.999, idx)
                weight = term1*term3/term2
            
            optim.param_groups[0]['lr'] = lr*weight
            
            if learn_indices.numel() > config.maxchunks:
                im_sub_chunked_list = []
                npix = config.ksize**3
                im_sub_chunked = torch.zeros((3, npix, learn_indices.numel()),
                                             device='cuda')
                lossval = 0
                weight = 0
                for sub_idx in range(0, learn_indices.numel(), config.maxchunks):
                    sub_idx2 = min(learn_indices.numel(),
                                   sub_idx+config.maxchunks)
                    learn_indices_sub = learn_indices[sub_idx:sub_idx2]
                    
                    optim.zero_grad(set_to_none=True)
            
                    im_sub_chunked_sub = model(coords_chunked,
                                            learn_indices_sub).permute(2, 1, 0)
                        
                    loss = criterion(im_sub_chunked_sub,
                                     imten_chunked[..., learn_indices_sub])
                    
                    loss.mean().backward()                    
                    optim.step()
                
                    lossval += loss.mean().item()    
                    weight += 1
                    
                    with torch.no_grad():
                        im_sub_chunked_list.append(im_sub_chunked_sub)
                                                
                lossval /= weight
                with torch.no_grad():
                    im_sub_chunked = torch.cat(im_sub_chunked_list, 2)

            else:
                optim.zero_grad(set_to_none=True)
                
                im_sub_chunked = model(coords_chunked,
                                      learn_indices).permute(2, 1, 0)
                loss = criterion(im_sub_chunked,
                            imten_chunked[..., learn_indices]).mean()
                                
                loss.backward()
                optim.step()                
                lossval = loss.item()
                
            with torch.no_grad():
                if config.loss == 'logistic':
                    chunk_err = criterion(im_sub_chunked,
                                          imten_chunked[..., learn_indices])
                else:
                    chunk_err = (im_sub_chunked -\
                        imten_chunked[..., learn_indices])**2
                chunk_err = chunk_err.mean(1, keepdim=True).mean(0, keepdim=True)
                
                # Freeze sections that have achieved target MSE
                learn_indices_sub, = torch.where(chunk_err.flatten() > \
                    switch_mse*signal_norm[learn_indices])
                
                new_indices = master_indices[learn_indices[learn_indices_sub]]
                
                if Ht*Wt*Tt >= max_size**3 + 1:
                    im_sub_chunked = im_sub_chunked.cpu()
                    new_indices = new_indices.cpu()
                    learn_indices_sub = learn_indices_sub.cpu()
                im_chunked_res_frozen[..., new_indices] = \
                    im_sub_chunked[..., learn_indices_sub]
                    
                epoch_time = time.time() - tic
                    
                if config.propmethod == 'coarse2fine':
                    if Ht*Wt*Tt < max_size**3 + 1:
                        im_chunked_full = im_chunked_res_frozen + im_chunked_prev
                        im_estim = fold(im_chunked_full)[:, 0, ...]
                        
                    else:
                        if idx%100 == 0:
                            im_chunked_full = im_chunked_res_frozen +\
                                im_chunked_prev
                            im_estim = fold(im_chunked_full)[:, 0, ...]
                else:
                    im_estim = fold(im_chunked_res_frozen)[:, 0, ...]
                 
                if H*W*T < max_size**3 + 1:      
                    if scale_idx < nscales - 1:        
                        im_estim_up = torch.nn.functional.interpolate(
                            im_estim[None, ...], size=(H, W, T), mode='trilinear')
                    else:
                        im_estim_up = im_estim[None, ...]
                        
                    if config.loss == 'logistic':
                        im_estim_up = torch.sigmoid(im_estim_up)
                        
                    if config.signaltype == 'occupancy':
                        mse_full = volutils.get_IoU_batch(
                                im_estim_up, imten_gpu,
                                config.mcubes_thres)
                    else:
                        mse_full = ((imten_gpu - im_estim_up)**2).mean().item()
                else:
                    if idx%100 == 0:
                        if scale_idx < nscales - 1:        
                            im_estim_up = torch.nn.functional.interpolate(
                                im_estim[None, ...].cpu(), size=(H, W, T), mode='trilinear')
                        else:
                            im_estim_up = im_estim[None, ...].cpu()
                            
                        if config.loss == 'logistic':
                            im_estim_up = torch.sigmoid(im_estim_up)
                            
                        if config.signaltype == 'occupancy':
                            mse_full = volutils.get_IoU_batch(
                                    im_estim_up, imten,
                                    config.mcubes_thres)
                        else:
                            mse_full = ((imten - im_estim_up)**2).mean().item()
                    # Do not updated too often, it takes up so much time!
                
            # Update indices to learn
            learn_indices = learn_indices[learn_indices_sub]
            
            if scale_idx < nscales - 1:
                if abs(lossval - prev_loss) < config.switch_thres:
                    break
            prev_loss = lossval
            
            descloss = mse_full
            
            total_time = epoch_time + (idx == 0)*init_time
            
            mse_array[scale_idx*config.epochs + idx] = descloss
            time_array[scale_idx*config.epochs + idx] = total_time
                        
            if lossval < best_mse:
                best_mse = lossval
            
            if config.signaltype == 'occupancy':
                if mse_full > stopping_mse:
                    break
            else:    
                if mse_full < stopping_mse:
                    break
                
            if visualize:
                im_estim_cpu = im_estim[..., idx%Tt].squeeze().detach().cpu()
                cv2.imshow('Estim', im_estim_cpu.numpy())
                cv2.imshow('GT', im[..., idx%Tt])
                cv2.waitKey(1)
                
            global_time = global_time + total_time
            
            if config.signaltype == 'occupancy':
                if global_time > last_save_time + save_mesh_every:
                    last_save_time = global_time

                    cube = im_estim.cpu().squeeze().numpy()
                    volutils.march_and_save(cube, config.mcubes_thres, 
                                            '%s/time_%.2f.dae'%(config.savedir, global_time),
                                            True)            
            
            # Memory usage
            if sys.platform != 'win32':
                mem_info = nvidia_smi.nvmlDeviceGetMemoryInfo(mem_handle)
                mem_usage = mem_info.used/2**30
            else:
                mem_usage = 0.0
                
            if idx == 0:
                mem_array[scale_idx] = mem_usage
            tbar.set_description('[%d/%d | %.1f | %.4e]'%(learn_indices.numel(),
                                                   nchunks,
                                                   mem_usage,
                                                   descloss))
            tbar.refresh()
            
        mse_epoch_array[scale_idx] = mse_full
        #time_epoch_array[scale_idx] = time.time()
        time_epoch_array[scale_idx] = global_time
        
        if scale_idx < nscales - 1:
            # Copy the model, we will need it to propagate parameters
            prev_params_sub = copy.deepcopy(model.state_dict())
            
            # First copy the new parameters
            for key in prev_params:
                prev_params[key][master_indices, ...] = prev_params_sub[key][...]
            
            # Now we will double up all the parameters
            indices = np.arange(nchunks).reshape(Ht//config.ksize,
                                                 Wt//config.ksize,
                                                 Tt//config.ksize)
            indices = torch.tensor(indices)[None, None, ...]
            indices = torch.nn.functional.interpolate(indices.float(),
                                                      scale_factor=2,
                                                      mode='nearest')
            indices = indices.flatten().cuda().long()
            
            for key in prev_params:
                if config.propmethod == 'coarse2fine':
                    div = np.sqrt(8)
                else:
                    div = 1
                prev_params[key] = prev_params[key][indices, ...]/div
                                       
        # Move tensors to CPU
        im_estim = im_estim.cpu()
        
        # Save data per iteration
        if config.signaltype == 'occupancy':
            cube = im_estim.squeeze().numpy()
            volutils.march_and_save(cube, config.mcubes_thres, 
                                    '%s/scale%d.dae'%(config.savedir, scale_idx),
                                    True)
            miner.drawcubes_single(master_indices, (Ht, Wt, Tt), config.ksize,
                            '%s/scale%d.png'%(config.savedir, scale_idx))
            torch.save(model.state_dict(),
                       'model%d.pth'%scale_idx)
                    
    # Remove zero valued entries
    indices, = np.where(time_array > 0)
    
    info = {'time_array': np.cumsum(time_array[indices]),
            'mse_array': mse_array[indices],
            'mse_epoch_array': mse_epoch_array,
            'mem_array': mem_array,
            'time_epoch_array': time_epoch_array,
            'num_units_array': num_units_array[indices],
            'nparams': nparams,
            'nparams_array': nparams_array,
            'learn_indices_list': learn_indices_list}
    
    #if config.signaltype == 'occupancy':
    if config.loss == 'logistic':
        best_img = im_estim
        best_img = torch.sigmoid(best_img)
    
    return im_estim.squeeze().cpu().numpy(), info

def miner_fixed_volfit(im, stopping_mse, config):
    '''
        Fixed Neural Implicit Representation fitting for 3d volumes --
        aka KiloNeRF
        
        Inputs:
            im: (H, W, T) Volume to fit
            switch_mse: Blocks are terminated when this MSE is achieved
            stopping_mse: Fitting is halted if this MSE is achieved. Set to 0
                to run for all iterations
            config: Namespace with configuration variables. Check configs folder
                for more examples
                
        Outputs:
            imfit: Final fitted volume
            info: Dictionary with debugging information
                mse_array: MSE as a function of epochs
                time_array: Time as a function of epochs
                num_units_array: Number of active units as a function of epochs
                nparams: Total number of parameters
                
        TODO: Return all model parameters
    '''
    tic = time.time()
    if sys.platform == 'win32':
        visualize = True
    else:
        visualize = False

    H, W, T = im.shape
    imten = torch.tensor(im).cuda()
    
    # Create folders and unfolders
    unfold = unfoldNd.UnfoldNd(kernel_size=config.ksize, stride=config.stride)
    fold = unfoldNd.FoldNd(output_size=(H, W, T), kernel_size=config.ksize,
                           stride=config.stride)
        
    # Find out number of chunks
    nchunks = int(H*W*T/config.ksize**3)
    
    # Create model
    model = siren.AdaptiveMultiSiren(in_features=config.in_features,
                                     out_features=config.out_features, 
                                     n_channels=nchunks,
                                     hidden_features=config.nfeat, 
                                     hidden_layers=config.nlayers,
                                     outermost_linear=True,
                                     first_omega_0=config.omega_0,
                                     hidden_omega_0=config.omega_0).cuda()
    nparams = utils.count_parameters(model)
    
    # Optimizer
    optim = torch.optim.Adam(lr=config.lr, params=model.parameters())
    
    # Criteria
    #if config.signaltype == 'occupancy':
    if config.loss == 'logistic':
        criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
    else:
        criterion = losses.L2Norm()
    
    # Create inputs
    coords_chunked = miner.get_coords([H, W, T], config.ksize, config.coordstype, 
                                unfold).cuda()
        
    imten_chunked = unfold(imten[None, None, ...])
        
    mse_array = np.zeros(config.epochs)
    time_array = np.zeros(config.epochs)
    num_units_array = np.zeros(config.epochs)
    best_mse = float('inf')
    best_img = None

    tbar = tqdm.tqdm(range(config.epochs))
    learn_indices = torch.arange(nchunks, device='cuda')
    
    im_chunked_frozen = torch.zeros(1, config.ksize**3, nchunks, device='cuda')
    
    #if config.signaltype == 'occupancy':
    if False:
        signal_err = torch.zeros(imten_chunked.shape[-1], device='cuda')
    else:
        with torch.no_grad():
            signal_err = (imten_chunked)**2
            signal_err = signal_err.mean(0, keepdim=True).mean(1, keepdim=True)
            signal_err = signal_err.flatten()
            
    init_time = time.time() - tic
    
    for idx in tbar:
        tic = time.time()
        if learn_indices.numel() == 0:
            break
        lr = pow(0.998, idx)*config.lr*learn_indices.numel()/nchunks
        optim.param_groups[0]['lr'] = lr
        
        im_sub_chunked_list = []
        for b_idx in range(0, learn_indices.numel(), config.maxchunks):
            b_idx2 = min(learn_indices.numel(), b_idx+config.maxchunks)
            learn_indices_sub = learn_indices[b_idx:b_idx2]
            im_sub_chunked_sub = model(coords_chunked,
                                   learn_indices_sub).permute(2, 1, 0)
            
            loss = criterion(im_sub_chunked_sub,
                             imten_chunked[..., learn_indices_sub])
        
            #if config.signaltype == 'occupancy':
            if config.loss == 'logistic':
                loss = loss.mean()
            
            optim.zero_grad()
            loss.backward()
            optim.step()
            
            with torch.no_grad():
                im_sub_chunked_list.append(im_sub_chunked_sub)
        
        with torch.no_grad():
            im_sub_chunked = torch.cat(im_sub_chunked_list, 2)
        
        with torch.no_grad():
            #if config.signaltype == 'occupancy':
            if config.loss == 'logistic':
                chunk_err_list = []
                
                for b_idx in range(0, learn_indices.numel(), config.maxchunks):
                    b_idx2 = min(learn_indices.numel(), b_idx+config.maxchunks)
                    learn_indices_sub = learn_indices[b_idx:b_idx2]
                    chunk_err_sub = criterion(im_sub_chunked[..., b_idx:b_idx2],
                                     imten_chunked[..., learn_indices_sub])
                    
                    chunk_err_list.append(chunk_err_sub.mean(1, keepdim=True))
                    
                chunk_err = torch.cat(chunk_err_list, -1)
            else:
                chunk_err = ((im_sub_chunked -\
                    imten_chunked[..., learn_indices])**2).mean(1, keepdim=True)
            
            # Freeze sections that have achieved target MSE
            learn_indices_sub, = torch.where(chunk_err.flatten() >\
                stopping_mse*signal_err[learn_indices])
            
            im_chunked_frozen[..., learn_indices[learn_indices_sub]] = \
                im_sub_chunked[..., learn_indices_sub]
            im_estim = fold(im_chunked_frozen)
        
        # Update indices to learn
        num_units_array[idx] = learn_indices.numel()
        learn_indices = learn_indices[learn_indices_sub]
                
        epoch_time = time.time() - tic

        lossval = loss.item()
        if config.loss == 'logistic':
            im_estim = torch.sigmoid(im_estim)
        if config.signaltype == 'occupancy':
            with torch.no_grad():
                mse_array[idx] = volutils.get_IoU_batch(
                    im_estim.squeeze(), imten,
                    config.mcubes_thres)
        else:
            mse_array[idx] = lossval
        time_array[idx] = epoch_time + (idx == 0)*init_time
        
        if config.signaltype != 'occupancy':
            if lossval < best_mse:
                best_mse = lossval
                best_img = copy.deepcopy(torch.sigmoid(im_estim).detach())
                
            if lossval < stopping_mse:
                best_img = copy.deepcopy(torch.sigmoid(im_estim).detach())
                break
        else:
            best_img = copy.deepcopy(im_estim.detach())
            
        if visualize:
            im_rec = im_estim[..., idx%T].squeeze().detach().cpu()
            cv2.imshow('GT', im[..., idx%T])
            cv2.imshow('Rec', im_rec.numpy())
            cv2.waitKey(1)
        
        tbar.set_description('[%d | %.4e]'%(learn_indices.numel(),
                                            mse_array[idx]))
        tbar.refresh()
        
    # Remove zero valued entries
    indices, = np.where(time_array > 0)
    
    info = {'time_array': np.cumsum(time_array[indices]),
            'mse_array': mse_array[indices],
            'num_units_array': num_units_array[indices],
            'nparams': nparams}
    
    return best_img.squeeze().cpu().numpy(), info

def miner_fixed_imfit(im, stopping_mse, config):
    '''
        Fixed Neural Implicit Representation fitting for images -- aka KiloNeRF
        
        Inputs:
            im: (H, W, 3) Image to fit. 
            switch_mse: Blocks are terminated when this MSE is achieved
            stopping_mse: Fitting is halted if this MSE is achieved. Set to 0
                to run for all iterations
            config: Namespace with configuration variables. Check configs folder
                for more examples
                
        Outputs:
            imfit: Final fitted image
            info: Dictionary with debugging information
                mse_array: MSE as a function of epochs
                time_array: Time as a function of epochs
                num_units_array: Number of active units as a function of epochs
                nparams: Total number of parameters
                
        TODO: Return all model parameters
        WARNING: Fitting beyond 4096x4096 is currently not implemented
    '''
    
    if sys.platform == 'win32':
        visualize = True
    else:
        visualize = False

    H, W, _ = im.shape
    imten = torch.tensor(im).cuda().permute(2, 0, 1)
    
    # Times obtained from MINER training
    #save_times = [2, 3, 9, 24, 50]
    save_times = [2]
    
    # Create folders and unfolders
    unfold = torch.nn.Unfold(kernel_size=config.ksize, stride=config.stride)
    fold = torch.nn.Fold(output_size=(H, W), kernel_size=config.ksize,
                         stride=config.stride)
        
    # Find out number of chunks
    nchunks = unfold(torch.zeros(1, 1, H, W)).shape[-1]    
    
    # Create model
    model = siren.AdaptiveMultiSiren(in_features=config.in_features,
                                     out_features=config.out_features, 
                                     n_channels=nchunks,
                                     hidden_features=config.nfeat, 
                                     hidden_layers=config.nlayers,
                                     outermost_linear=True,
                                     first_omega_0=config.omega_0,
                                     hidden_omega_0=config.omega_0).cuda()
    nparams = utils.count_parameters(model)
    
    # Optimizer
    optim = torch.optim.Adam(lr=config.lr, params=model.parameters())
    
    # Criteria
    criterion = losses.L2Norm()
    
    # Create inputs
    coords_chunked = miner.get_coords([H, W], config.ksize, config.coordstype,
                                      unfold)
    coords_chunked = coords_chunked.cuda()
    
    imten_chunked = unfold(imten[:, None, ...])
        
    mse_array = np.zeros(config.epochs)
    time_array = np.zeros(config.epochs)
    num_units_array = np.zeros(config.epochs)
    best_mse = float('inf')
    best_img = None

    tbar = tqdm.tqdm(range(config.epochs))
    learn_indices = torch.arange(nchunks, device='cuda')
    
    init_time = time.time()
    cur_step = 0
    save_psnr = np.zeros(len(save_times))
    save_actual_times = np.zeros(len(save_times))
    
    im_chunked_frozen = torch.zeros(3, config.ksize**2, nchunks, device='cuda')
    
    if visualize:
        cv2.imshow('GT', im)
        cv2.waitKey(1)
        
    with torch.no_grad():
        signal_err = (imten_chunked)**2
        signal_err = signal_err.mean(0, keepdim=True).mean(1, keepdim=True)
        signal_err = signal_err.flatten()
        
    # Memory usage helpers
    nvidia_smi.nvmlInit()
    mem_handle = nvidia_smi.nvmlDeviceGetHandleByIndex(7)
    
    for idx in tbar:
        tic = time.time()
        
        if learn_indices.numel() == 0:
            break
        lr = pow(0.999, idx)*config.lr*learn_indices.numel()/nchunks
        optim.param_groups[0]['lr'] = lr
        
        if learn_indices.numel() < config.maxchunks:
            optim.zero_grad()
                
            im_sub_chunked = model(coords_chunked,
                                    learn_indices).permute(2, 1, 0)
            loss = criterion(im_sub_chunked,
                        imten_chunked[..., learn_indices])
            
            loss.backward()
            optim.step()                
            lossval = loss.item()
            
        else:
            im_sub_chunked_list = []
            npix = config.ksize**2
            im_sub_chunked = torch.zeros((3, npix, learn_indices.numel()),
                                          device='cuda')
            lossval = 0
            weight = 0
            for sub_idx in range(0, learn_indices.numel(), config.maxchunks):
                sub_idx2 = min(learn_indices.numel(),
                                sub_idx+config.maxchunks)
                learn_indices_sub = learn_indices[sub_idx:sub_idx2]
                
                optim.zero_grad()
        
                im_sub_chunked_sub = model(coords_chunked,
                                        learn_indices_sub).permute(2, 1, 0)
                    
                loss = criterion(im_sub_chunked_sub,
                                 imten_chunked[..., learn_indices_sub]) 
                
                loss.backward()                    
                optim.step()
            
                lossval += loss.item()    
                weight += 1
                
                with torch.no_grad():
                    im_sub_chunked_list.append(im_sub_chunked_sub)
                    
            lossval /= weight
            with torch.no_grad():
                im_sub_chunked = torch.cat(im_sub_chunked_list, 2)
        
        with torch.no_grad():
            chunk_err = ((im_sub_chunked - imten_chunked[..., learn_indices])**2)
            chunk_err = chunk_err.mean(0, keepdim=True).mean(1, keepdim=True)
            
            # Freeze sections that have achieved target MSE
            learn_indices_sub, = torch.where(chunk_err.flatten() >\
                stopping_mse*signal_err[learn_indices])
            
            im_chunked_frozen[..., learn_indices[learn_indices_sub]] = \
                im_sub_chunked[..., learn_indices_sub]
            im_estim = fold(im_chunked_frozen).squeeze()[None, ...]
                
        # Update indices to learn
        num_units_array[idx] = learn_indices.numel()
        learn_indices = learn_indices[learn_indices_sub]
        
        im_estim_cpu = im_estim.squeeze().detach().permute(1, 2, 0).cpu().numpy()
        
        # Memory usage
        mem_info = nvidia_smi.nvmlDeviceGetMemoryInfo(mem_handle)
        mem_usage = mem_info.used/2**30
        
        if idx == 0:
            mem_usage_init = mem_usage
        
        mse_array[idx] = lossval
        
        if idx > 0:
            time_array[idx] = time.time() - tic + time_array[idx-1]
        else:
            time_array[idx] = time.time() - tic
        
        if lossval < best_mse:
            best_mse = lossval
            best_img = im_estim_cpu
            
        if lossval < stopping_mse:
            best_img = im_estim_cpu
            break
        
        if config.save_during:
            if cur_step < len(save_times):
                if time_array[idx] -  save_times[cur_step] > 0:
                    plt.imsave('../results/%s/kilonerf_T_%d.png'%(
                        config.expname, np.ceil(time_array[idx])),
                            np.clip(best_img[..., ::-1], 0, 1))
                    save_psnr[cur_step] = best_mse
                    save_actual_times[cur_step] = time_array[idx]
                    cur_step += 1
            
        if visualize:
            cv2.imshow('Estim', im_estim_cpu)
            cv2.waitKey(1)
        
        tbar.set_description('[%d | %.2f | %.4e]'%(
            learn_indices.numel(),
            mem_usage,
            lossval))
        tbar.refresh()
        
    # Remove zero valued entries
    indices, = np.where(time_array > 0)
    
    info = {'time_array': time_array[indices],
            'mse_array': mse_array[indices],
            'num_units_array': num_units_array[indices],
            'save_psnr': save_psnr,
            'save_actual_times': save_actual_times,
            'memory': mem_usage_init,
            'nparams': nparams}
    
    return best_img, info

@torch.no_grad()
def get_model(config, nchunks, nfeat, master_indices=None, 
              prev_params=None, scale_idx=0):
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
        if scale_idx == -1:
            omega = 5
            scale = 5
        else:
            omega = config.omega_0
            scale = config.scale
        model = wire.AdaptiveMultiWIRE(
            in_features=config.in_features,
            out_features=config.out_features, 
            n_channels=nchunks,
            hidden_features=nfeat, 
            hidden_layers=config.nlayers,
            outermost_linear=True,
            first_omega_0=omega,
            hidden_omega_0=5.0,
            scale=scale,
            const=const
        ).cuda()
    else:
        model = siren.AdaptiveMultiSiren(in_features=config.in_features,
                                        out_features=config.out_features, 
                                        n_channels=nchunks,
                                        hidden_features=nfeat, 
                                        hidden_layers=config.nlayers,
                                        outermost_linear=True,
                                        first_omega_0=config.omega_0,
                                        hidden_omega_0=config.omega_0,
                                        nonlin=config.nonlin,
                                        const=const).cuda()
        
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

@torch.no_grad()
def get_next_signal(scale_idx, config, unfold, nchunks, im_estim=None, ndim=2,
                    nchan=1, switch_std=1e-3):
    '''
        Get the signal for next scale
        
        Inputs:
            imtarget_ten: (1, nchan, H, W) image to fit
            scale_idx: Scale of fitting
            config: Configuration structure
            unfold: Unfolding operator
            im_estim: estimate from previous fitting. Ignored if scale_idx is 0
            ndim: Number of dimensions of the signal. 2 for images and 3 for
                volumetric data
            nchan: number of channels
    '''
    if ndim == 2:
        mode = 'bilinear'
    else:
        mode = 'trilinear'
    
    if scale_idx == 0:
        # For first scale, we need to use up all indices. Next scale is
        # decided by error from upsampling
        learn_indices = torch.arange(nchunks)
        master_indices = torch.arange(nchunks)
        
        im_chunked_prev = torch.zeros(nchan, config.ksize**ndim, nchunks)
    else:
        # Upsample previous reconstruction
        im_estim_prev = torch.nn.functional.interpolate(im_estim[None, ...], 
                                                        scale_factor=2,
                                                        mode=mode)
        
        if ndim == 2:
            im_chunked_prev = unfold(im_estim_prev.permute(1, 0, 2, 3))
        else:
            im_chunked_prev = unfold(im_estim_prev)
                        
        # If the energy in residue is very small, do not learn anything
        # over that patch
        chunk_err = im_chunked_prev.std(1, keepdim=True)
        chunk_err = chunk_err.mean(0, keepdim=True)
        master_indices, = torch.where(chunk_err.flatten() > switch_std)
        
        # Since we are truncating the model, we will have learn
        # indices as all 1 to numel
        if master_indices.numel() == 0:
            # Nominal number of blocks
            master_indices = torch.tensor([0, 1])
            learn_indices = torch.tensor([0, 1])
        else:
            learn_indices = torch.arange(master_indices.numel())
        
    ret_list = [im_chunked_prev, master_indices, learn_indices] 
        
    return ret_list