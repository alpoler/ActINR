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
import math
import numpy as np
from scipy import io
from skimage.metrics import structural_similarity as ssim_func

import cv2
import torch
from torch.optim.lr_scheduler import ExponentialLR
from torch.nn import Parameter
from dataset_class import VideoDataset, BalancedSampler 
import folding_utils as unfoldNd

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
plt.gray()

import utils
import siren
import losses
import volutils
import wire
import models

utils = importlib.reload(utils)
siren = importlib.reload(siren)
losses = importlib.reload(losses)
volutils = importlib.reload(volutils)
wire = importlib.reload(wire)
models = importlib.reload(models)





def kilonerf(im, stopping_mse, config, model=None,
             bias_only=False, warm_start=False,
             weights=None):
    '''
        Fixed Neural Implicit Representation fitting for images -- aka KiloNeRF
        
        kilonerf2 evaluates loss based on reconstructed image, not on individual patches.
        
        Inputs:
            im: (H, W, 3) Image to fit. 
            switch_mse: Blocks are terminated when this MSE is achieved
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
                num_units_array: Number of active units as a function of epochs
                nparams: Total number of parameters
                
        TODO: Return all model parameters
        WARNING: Fitting beyond 4096x4096 is currently not implemented
    '''
    
    if sys.platform == 'win32':
        visualize = True
    else:
        visualize = True

    if not hasattr(config, 'save_epochs'):
        config.save_epochs = False

    H, W, _ = im.shape
    imten = torch.tensor(im).cuda().permute(2, 0, 1)
    
    # Create folders and unfolders
    unfold = torch.nn.Unfold(kernel_size=config.ksize, stride=config.stride)
    fold = torch.nn.Fold(output_size=(H, W), kernel_size=config.ksize,
                         stride=config.stride)
    
    # Find out number of chunks
    weighing = torch.ones(1, 1, H, W)
    nchunks = unfold(weighing).shape[-1]    
    
    # Create model
    if model is None:
        model = models.get_model(config, nchunks)
            
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
    coords_chunked = utils.get_coords([H, W],
                                      config.ksize,
                                      config.coordstype,
                                      unfold)
    coords_chunked = coords_chunked.cuda()
        
    mse_array = np.zeros(config.epochs)
    time_array = np.zeros(config.epochs)
    best_mse = float('inf')
    best_img = None
    
    if config.save_epochs:
        imstack = np.zeros((config.epochs, H, W, 3),
                           dtype=np.uint8)
    else:
        imstack = np.zeros(1)
    
    learn_indices = torch.arange(nchunks, device='cuda')

    tbar = tqdm.tqdm(range(config.epochs))
    
    if visualize:
        cv2.imshow('GT', im)
        cv2.waitKey(1)
        
    for idx in tbar:
        tic = time.time()
        lr = config.lr*pow(0.1, idx/config.epochs)
        
        if bias_only:
            lr = lr*(1-np.exp(-idx*0.05))
            
        optim.param_groups[0]['lr'] = lr
        
        optim.zero_grad()
        
        im_estim_chunked = model(coords_chunked, learn_indices).permute(2, 1, 0)
        
        im_estim = fold(im_estim_chunked).squeeze()[None, ...]
        
        loss = criterion(im_estim, imten)
        loss.backward()
        optim.step()
        lossval = loss.item()
        
        epoch_time = time.time() - tic
        
        im_estim_cpu = im_estim.squeeze().detach().permute(1, 2, 0).cpu().numpy()
        
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

def single_inrt(im_list, nscales, stopping_mse, config, pretrained_models=None, temporal_weightshare=None):
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
    
    params = []
    nparams_array = np.zeros(nimg)
    
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
            hidden_omega_0=config.omega_0,
            pos_encode=True
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
            nonlin_type=config.nonlin_type,
            const=1.0
        ).cuda()
    
    params+= list(model.parameters())
    nparams_array[0] = utils.count_parameters(model)

    if pretrained_models is not None:
        model.load_state_dict(pretrained_models[0])
    
    #model_state_dict = model.state_dict()    
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
    
    #print(coords_chunked.shape)
    t_coords = torch.ones(coords_chunked.shape[0], coords_chunked.shape[1],1).cuda()
        
    mse_array = np.zeros(config.epochs)
    best_mse = float('inf')
    best_img = None
    
    learn_indices = torch.arange(nchunks, device='cuda')

    tbar = tqdm.tqdm(range(config.epochs))
    
    for idx in tbar:
        tic = time.time()

        #if config.nonlin=='wire':
        lr = config.lr*pow(0.1, idx/config.epochs)
        optim.param_groups[0]['lr'] = lr
        
        optim.zero_grad()
        
        im_estim_list = []
        
        for m_idx in range(nimg):
            t_idx = t_coords * (m_idx - nimg/2) / (nimg/2)
            inp_coords = torch.cat([coords_chunked, t_idx], dim=2)
            
            im_out = model(inp_coords, learn_indices)
            im_estim_list.append(
                im_out.permute(2, 1, 0))
        
        im_estim_chunked = torch.cat(im_estim_list, 0)
        
        im_estim = fold(im_estim_chunked).reshape(nimg, -1, H, W)
        
        if config.warm_start and idx < config.warm_epochs:
            loss = criterion(im_estim[0], imten[0])
        else:
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
        
    psnr_array = -10*np.log10(np.array(mse_list))
    info = {'psnr_array': psnr_array,
            'nparams_array': nparams_array}
    
    if nimg == 2:
        info['imdiff'] = im_diff_cpu
    
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

    #nimg = len(im_list)
    #H, W, _ = im_list[0].shape
    
    #imten_list = [torch.tensor(im).permute(2, 0, 1)[None, ...] for im in im_list]
    
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

def multibias(im_list, nscales, stopping_mse, config, pretrained_models=None, temporal_weightshare=True,test_data=None):
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
    
    def backward_hook(module, grad_input, grad_output):
        global gradients # refers to the variable in the global scope
        print('Backward hook running...')
        gradients = grad_output
        # In this case, we expect it to be torch.Size([batch size, 1024, 8, 8])
        print(f'Gradients size: {gradients[0].size()}') 
        # We need the 0 index because the tensor containing the gradients comes
        # inside a one element tuple.


    if sys.platform == 'win32':
        visualize = True
    else:
        visualize = False

    #nimg = len(im_list)
    #nimg_test = len(test_data)
    
    

    #imten_list = [torch.tensor(im).permute(2, 0, 1)[None, ...] for im in im_list]
   

    #imten = torch.cat(imten_list, 0).cuda()


    # resize
    #imten = torch.nn.functional.interpolate(imten,size=(480,960),mode="area")
    
    #B, C, H, W = imten.shape
    if config.resize != -1:
        H,W = config.resize
    else:
        H,W = 960,1920
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
    nparams_array = np.zeros(1)
    
    for idx in range(1):
        # Create model
        model = models.get_model(config, nchunks,rank=None)

        if False:
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

        if pretrained_models is not None:
            model.load_state_dict(pretrained_models[idx])
            
        model_list.append(model)
    
    #model_state_dict = model_list[0].state_dict()    
    #for key in model_state_dict.keys():
    #    print(key, model_state_dict[key].dtype)
    
    
    optim = torch.optim.Adam(lr=config.lr, params=params)
        
    # Criteria
    criterion = losses.L2Norm()
    train_dataset = VideoDataset(config.path,config.freq, config.n_frames,
                                 nchunks,True,config.partition_size, config.resize)
    test_dataset = VideoDataset(config.path,config.freq, config.n_frames,
                                 nchunks,False,config.partition_size, config.resize)
    train_sampler = BalancedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.bs, shuffle=False,
            num_workers=4, pin_memory=True, sampler=train_sampler, drop_last=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=2, shuffle=False,
            num_workers=4, pin_memory=True, sampler=train_sampler, drop_last=False)
    
    # Create inputs
    coords_chunked = utils.get_coords((H,W),
                                      config.ksize,
                                      config.coordstype,
                                      unfold)
    

    coords_chunked = coords_chunked.cuda()
    #t_idx_train = torch.tensor(config.train_idx,device="cuda",dtype=torch.float)
    #y_ = config.n_frames // (config.partition_size)
    #x_ = y_ // 2
    #centers = torch.tensor([x_ + y_*center_no for center_no in range(config.partition_size)],dtype=torch.float,device="cuda")
    #centers = centers.unsqueeze(-1)
    #cluster_to_idx_dist = (torch.abs(centers - t_idx_train.unsqueeze(0)) - (torch.arange(config.partition_size,device="cuda")*1e-3).unsqueeze(-1))
    #model_idx = torch.argmin(cluster_to_idx_dist,dim=0)
    #model_idx = model_idx.cuda()
    #indices_t = torch.stack([torch.randperm(y_//2)+(y_//2)*j for j in range(config.partition_size)]).transpose(1,0).flatten()




    #n_frames_ = config.train_idx[-1]
    #t_idx_train = ((t_idx_train - (n_frames_)/2)/((n_frames_)/2))
    #t_idx_train = t_idx_train[None,:,None]
    #t_idx_train = torch.repeat_interleave(t_idx_train,repeats=coords_chunked.shape[0],dim=0)
    # if config.slowmo:
    #     t_idx_test = torch.arange(0,config.n_frames,0.125,device='cuda',dtype=torch.float)
    # else:
    #     t_idx_test = torch.tensor(config.test_idx,device='cuda',dtype=torch.float)
    #     cluster_to_idx_dist = (torch.abs(centers - t_idx_test.unsqueeze(0)) - (torch.arange(config.partition_size,device="cuda")*1e-3).unsqueeze(-1))
    #     model_idx_test = torch.argmin(cluster_to_idx_dist,dim=0)
    #     model_idx_test = model_idx_test.cuda()

    #t_idx_test = ((t_idx_test - (n_frames_)/2) / ((n_frames_)/2))[None,:,None]
    #t_idx_test = torch.repeat_interleave(t_idx_test,repeats=coords_chunked.shape[0],dim=0)
    

   

    mse_array = np.zeros(config.epochs)
    best_mse = float('inf')
    best_img = None
    
    learn_indices = torch.arange(nchunks, device='cuda')

    tbar = tqdm.tqdm(range(config.epochs))
    #rec_t = torch.zeros((config.n_frames // config.freq ,3, H, W), dtype=torch.float)

    for idx in tbar:
        lr = config.lr*pow(0.1, idx/config.epochs)
        optim.param_groups[0]['lr'] = lr
        psnr_list = []
        #indices_t = torch.randperm(nimg)
        for sample in train_loader: 
            t_coords = sample["t"].cuda().permute(1,0,2)
            imten = sample["img"].cuda()
            model_idx = sample["model_idx"].cuda()
    
            #t_indices = indices_t[t_idx:min(nimg,t_idx+3)]
            #split_size = t_indices.shape[0]
            #t_coords = t_idx_train[:, t_indices, :].cuda()
            #model_coords = model_idx[t_indices]
            t_coords = (t_coords,model_idx)
            #t_indices = t_indices.cuda()
            optim.zero_grad()
            

            im_out = model(coords_chunked, learn_indices,t_coords)
            im_out = im_out.permute(0, 3, 2, 1).flatten(1,2)
                #im_estim_list.append(
                #    im_out.permute(2, 1, 0))
            
            #im_estim_chunked = torch.cat(im_estim_list, 0)
            
            im_estim = fold(im_out).reshape(config.bs, -1, H, W)
            
            if config.warm_start and idx < config.warm_epochs:
                im_estim = fold(im_out).reshape(1, -1, H, W)
                loss = criterion(im_estim, imten[0,...])
            else:
                #im_estim = fold(im_out).reshape(split_size, -1, H, W)
                loss = criterion(im_estim, imten)
            


            loss.backward()
            optim.step()
            with torch.no_grad():
                lossval = loss.item()
                psnr_list.append(-10*math.log10(lossval))

            
        # Do a nuclear norm prox gradient step here
        
        #im_diff = abs(imten - im_estim).mean(1)*10
        #im_diff_cpu = im_diff.detach().cpu().numpy()
        
        #im_estim_cpu = rec_t.detach().permute(0, 2, 3, 1).cpu().numpy()
        
        
        # with torch.no_grad():
        #     mse_list = ((rec_t - imten)**2).mean([1, 2, 3])
        #     lossval = torch.mean(mse_list)
        #     mse_array[idx] = lossval.item()
        #     mse_list = tuple(mse_list.cpu().numpy().tolist())
        
        # if lossval < best_mse:
        #     best_mse = lossval
        #     #best_img = im_estim_cpu
            
        # if lossval < stopping_mse:
        #     #best_img = im_estim_cpu
        #     break
        
        # if visualize:
        #     for idx in range(nimg):
        #         cv2.imshow('Estim %d'%idx, im_estim_cpu[idx, ...])
        #     cv2.waitKey(1)
        
        # #mse_minmax = (min(mse_list), max(mse_list))
        # max_psnr = -10*np.log10(min(mse_list))
        # min_psnr = -10*np.log10(max(mse_list))
        avg_psnr = sum(psnr_list) / len(psnr_list)
        # mse_minmax = (min_psnr, max_psnr)
        tbar.set_description(('%.3f')%(avg_psnr))
        tbar.refresh()
    

    #imtest_list = [torch.tensor(im).permute(2, 0, 1)[None, ...] for im in test_data]
    #imtest = torch.cat(imtest_list,0).cuda()
    #imtest = torch.nn.functional.interpolate(imtest,size=(480,960),mode="area")
    error_test = []
    rec_test = []
    # Test for intermediate coordinates 
    with torch.no_grad():
        model = model_list[0]
        if config.slowmo:
            t_idx_test_batch = torch.split(t_idx_test,50,dim=1)
            t_len = t_idx_test.shape[1]
            im_out_list = []
            for t_idx_inst in t_idx_test_batch:
                im_out = model(coords_chunked,learn_indices,t_idx_inst) 
                im_out_list.append(im_out)
            im_out = torch.cat(im_out_list,0)
        else:
            #indices_t = torch.arange(0,nimg_test)
            for sample in test_loader: 
                #t_indices = indices_t[t_idx:min(nimg,t_idx+2)]
                #split_size = t_indices.shape[0]
                #t_coords = t_idx_test[:, t_indices, :].cuda()
                #model_coords = model_idx[t_indices]
                #t_coords = (t_coords,model_coords)
                #t_indices = t_indices.cuda()
                t_coords = sample["t"].cuda().permute(1,0,2)
                imten = sample["img"].cuda()
                model_idx = sample["model_idx"].cuda()
                t_coords = (t_coords,model_idx)
                im_out = model(coords_chunked,learn_indices,t_coords)   
                im_out = im_out.permute(0, 3, 2, 1).flatten(1,2)
                im_estim = fold(im_out).reshape(2, -1, H, W) 
                with torch.no_grad():
                    error_test.append(((imten-im_estim)**2).detach().cpu())
                    rec_test.append(im_estim.detach().cpu())


        if config.slowmo:
            im_estim = fold(im_out).reshape(t_len, -1, H, W)
        best_img =  torch.cat(rec_test,dim=0).permute(0, 2, 3, 1).numpy()
        if not config.slowmo:
            mse_list_test = (torch.cat(error_test,dim=0)).mean([1, 2, 3])
            mse_list_test = tuple(mse_list_test.numpy().tolist())
            psnr_array_test = -10*np.log10(np.array(mse_list_test))
            print('test psnr: {:.3f}'.format(np.average(psnr_array_test)))
    psnr_array_train = avg_psnr
    
    print('train psnr: {:.3f}'.format(psnr_array_train))
    info = {'psnr_array_train': psnr_array_train,
            'psnr_array_test': 0 if config.slowmo else psnr_array_test,
            'nparams_array': nparams_array}
    

    
    return best_img, info, model

def multibias_biaspred(im_list, nscales, stopping_mse, config, pretrained_models=None, temporal_weightshare=True, alpha=0.5):
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
    print(nimg, alpha)
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
    
    pred_params = []
    nparams_array = np.zeros(nimg)
    
    hidden_omega_0 = 1.0
    #model_pred = wire.AdaptiveLinearWithChannel(input_size=1, 
    #                output_size=config.nfeat*2, 
    #                channel_size=nchunks,
    #                is_first=True, share_weights=False, const=1.0,
    #                bias=True, data_type=None, add_dim=False).cuda()
    
    '''
    model_pred = wire.AdaptiveMultiWIRE(
        in_features=1,
        out_features=config.nfeat*2, 
        n_channels=nchunks,
        hidden_features=config.nfeat*2, 
        hidden_layers=config.nlayers,
        outermost_linear=True,
        share_weights=config.share_weights,
        first_omega_0=config.omega_0,
        hidden_omega_0=hidden_omega_0,
        scale=config.scale,
        mode='1d',
        nonlin_type=config.nonlin_type,
        const=1.0
    ).cuda()
    
    '''
    rand_factor = torch.load('a.pt').cuda()
    model_pred = siren.AdaptiveMultiSiren(
                    in_features=1,
                    out_features=2*config.nfeat, 
                    n_channels=nchunks,
                    hidden_features=2*config.nfeat, 
                    hidden_layers=config.nlayers,
                    outermost_linear=True,
                    nonlin='relu',
                    share_weights=config.share_weights,
                    first_omega_0=config.omega_0,
                    hidden_omega_0=hidden_omega_0
                ).cuda()
        
    pred_params = list(model_pred.parameters())    
    
    if pretrained_models is not None:
        model_pred.load_state_dict(pretrained_models[0])
        pretrained_models = pretrained_models[1:]
              
    
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
                nonlin_type=config.nonlin_type,
                const=1.0
            ).cuda()
            
        if temporal_weightshare:
            #print('Sharing weights temporally!')
            if idx > 0:
                model.set_weights(model_list[0])
                #model.set_bias()
                params += model.bias_parameters()
                nparams_array[idx] = model.bias_nparams
            else:
                #model.set_bias()
                params+= list(model.parameters())
                nparams_array[idx] = utils.count_parameters(model)
        else:
            params+= list(model.parameters())
            nparams_array[idx] = utils.count_parameters(model)

        if pretrained_models is not None:
            model.load_state_dict(pretrained_models[idx])
            
        model_list.append(model)    
    #model_state_dict = model_list[0].state_dict()    
    #for key in model_state_dict.keys():
    #    print(key, model_state_dict[key].dtype)
    
    
    optim = torch.optim.Adam(lr=config.lr, params=params+pred_params)
        
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

    if pretrained_models is not None:
        config.epochs = 1
            
    tbar = tqdm.tqdm(range(config.epochs))
    for idx in tbar:
        tic = time.time()

        lr = config.lr*pow(0.1, idx/config.epochs)
        optim.param_groups[0]['lr'] = lr

        if pretrained_models is not None:

            im_estim_list = []

            model_pred.eval()
            print('eval')
            for m_idx in range(nimg):

                model = model_list[m_idx]
                model.eval()
            
                t_idx = (m_idx - 4) / 5.0
                t_idx = t_idx * torch.ones_like(learn_indices)
                t_idx = t_idx.reshape(nchunks,1,1) #+ rand_factor[m_idx]
                t_idx.cuda() # 576

                t_idx2 = (m_idx + 1 - 4) / 5.0
                t_idx2 = t_idx2 * torch.ones_like(learn_indices)
                t_idx2 = t_idx2.reshape(nchunks,1,1) #+ rand_factor[m_idx]
                t_idx2.cuda() # 576

                pred_biases1 = model_pred(t_idx, learn_indices) # nchunks, 1, 2*f
                pred_biases2 = model_pred(t_idx2, learn_indices) # nchunks, 1, 2*f
                #pred_biases_ = model_pred(t_idx, learn_indices) # nchunks, 1, 2*f
                pred_biases = (1-alpha)*pred_biases1 + alpha*pred_biases2
                #pred_biases = utils.sperical_interpolation(pred_biases1.reshape(nchunks, 2, config.nfeat), pred_biases2.reshape(nchunks, 2, config.nfeat), alpha=alpha)
                
                #pred_biases = [None]*2
                #pred_biases[0] = ((pred_biases1[1] + pred_biases2[1]).real/2).reshape(nchunks, 2, config.nfeat)
                #pred_biases[1] = (pred_biases1[2] + pred_biases2[2]).reshape(nchunks, 2, config.nfeat)/2
                
                #pred_biases[0] = pred_biases2[1].reshape(nchunks, 2, config.nfeat)
                #pred_biases[1] = pred_biases2[2].reshape(nchunks, 2, config.nfeat)
                
                pred_biases = pred_biases.reshape(nchunks, 2, config.nfeat)
                im_reconst = model(coords_chunked, learn_indices, ex_biases=pred_biases)
                im_estim_list.append(
                    im_reconst.permute(2, 1, 0))
    
            im_estim_chunked = torch.cat(im_estim_list, 0)
            im_estim = fold(im_estim_chunked).reshape(nimg, -1, H, W)
            loss = criterion(im_estim, imten)
            lossval = loss.item()

        else:
            optim.zero_grad()            
            im_estim_list = []
            
            for m_idx in range(nimg):
                model = model_list[m_idx]
                            
                t_idx = (m_idx - 4) / 5.0
                t_idx = t_idx * torch.ones_like(learn_indices)
                t_idx = t_idx.reshape(nchunks,1,1) #+ rand_factor[m_idx]
                t_idx.cuda() # 576
                
                pred_biases = model_pred(t_idx, learn_indices) # nchunks, 1, 2*f
                #print(len(pred_biases))
                
                #pred_biases[1] = pred_biases[1].real.reshape(nchunks, 2, config.nfeat)
                #pred_biases[2] = pred_biases[2].reshape(nchunks, 2, config.nfeat)
                
                pred_biases = pred_biases.reshape(nchunks, 2, config.nfeat)
                im_reconst =  model(coords_chunked, learn_indices, ex_biases=pred_biases)
                im_estim_list.append( im_reconst.permute(2, 1, 0) )
        
            im_estim_chunked = torch.cat(im_estim_list, 0)
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
        
    psnr_array = -10*np.log10(np.array(mse_list))
    info = {'psnr_array': psnr_array,
            'nparams_array': nparams_array}
    
    if nimg == 2:
        info['imdiff'] = im_diff_cpu
    
    return best_img, info, model_list, model_pred



def multibias_2d(im_list, nscales, stopping_mse, config, pretrained_models=None):
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
    model_2_list = []
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
                nonlin_type=config.nonlin_type,
                const=1.0
            ).cuda()
            
            model_2 = wire.AdaptiveMultiWIRE(
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
                nonlin_type=config.nonlin_type,
                const=1.0
            ).cuda()     
        
    
        if idx > 0:
            model.set_weights(model_list[0])
            model_2.set_weights(model_2_list[0])
            params += model.bias_parameters()
            params += model_2.bias_parameters()
            nparams_array[idx] = model.bias_nparams + model_2.bias_nparams
        else:
            params+= list(model.parameters())
            params+= list(model_2.parameters())
            nparams_array[idx] = utils.count_parameters(model) + utils.count_parameters(model_2)

        if pretrained_models is not None:
            model.load_state_dict(pretrained_models[idx])
            
        model_list.append(model)
        model_2_list.append(model_2)
    
    #model_state_dict = model_list[0].state_dict()    
    #for key in model_state_dict.keys():
    #    print(key, model_state_dict[key].dtype)
    
    
    optim = torch.optim.Adam(lr=config.lr, params=params)
        
    # Criteria
    criterion = losses.L2Norm()
    criterion_cossim = torch.nn.CosineSimilarity(dim=2, eps=1e-08)
    
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
        
        im_estim_list1 = []
        im_estim_list2 = []
        
        reg_loss = 0
        for m_idx in range(nimg):
            model = model_list[m_idx]
            model_2 = model_2_list[m_idx]
            
            m1bias = model.bias_parameters() # all layers biases
            m2bias = model_2.bias_parameters() # all layers biases
            for each_l in range(len(m1bias)):
                tmp=10
                m1l = m1bias[each_l]
                m2l = m2bias[each_l]
                
                if m1l.dtype==torch.complex64:
                    # enforce spatial similarity with in model
                    reg_loss += criterion(m1l[1:].real, m1l[0:-1].real) +  criterion(m1l[1:].imag, m1l[0:-1].imag)
                    reg_loss += criterion(m2l[1:].real, m2l[0:-1].real) +  criterion(m2l[1:].imag, m2l[0:-1].imag)
                else:
                    reg_loss += criterion(m1l[1:], m1l[0:-1]) 
                    reg_loss += criterion(m2l[1:], m2l[0:-1]) 

                # enforce disimilarity across model
                
                #num = torch.inner(m1bias[each_l], m2bias[each_l])
                #den1 = torch.inner(m1bias[each_l], m1bias[each_l])
                #den2 = torch.inner(m2bias[each_l], m2bias[each_l])
                
            b = int(torch.rand(1)[0]>0.5)
            im_estim_list.append(
                b*model(coords_chunked, learn_indices).permute(2, 1, 0) + (1-b)*model_2(coords_chunked, learn_indices).permute(2, 1, 0))
            
            im_estim_list1.append(model(coords_chunked, learn_indices).permute(2, 1, 0))
            im_estim_list2.append(model_2(coords_chunked, learn_indices).permute(2, 1, 0))
            
            
        
        im_estim_chunked = torch.cat(im_estim_list, 0)
        im_estim_chunked1 = torch.cat(im_estim_list1, 0)
        im_estim_chunked2 = torch.cat(im_estim_list2, 0)
        
        im_estim = fold(im_estim_chunked).reshape(nimg, -1, H, W)
        im_estim1 = fold(im_estim_chunked1).reshape(nimg, -1, H, W)
        im_estim2 = fold(im_estim_chunked2).reshape(nimg, -1, H, W)
        
        if config.warm_start and idx < config.warm_epochs:
            loss_ = criterion(im_estim[0], imten[0])
        else:
            loss_ = criterion(im_estim, imten)
        
        loss = loss_ + 0.08*reg_loss
        loss.backward()
        optim.step()
        lossval = loss_.item()
        
        im_diff = abs(imten - im_estim).mean(1)*10
        im_diff_cpu = im_diff.detach().cpu().numpy()
        
        im_estim_cpu = im_estim.detach().permute(0, 2, 3, 1).cpu().numpy()
        
        im_estim_cpu1 = im_estim1.detach().permute(0, 2, 3, 1).cpu().numpy()
        im_estim_cpu2 = im_estim2.detach().permute(0, 2, 3, 1).cpu().numpy()
        
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
    
    return best_img, info, model_list, im_estim_cpu1, im_estim_cpu2


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

def adjust_lr(optimizer, cur_epoch, args):
    
    up_ratio, up_pow, min_lr = [float(x) for x in args.lr_type.split('_')[1:]]
    if cur_epoch < up_ratio:
        lr_mult = min_lr + (1. - min_lr) * (cur_epoch / up_ratio)** up_pow
    else:
        lr_mult = 0.5 * (math.cos(math.pi * (cur_epoch - up_ratio)/ (1 - up_ratio)) + 1.0)

    for i, param_group in enumerate(optimizer.param_groups):
        param_group['lr'] = args.lr * lr_mult

    return args.lr * lr_mult