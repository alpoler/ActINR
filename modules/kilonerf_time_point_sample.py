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
from distr_sampler import MyDistributedSampler
import cv2
import torch
from torch.optim.lr_scheduler import ExponentialLR
from torch.nn import Parameter
from modules.dataset_class import VideoDataset, BalancedSampler, DistributedSamplerWrapper
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
from torch.utils.data.distributed import DistributedSampler




def multibias(rank, stopping_mse, config, pretrained_models=None):
    '''
        Kilonerf training that runs multiple INRs but with
        shared weights across INRs.
        
        Inputs:
            im_list: List of (H, W, 3) images to fit. 
            nscales: Required for compatibility with miner
            switch_mse: Blocks are terminated when this MSE is achieved
            stopping_mse: Fitting is halted if this MSE is achieved. Set to 0 to run for all iterations
                
        Outputs:
            imfit_list: List of final fitted imageF
            info: Dictionary with debugging information
                mse_array: MSE as a function of epochs
                time_array: Time as a function of epochs
                num_units_array: Number of active units as a function of epochs
                nparams: Total number of parameters
            model: Trained model
    '''
    
    #device = torch.device("cuda:{.d}".format(rank))
    if sys.platform == 'win32':
        visualize = True
    else:
        visualize = False

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
        
        model = models.get_model(config, nchunks,rank)

        if idx > 0:
            model.set_weights(model_list[0])
            params += model.bias_parameters()
            nparams_array[idx] = model.bias_nparams
        else:
            params+= list(model.parameters())
            nparams_array[idx] = utils.count_parameters(model)

        if pretrained_models is not None:
            new_state_dict = {key.replace('module.', ''): value for key, value in pretrained_models[idx].items()}
            pretrained_models[idx].clear()
            pretrained_models[idx].update(new_state_dict)
            model.load_state_dict(pretrained_models[idx])
            
        model_list.append(model)
    

    
    optim = torch.optim.Adam(lr=config.lr, params=params)
        
    # Criteria
    criterion = losses.L2Norm()

    # Dataset Preparation
    train_dataset = VideoDataset(config.path,config.freq, config.n_frames,
                                 nchunks,True,config.partition_size, config.resize)
    test_dataset = VideoDataset(config.path,config.freq, config.n_frames,
                                 nchunks,False,config.partition_size, config.resize)
    train_sampler = MyDistributedSampler(train_dataset,num_replicas=2,rank=rank)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.bs, shuffle=False,
            num_workers=0, pin_memory=True, sampler=train_sampler, drop_last=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=2, shuffle=False,
            num_workers=0, pin_memory=True, sampler=None, drop_last=False)
    
    # Create inputs
    coords_chunked = utils.get_coords((H,W),
                                      config.ksize,
                                      config.coordstype,
                                      unfold)
    

    coords_chunked = coords_chunked.cuda(rank)
   

    mse_array = np.zeros(config.epochs)
    best_mse = float('inf')
    best_img = None
    master = (rank == 0)
    learn_indices = torch.arange(nchunks).cuda(rank)

    tbar = tqdm.tqdm(range(config.epochs),disable = not master)

    if not config.inference:
        for idx in tbar:
            if master:
                lr = config.lr*pow(0.1, idx/config.epochs)
                optim.param_groups[0]['lr'] = lr
            train_sampler.set_epoch(idx)
            psnr_list = []
            idx_list = []
            idx_list1= []
            #indices_t = torch.randperm(nimg)
            for sample in train_loader: 
                t_coords = sample["t"].cuda(rank).permute(1,0,2)
                imten = sample["img"].cuda(rank)
                model_idx = sample["model_idx"].cuda(rank)
                t_coords = (t_coords,model_idx)
                
                if rank == 0:
                    idx_list.append(sample["data_idx"])
                else:
                    idx_list1.append(sample["data_idx"])
                optim.zero_grad()
                

                im_out = model(coords_chunked, learn_indices,t_coords)
                im_out = im_out.permute(0, 3, 2, 1).flatten(1,2)

                im_estim = fold(im_out).reshape(-1, 3, H, W)
                
                if config.warm_start and idx < config.warm_epochs:
                    im_estim = fold(im_out).reshape(1, -1, H, W)
                    loss = criterion(im_estim, imten[0,...])
                else:
                    loss = criterion(im_estim, imten)
                


                loss.backward()
                optim.step()
                with torch.no_grad():
                    lossval = loss.item()
                    psnr_list.append(-10*math.log10(lossval))

                
            avg_psnr = sum(psnr_list) / len(psnr_list)
            if rank == 0:
                tbar.set_description(('%.3f')%(avg_psnr))
                tbar.refresh()
    

    if rank==0:
        error_test = []
        rec_test = []
        # Test for intermediate coordinates 
        with torch.no_grad():
            
            #indices_t = torch.arange(0,nimg_test)
            for sample in test_loader: 
                t_coords = sample["t"].cuda(rank).permute(1,0,2)
                imten = sample["img"].cuda(rank)
                model_idx = sample["model_idx"].cuda(rank)
                t_coords = (t_coords,model_idx)
                im_out = model(coords_chunked,learn_indices,t_coords)   
                im_out = im_out.permute(0, 3, 2, 1).flatten(1,2)
                im_estim = fold(im_out).reshape(2, -1, H, W) 
                with torch.no_grad():
                    error_test.append(((imten-im_estim)**2).detach().cpu())
                    rec_test.append(im_estim.detach().cpu())


            best_img =  torch.cat(rec_test,dim=0).permute(0, 2, 3, 1).numpy()
            if not config.slowmo:
                mse_list_test = (torch.cat(error_test,dim=0)).mean([1, 2, 3])
                mse_list_test = tuple(mse_list_test.numpy().tolist())
                psnr_array_test = -10*np.log10(np.array(mse_list_test))
                print('test psnr: {:.3f}'.format(np.average(psnr_array_test)))

        if not config.inference:       
            psnr_array_train = avg_psnr
            print('train psnr: {:.3f}'.format(psnr_array_train))
        else:
            psnr_array_train= 0

        info = {'psnr_array_train': psnr_array_train,
                'psnr_array_test': 0 if config.slowmo else psnr_array_test,
                'nparams_array': nparams_array}
    

        return best_img, info, model
    else:
        return None, None, None