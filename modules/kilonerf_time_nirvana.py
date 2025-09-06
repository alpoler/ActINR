#!/usr/bin/env python

import os
import sys
from numpy.lib.twodim_base import mask_indices
from modules.quantize import quant_model
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
# from distr_sampler import MyDistributedSampler
from pytorch_msssim import ms_ssim
import cv2
import torch
import torch.nn.utils.prune as prune
from torch.optim.lr_scheduler import ExponentialLR
from torch.nn import Parameter
from modules.dataset_class import VideoDataset, BalancedSampler, DistributedSamplerWrapper
import folding_utils as unfoldNd
import torch.nn.functional as F
# from dahuffman import HuffmanCodec

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
plt.gray()

import utils
import siren_time
import losses
import volutils
import wire
import models

utils = importlib.reload(utils)
siren = importlib.reload(siren_time)
losses = importlib.reload(losses)
volutils = importlib.reload(volutils)
wire = importlib.reload(wire)
models = importlib.reload(models)
# from distr_sampler import MyDistributedSampler
from torch.utils.data.distributed import DistributedSampler


class qfn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, k):
        
        if input.dtype==torch.cfloat:
            n = float(2**(k-1) - 1)
            input = torch.view_as_real(input)
            max_w = torch.max(torch.abs(input)).detach()
            input = input / 2 / max_w + 0.5
            input = torch.round(input * n) / n
            out = max_w * (2 * input - 1)
            out = torch.view_as_complex(out)
        else:
            n = float(2**(k-1) - 1)
            max_w = torch.max(torch.abs(input)).detach()
            input = input / 2 / max_w + 0.5
            input = torch.round(input * n) / n
            out = max_w * (2 * input - 1)

        return out

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None



def multibias(rank, stopping_mse, config, pretrained_models=None,world_size=None):
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
    save_path = os.path.join(config.save_path, config.dataset_name, config.model_type+'_'+str(config.nfeat).zfill(2))
    #device = torch.device("cuda:{.d}".format(rank))
    if sys.platform == 'win32':
        visualize = True
    else:
        visualize = False
    # Dataset Preparation
    if config.resize != -1:
        H,W = config.resize
    else:
        H,W = 1080,1920
        #H,W = 960,1920
        #H,W = 480,480

    unfold = torch.nn.Unfold(kernel_size=config.ksize, stride=config.stride)
    if not config.inference:
        train_dataset = VideoDataset(config.path,config.freq, config.n_frames,
                                 True,config.partition_size, config.resize, unfold=unfold,start=config.start,end=config.end,config=config)
        
        train_sampler = DistributedSampler(train_dataset,num_replicas=world_size,rank=rank,shuffle=True)
    
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.bs, shuffle=False,
                    num_workers=0, pin_memory=True, sampler=train_sampler, drop_last=False)

        # Toy 1d example    
        # train_dataset.h_max =1
        # train_dataset.w_max = 300
    
    if config.freq == 1:
        test_dataset = VideoDataset(config.path,config.freq, config.n_frames,
                            True,config.partition_size, config.resize,unfold=unfold,start=config.start,end=config.end,config=config)
        
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False,
                num_workers=0, pin_memory=True, sampler=None, drop_last=False)

    else:
        test_dataset = VideoDataset(config.path,config.freq, config.n_frames,
                                    False,config.partition_size, config.resize,unfold=unfold,start=config.start,end=config.end,config=config)
        
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False,
                num_workers=0, pin_memory=False, sampler=None, drop_last=False)



    H = test_dataset.h_max+1
    W = test_dataset.w_max+1

    nchunks= test_dataset.nchunks

    
    fold = torch.nn.Fold(output_size=(H, W),
                         kernel_size=config.ksize,
                         stride=config.stride)

    config.no_tiles = test_dataset.no_tiles
    print(config.no_tiles)

    
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
            print(nparams_array[idx]/1e6)

        if pretrained_models is not None:
            if config.inference:
                new_state_dict = {key.replace("module.",""): value for key, value in pretrained_models[idx].items()}
                pretrained_models[idx].clear()
                pretrained_models[idx].update(new_state_dict)
            model.load_state_dict(pretrained_models[idx])
            
        model_list.append(model)


    param_groups=[]
    for name,params in model.named_parameters():
        if name.endswith("weight"):
            param_groups+=[{"params": [params],"lr": config.lr,"name":name}]
        elif name.endswith("scale"):
            param_groups+=[{"params": [params],"lr": config.lr,"name":name}]
        elif name.endswith("shift"):
            param_groups+=[{"params": [params],"lr": config.lr,"name":name}]
        
    param_groups += [{"params": [p for n,p in model.named_parameters() if n.endswith("bias")],
                     "lr": config.lr,"name": "biases"}]
    optim = torch.optim.Adam(param_groups)
    prob_models=[{"params": [p for n,p in model.named_parameters() if "cdf" in n],"lr": config.prob_lr,"name": "prob_params"}]
    prob_optim = torch.optim.Adam(prob_models)
    
        
    # Criteria
    criterion = losses.L2Norm()


    # Create inputs
    coords_chunked = utils.get_coords((H,W),
                                     config.ksize,
                                     config.coordstype,
                                     unfold)
    

    coords_chunked = coords_chunked.cuda(rank)
    # coords_chunked = (torch.linspace(-1,1,config.ksize[1])[None,:,None]).cuda(rank) # toy 1 d example
    mse_array = np.zeros(config.epochs)
    best_mse = float('inf')
    best_img = None
    master = (rank == 0)
    learn_indices = torch.arange(nchunks).cuda(rank)
    total_params = sum([p.data.nelement() for p in model.parameters()]) / 1e6
    tbar = tqdm.tqdm(range(config.epochs),disable = not master)

    if not config.inference:

        for idx in tbar:
            lr = config.lr*pow(0.1, idx/config.epochs)
            optim.param_groups[0]['lr'] = lr
            train_sampler.set_epoch(idx)
            psnr_list = []


                        
            if idx % 10 == 0:
                model.module.set_divider()
            
            for sample in train_loader: 
                t_coords = sample["t"].cuda(rank).permute(1,0,2)
                imten = sample["img"].cuda(rank)
                model_idx = sample["model_idx"].cuda(rank)
                t_coords = (t_coords,model_idx)
                
                optim.zero_grad()
                prob_optim.zero_grad()

                entr_loss, bits = model.module.calculate_bit_per_parameter()
                im_out = model(coords_chunked, learn_indices,t_coords,epochs=idx)
                im_out = im_out.permute(0, 3, 2, 1).reshape(-1,config.out_features,config.ksize[0],config.ksize[1],nchunks)
                #im_out = im_out#*window_weights
                im_out = im_out.reshape(-1,config.out_features*config.ksize[0]*config.ksize[1],nchunks)
                im_estim = fold(im_out).reshape(-1, config.out_features, H, W)
                
                if config.warm_start and idx < config.warm_epochs:
                    im_estim = fold(im_out).reshape(1, -1, H, W)
                    loss = criterion(im_estim, imten[0,...])
                else:
                    loss=criterion(im_estim, imten)
                    total_loss = loss + config.lambda_rate*entr_loss
                


                total_loss.backward()
                optim.step()
                prob_optim.step()
                with torch.no_grad():
                    bpp = bits / (H*W*config.n_frames)
                    lossval = loss.item()
                    psnr_list.append(-10*math.log10(lossval))

                
            avg_psnr = sum(psnr_list) / len(psnr_list)
            if rank == 0:
                tbar.set_description(('%.3f %.3f')%(avg_psnr,bpp))
                tbar.refresh()
    

    if rank==0:

        # Test for intermediate coordinates 
        pred_videos = []

        with torch.no_grad():

            error_test = []
            rec_test = []
            mssim_list = []
            time_list= []
            for sample in test_loader: 
                t_coords = sample["t"].cuda(rank).permute(1,0,2)
                imten = sample["img"].cuda(rank)
                model_idx = sample["model_idx"].cuda(rank)
                t_coords = (t_coords,model_idx)
                im_out = model(coords_chunked,learn_indices,t_coords)
                im_out = im_out.permute(0, 3, 2, 1).reshape(1,config.out_features,config.ksize[0],config.ksize[1],-1)
                im_out = im_out.reshape(1,config.out_features*config.ksize[0]*config.ksize[1],-1)
                im_estim = fold(im_out).reshape(-1, config.out_features, H, W) 
                with torch.no_grad():
                    error_test.append(((imten-im_estim)**2).detach().cpu())
                    rec_test.append(im_estim.detach().cpu())
                    mssim_list.append(msssim_fn_batch([im_estim], imten))
                

            #print(time_list)
            best_img =  torch.cat(rec_test,dim=0).permute(0, 2, 3, 1).numpy()
            pred_videos.append(best_img)
            if not config.slowmo:
                mse_list_test = (torch.cat(error_test,dim=0)).mean([1, 2, 3])
                mse_list_test = tuple(mse_list_test.numpy().tolist())
                mssim_mean = torch.cat(mssim_list,dim=0).mean()
                psnr_array_test = -10*np.log10(np.array(mse_list_test))
                avg_test_psnr = np.average(psnr_array_test)
                print('Test psnr: {:.3f}'.format(avg_test_psnr))
                with open("{}/rank0.txt".format(save_path),"a") as f:
                    f.write("Average Test PSNR : {:.4f} \n".format(avg_test_psnr))
                    f.write("Average Test SSIM : {:.4f} \n".format(mssim_mean))
                    if not config.inference:
                        f.write("Average Train PNSR: {:.4f} \n".format(avg_psnr))

        if not config.inference:       
            psnr_array_train = avg_psnr
            print('train psnr: {:.3f}'.format(psnr_array_train))
        else:
            psnr_array_train= 0

        info = {'psnr_array_train': psnr_array_train,
                'psnr_array_test': 0 if config.slowmo else psnr_array_test,
                'nparams_array': nparams_array}
    

        return pred_videos[0], info, model
    else:
        return None, None, None
    

def msssim_fn_single(output, gt):
    msssim = ms_ssim(output.float().detach(), gt.detach(), data_range=1, size_average=False)
    return msssim.cpu()

def msssim_fn_batch(output_list, gt):
    msssim_list = [msssim_fn_single(output.detach(), gt.detach()) for output in output_list]
    # for output in output_list:
    #     msssim = ms_ssim(output.float().detach(), gt.detach(), data_range=1, size_average=False)
    #     msssim_list.append(msssim)
    return torch.stack(msssim_list, 0).cpu()




def get_bilinear(H,W,ksize,stride):
    last_row = (H - ksize[0])/stride[0]
    last_column = (W-ksize[1])/stride[1]

    unfold = torch.nn.Unfold(kernel_size=ksize, stride=stride)
    fold = torch.nn.Fold(output_size=(H, W), kernel_size=ksize,
                         stride=stride)
    template = unfold(fold(unfold(torch.ones(1, 1, H, W))))
    x_grid, y_grid = torch.meshgrid(torch.arange(H),torch.arange(W),indexing="ij")
    x_grid_unfold = unfold(x_grid.float().unsqueeze(0).unsqueeze(0))
    y_grid_unfold = unfold(y_grid.float().unsqueeze(0).unsqueeze(0))
    x_grid_patch = x_grid_unfold.view(1,ksize[0],ksize[1],-1)
    y_grid_patch = y_grid_unfold.view(1,ksize[0],ksize[1],-1)
    grid_patches = torch.cat([x_grid_patch,y_grid_patch],dim=0)
    template_patch = template.view(1,ksize[0],ksize[1],-1)
    centers = grid_patches[:,ksize[0]//2,ksize[1]//2,:]
    windows=[]
    for i in range(grid_patches.shape[-1]):
        

        window = torch.ones(ksize[0],ksize[1])
        coord_in_patch = grid_patches[:,:,:,i]
        coord_x = coord_in_patch[0,...] 
        coord_y = coord_in_patch[1,...]
        cur_cen = centers[:,i]
        # Bilinear weights
        a = torch.where(template_patch[0,:,:,i]==4)
        four_overlaps = coord_in_patch[:,a[0],a[1]].permute(1,0)
        area=(stride[0]-abs(four_overlaps[:,0]-cur_cen[0]))*(stride[1]-abs(four_overlaps[:,1]-cur_cen[1])) 
        area = area/(stride[0]*stride[1])
        window[a[0],a[1]]= area

        # Linear weights
        condition = template_patch[0,:,:,i]==2
        (x,y) = torch.where(template_patch[0,:,:,i]==2)
        
        ref_x_d = cur_cen[0] - ksize[0]//2 + stride[0]
        ref_x_u = cur_cen[0] - stride[0] + ksize[0]//2
        ref_y_r = cur_cen[1] - ksize[1]//2 + stride[1]
        ref_y_l = cur_cen[1] - stride[1] + ksize[1]//2

        #Exceptions first column, first row, last column, last row
        row_no = (cur_cen[0] - ksize[0]//2)/stride[0]
        col_no = (cur_cen[1] - ksize[1]//2)/stride[1]
        if (row_no == 0) and (col_no==0):

            hor_con = (coord_y>=ref_y_r)*condition
            ver_con = (coord_x>=ref_x_d)*condition
            
        elif (row_no==0) and (col_no<last_column):
            hor_con = torch.logical_or(coord_y<ref_y_l,coord_y>=ref_y_r)*condition
            ver_con = (coord_x>=ref_x_d)*condition
        elif (row_no==0) and (col_no==last_column):
            hor_con = (coord_y<ref_y_l)*condition
            ver_con = (coord_x>=ref_x_d)*condition

        elif (row_no < last_row) and (col_no==0):
            hor_con = (coord_y>=ref_y_r)*condition
            ver_con = (torch.logical_or(coord_x>=ref_x_d,coord_x<ref_x_u))*condition
        elif (row_no < last_row) and (col_no == last_column):
            hor_con = (coord_y<=ref_y_l)*condition
            ver_con = torch.logical_or(coord_x<ref_x_u,coord_x>=ref_x_d)*condition
        elif (row_no==last_row) and (col_no == 0):
            hor_con = (coord_y>=ref_y_r)*condition
            ver_con = (coord_x<ref_x_u)*condition
        elif (row_no==last_row) and (col_no < last_column):
            hor_con = torch.logical_or(coord_y<ref_y_l,coord_y>=ref_y_r)*condition
            ver_con = (coord_x<ref_x_u)*condition
        elif (row_no==last_row) and (col_no==last_column):
            hor_con = (coord_y<ref_y_l)*condition
            ver_con = (coord_x<ref_x_u)*condition
        else:
            hor_con = torch.logical_or(coord_y<ref_y_l,coord_y>=ref_y_r)*condition
            ver_con = torch.logical_or(coord_x<ref_x_u,coord_x>=ref_x_d)*condition
    
        
        (x_hor,y_hor) = torch.where(hor_con)        
        (x_ver,y_ver) = torch.where(ver_con)
        hor_coords = coord_y[x_hor,y_hor] 
        ver_coords = coord_x[x_ver,y_ver]
        window[x_hor,y_hor] = ((stride[1]- torch.abs(hor_coords-cur_cen[1]))/(stride[1]))
        window[x_ver,y_ver] = ((stride[0]- torch.abs(ver_coords-cur_cen[0]))/(stride[0]))

        windows.append(window)

    windows = torch.stack(windows,dim=-1)
    windows = windows.reshape(1,1,ksize[0],ksize[1],-1)
    #windows = torch.repeat_interleave(windows,repeats=1,dim=1)
    
    return windows


