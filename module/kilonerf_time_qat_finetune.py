#!/usr/bin/env python

import os
import sys
from numpy.lib.twodim_base import mask_indices
from modules.quantize import quant_model_
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
from pytorch_msssim import ms_ssim
import cv2
import torch
import torch.nn.utils.prune as prune
from torch.optim.lr_scheduler import ExponentialLR
from torch.nn import Parameter
from dataset_class import VideoDataset, BalancedSampler, DistributedSamplerWrapper
import folding_utils as unfoldNd
import torch.nn.functional as F
from dahuffman import HuffmanCodec
from wire_time_qat_finetune import QuantNoise
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
#from distr_sampler import MyDistributedSampler
from torch.utils.data.distributed import DistributedSampler





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
    # H=1
    # test_dataset.h_max =1 
    # test_dataset.w_max = 300
    # W=300
    # nchunks=1
    nchunks= test_dataset.nchunks

    # if config.resize != -1:
    #     H,W = config.resize
    # else:
        
    #     H,W = 1080,1920
    #     #H,W = 480,480
    # Create folders and unfolders



    
    fold = torch.nn.Fold(output_size=(H, W),
                         kernel_size=config.ksize,
                         stride=config.stride)
    
    #window_weights = get_bilinear(H,W,config.ksize,config.stride)
    #window_weights = window_weights.cuda(rank)
    




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
    
    #hyper_parameters = [param for name, param in model.named_parameters() if 'hyper' in name]
    #other_parameters = [param for name, param in model.named_parameters() if 'hyper' not in name]
    #train_dataset.edge_init(model)
    #from optimizer import Adan
    #optim = Adan(params=params,lr=config.lr)
    # optim = torch.optim.Adam([
    # {'params': hyper_parameters, 'lr': 1e-3},  # Learning rate for 'hyper' parameters
    # {'params': other_parameters, 'lr': 5e-3}])   # Learning rate for other parameters

    optim = torch.optim.Adam(lr=config.lr, params=params)
    #optim_prune = torch.optim.Adam(lr=config.lr, params=params)
    #optim_qat = torch.optim.Adam(lr=config.lr, params=params)
    # Criteria
    criterion = losses.L2Norm()

    #prune parameters
    if config.prune:
        param_list = []
        for k,v in model.named_modules():
            if hasattr(v,"weight"):
                param_list.append(v)
        param_to_prune = [(ele,"weight") for ele in param_list]
        prune_base_ratio = config.prune_ratio
        prune_num = 0
    

    
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
        # train without pruning
        for idx in tbar:

            lr = config.lr*pow(0.1, idx/config.epochs)
            optim.param_groups[0]['lr'] = lr 
            train_sampler.set_epoch(idx)
            psnr_list = []
            idx_list = []
            idx_list1= []
            for sample in train_loader: 
                t_coords = sample["t"].cuda(rank).permute(1,0,2)
                imten = sample["img"].cuda(rank)
                model_idx = sample["model_idx"].cuda(rank)
                t_coords = (t_coords,model_idx)
                optim.zero_grad()
                im_out = model(coords_chunked, learn_indices,t_coords,epochs=idx)
                im_out = im_out.permute(0, 3, 2, 1).reshape(-1,config.out_features,config.ksize[0],config.ksize[1],nchunks)
                im_out = im_out.reshape(-1,config.out_features*config.ksize[0]*config.ksize[1],nchunks)
                im_estim = fold(im_out).reshape(-1, config.out_features, H, W)
                
                if config.warm_start and idx < config.warm_epochs:
                    im_estim = fold(im_out).reshape(1, -1, H, W)
                    loss = criterion(im_estim, imten[0,...])
                else:
                    loss=criterion(im_estim, imten)#*grad_map[:,None,...]
                loss.backward()
                optim.step()
                with torch.no_grad():
                    lossval = loss.item()
                    psnr_list.append(-10*math.log10(lossval))

                
            avg_psnr = sum(psnr_list) / len(psnr_list)
            if rank == 0:
                tbar.set_description(('%.3f')%(avg_psnr))
                tbar.refresh()

        # Prune weights whose absolute value is near to zero
        if config.prune:
            prune_num += 1 
            prune.global_unstructured(
                param_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=1 - prune_base_ratio ** prune_num,
            )
            sparisity_num = 0.
            for param in param_list:
                sparisity_num += (param.weight == 0).sum()
            print(f'Model sparsity at Epoch{idx}: {sparisity_num / 1e6 / total_params}')
        tbar = tqdm.tqdm(range(config.prune_epoch),disable = not master)
        # Fine-tune with pruned weights
        config.lr = config.lr*0.2
        for idx in tbar:

            lr = config.lr*pow(0.1, idx/config.epochs)
            optim.param_groups[0]['lr'] = lr 
            train_sampler.set_epoch(idx)
            psnr_list = []
            idx_list = []
            idx_list1= []
            for sample in train_loader: 
                t_coords = sample["t"].cuda(rank).permute(1,0,2)
                imten = sample["img"].cuda(rank)
                model_idx = sample["model_idx"].cuda(rank)
                t_coords = (t_coords,model_idx)
                optim.zero_grad()
                im_out = model(coords_chunked, learn_indices,t_coords,epochs=idx)
                im_out = im_out.permute(0, 3, 2, 1).reshape(-1,config.out_features,config.ksize[0],config.ksize[1],nchunks)
                im_out = im_out.reshape(-1,config.out_features*config.ksize[0]*config.ksize[1],nchunks)
                im_estim = fold(im_out).reshape(-1, config.out_features, H, W)
                
                if config.warm_start and idx < config.warm_epochs:
                    im_estim = fold(im_out).reshape(1, -1, H, W)
                    loss = criterion(im_estim, imten[0,...])
                else:
                    loss=criterion(im_estim, imten)#*grad_map[:,None,...]
                loss.backward()
                optim.step()
                with torch.no_grad():
                    lossval = loss.item()
                    psnr_list.append(-10*math.log10(lossval))

                
            avg_psnr = sum(psnr_list) / len(psnr_list)
            if rank == 0:
                tbar.set_description(('%.3f')%(avg_psnr))
                tbar.refresh()
        # Activate quantization
        with torch.no_grad():
            for k, v in model.named_modules():
                if "quantize" in k:
                    v.bitwidth.copy_(config.quant_model_bit)
                    v.noise_ratio.copy_(0.9)
                    v.ste = True
        tbar = tqdm.tqdm(range(config.qat_epoch),disable = not master)
        # Fine-tune with QAT
        for idx in tbar:
            if master:
                lr = config.lr*pow(0.1, idx/config.epochs)
                optim.param_groups[0]['lr'] = lr 
            train_sampler.set_epoch(idx)
            psnr_list = []
            idx_list = []
            idx_list1= []
            for sample in train_loader: 
                t_coords = sample["t"].cuda(rank).permute(1,0,2)
                imten = sample["img"].cuda(rank)
                model_idx = sample["model_idx"].cuda(rank)
                t_coords = (t_coords,model_idx)
                optim.zero_grad()
                im_out = model(coords_chunked, learn_indices,t_coords,epochs=idx)
                im_out = im_out.permute(0, 3, 2, 1).reshape(-1,config.out_features,config.ksize[0],config.ksize[1],nchunks)
                im_out = im_out.reshape(-1,config.out_features*config.ksize[0]*config.ksize[1],nchunks)
                im_estim = fold(im_out).reshape(-1, config.out_features, H, W)
                
                if config.warm_start and idx < config.warm_epochs:
                    im_estim = fold(im_out).reshape(1, -1, H, W)
                    loss = criterion(im_estim, imten[0,...])
                else:
                    loss=criterion(im_estim, imten)#*grad_map[:,None,...]
                loss.backward()
                optim.step()
                with torch.no_grad():
                    lossval = loss.item()
                    psnr_list.append(-10*math.log10(lossval))

                
            avg_psnr = sum(psnr_list) / len(psnr_list)
            if rank == 0:
                tbar.set_description(('%.3f')%(avg_psnr))
                tbar.refresh()        

    # Inference
    if rank==0:

        # Test for intermediate coordinates 
        pred_videos = []
        if config.prune:
            if config.inference:
                prune_num=0
                prune.global_unstructured(
                param_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=1 - prune_base_ratio ** prune_num)
            
            for param in param_to_prune:
                prune.remove(param[0], param[1])
            sparisity_num = 0.
            for param in param_list:
                sparisity_num += (param.weight == 0).sum()
            print(f'Model sparsity at Epoch{0}: {sparisity_num / 1e6 / total_params}\n')
        with torch.no_grad():
            model_list, quant_ckt = quant_model_(model,config)
        with torch.no_grad():

            for model_ind, cur_model in enumerate(model_list):
                cur_model.eval()
                error_test = []
                rec_test = []
                mssim_list = []
                for sample in test_loader: 
                    t_coords = sample["t"].cuda(rank).permute(1,0,2)
                    imten = sample["img"].cuda(rank)
                    model_idx = sample["model_idx"].cuda(rank)
                    t_coords = (t_coords,model_idx)
                    im_out = cur_model(coords_chunked,learn_indices,t_coords,epochs=10000,gt_data=imten)#, toy visualization 
                    im_out = im_out.permute(0, 3, 2, 1).reshape(1,config.out_features,config.ksize[0],config.ksize[1],-1)
                    im_out = im_out#*window_weights
                    im_out = im_out.reshape(1,config.out_features*config.ksize[0]*config.ksize[1],-1)
                    im_estim = fold(im_out).reshape(-1, config.out_features, H, W) 
                    with torch.no_grad():
                        error_test.append(((imten-im_estim)**2).detach().cpu())
                        rec_test.append(im_estim.detach().cpu())
                        mssim_list.append(msssim_fn_batch([im_estim], imten))


                best_img =  torch.cat(rec_test,dim=0).permute(0, 2, 3, 1).numpy()
                pred_videos.append(best_img)
                if not config.slowmo:
                    mse_list_test = (torch.cat(error_test,dim=0)).mean([1, 2, 3])
                    mse_list_test = tuple(mse_list_test.numpy().tolist())
                    mssim_mean = torch.cat(mssim_list,dim=0).mean()
                    psnr_array_test = -10*np.log10(np.array(mse_list_test))
                    avg_test_psnr = np.average(psnr_array_test)
                    quant_str = "quantized" if model_ind else "non-quantized"
                    print('{} test psnr: {:.3f}'.format(quant_str,avg_test_psnr))
                    with open("{}/rank0.txt".format(save_path),"a") as f:
                        f.write("{} Average Test PSNR : {:.4f} \n".format(quant_str,avg_test_psnr))
                        f.write("{} Average Test SSIM : {:.4f} \n".format(quant_str,mssim_mean))
                        if not config.inference:
                            f.write("Average Train PNSR: {:.4f} \n".format(avg_psnr))
            quant_v_list = []
            tmin_scale_len = 0
            for k, layer_wt in quant_ckt.items():
                cand= layer_wt
                #cand = cand[cand!=0]
                quant_v_list.extend(cand.flatten().tolist())
                #tmin_scale_len += layer_wt['min'].nelement() + layer_wt['scale'].nelement()
                
            unique, counts = np.unique(quant_v_list, return_counts=True)
            num_freq = dict(zip(unique, counts))
            codec = HuffmanCodec.from_data(quant_v_list)
            sym_bit_dict = {}
            for k, v in codec.get_code_table().items():
                sym_bit_dict[k] = v[0]
            total_bits = 0
            for num, freq in num_freq.items():
                total_bits += freq * sym_bit_dict[num]
            bits_per_param = total_bits / len(quant_v_list)
            #total_bits += tmin_scale_len * 16 
            full_bits_per_param = total_bits / len(quant_v_list)
            total_bpp = total_bits / (H*W) / config.n_frames
            with open("{}/rank0.txt".format(save_path),"a") as f:
                 f.write(f'After quantization and encoding: \n bits per parameter: {round(full_bits_per_param, 2)}, bits per pixel: {round(total_bpp, 4)}')
            print(f'After quantization and encoding: \n bits per parameter: {round(full_bits_per_param, 2)}, bits per pixel: {round(total_bpp, 4)}')
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


