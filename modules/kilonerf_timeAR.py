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
import itertools
import ast
import math
import numpy as np
from scipy import io
from skimage.metrics import structural_similarity as ssim_func
# from distr_sampler import MyDistributedSampler
from pytorch_msssim import ms_ssim
import cv2
import torchac
import torch
import torch.nn.utils.prune as prune
from torch.optim.lr_scheduler import ExponentialLR
from torch.nn import Parameter
from modules.dataset_class import VideoDataset, BalancedSampler, DistributedSamplerWrapper
import folding_utils as unfoldNd
import torch.nn.functional as F
from dahuffman import HuffmanCodec

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
plt.gray()
from losses import entropy_reg
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
## from distr_sampler import MyDistributedSampler
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
        H,W = 960,1920

    unfold = torch.nn.Unfold(kernel_size=config.ksize, stride=config.stride)
    if not config.inference:
        train_dataset = VideoDataset(config.path,config.freq, config.n_frames,
                                 True,config.partition_size, config.resize, unfold=unfold,start=config.start,end=config.end,config=config)
        
        train_sampler = DistributedSampler(train_dataset,num_replicas=world_size,rank=rank,shuffle=True)
    
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.bs, shuffle=False,
                    num_workers=0, pin_memory=True, sampler=train_sampler, drop_last=False)
    
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

    
        
    # Criteria
    criterion = losses.L2Norm()
    entropy_loss = entropy_reg(config)
    # define optimizer
    uncompressed_paramaters= 0
    param_groups = []
    for n,p in model.named_parameters():
        if (n.endswith("linear.weight_real") or n.endswith("orth_scale.weight_real") or n.endswith("orth_scale.weight_imag")
            or n.endswith("scale") or n.endswith("linear.weight") or n.endswith("orth_scale.weight")
            or n.endswith("linear.weight_imag")) and (not "hyper" in n):
            param_groups+=[{"params":[p],"lr":config.lr}]
        else:
            uncompressed_paramaters+=p.numel()
    print(f'Parameters which are not compressed:{uncompressed_paramaters/10**3}K')
    # Maybe we can downgrade hyper training to catch up with weight configuration
    param_groups += [{
        'params': [p for n,p in model.named_parameters() if (("bias" in n) or ("hyper"in n))]
        ,"lr": config.lr,"name": "params"
    }]
    optim = torch.optim.Adam(param_groups)
    prob_model_parameters = []
    for prob_model in model.module.prob_models.values():
        if isinstance(prob_model,dict):
            lin= prob_model["linear"]
            orth = prob_model["orthogonal"]
            if isinstance(lin,list):
                for cur_lin,cur_ortho in zip(lin,orth):
                    prob_model_parameters = itertools.chain(prob_model_parameters,cur_lin.parameters())
                    prob_model_parameters = itertools.chain(prob_model_parameters,cur_ortho.parameters())
            else:
                prob_model_parameters = itertools.chain(prob_model_parameters,lin.parameters())
                prob_model_parameters = itertools.chain(prob_model_parameters,orth.parameters())
        else:
            prob_model_parameters = itertools.chain(prob_model_parameters,prob_model[0].parameters())
            prob_model_parameters = itertools.chain(prob_model_parameters,prob_model[1].parameters())
    prob_optimizer = torch.optim.Adam(prob_model_parameters, lr = config.prob_lr)
    
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
    total_params = sum([p.data.nelement()*32 if p.dtype==torch.float else p.data.nelement()*32 for p in model.parameters()]) 
    tbar = tqdm.tqdm(range(config.epochs),disable = not master)

    if not config.inference:

        for idx in tbar:
            lr = config.lr*pow(0.1, idx/config.epochs)
            prob_lr = config.prob_lr*pow(0.1, idx/config.epochs)
            for k in range(len(param_groups)):     
                optim.param_groups[k]['lr'] = lr
            for k in range(len(list(prob_model_parameters))):
                prob_optimizer.param_groups[k]['lr'] = prob_lr

            train_sampler.set_epoch(idx)
            psnr_list = []
            for sample in train_loader: 
                t_coords = sample["t"].cuda(rank).permute(1,0,2)
                imten = sample["img"].cuda(rank)
                model_idx = sample["model_idx"].cuda(rank)
                t_coords = (t_coords,model_idx)
                optim.zero_grad()
                
                prob_optimizer.zero_grad()
                latents = model.module.get_latents()
                # Find INFINITY NORM of continious weight surrogate for its standardazition with specific interval
                with torch.no_grad():
                    if (idx % 2 == 0):
                        for k,v in latents.items():

                            if "_" in k and "layer1" in k:
                                #divider = torch.max(torch.abs(v.min(dim=0,keepdim=True)[0]),\
                                #torch.abs(v.max(dim=0,keepdim=True)[0]))
                                divider= v.std(dim=0,keepdim=False)

                                layer_no, layer_type = k.split("_")
                                decoder= model.module.weight_decoders[layer_no][layer_type]
                            elif "_" in k and (not "layer1" in k):
                                try:
                                    layer_no, layer_type, layer_isreal = k.split("_")
                                    decoder_idx = 0 if "real" in layer_isreal else 1
                                    decoder= model.module.weight_decoders[layer_no][layer_type][decoder_idx]
                                except:
                                    layer_no, layer_isreal = k.split("_")
                                    decoder_idx = 0 if "real" in layer_isreal else 1
                                    decoder= model.module.weight_decoders[layer_no][decoder_idx]
           
                                #divider = torch.amax((torch.cat([torch.abs(v.min(dim=0,keepdim=True)[0]),
                                #                                 torch.abs(v.max(dim=0,keepdim=True)[0])],dim=0)),
                                #                                 dim=0,keepdim=True)
                                divider= v.std(dim=0,keepdim=False)
                                
                            decoder.div = divider
                            decoder.div[decoder.div==0] += 1 # prevent 0 divisions 


                im_out = model(coords_chunked, learn_indices,t_coords,epochs=idx)
                im_out = im_out.permute(0, 3, 2, 1).reshape(-1,config.out_features,config.ksize[0],config.ksize[1],nchunks)
                im_out = im_out.reshape(-1,config.out_features*config.ksize[0]*config.ksize[1],nchunks)
                im_estim = fold(im_out).reshape(-1, config.out_features, H, W)
                
                if config.warm_start and idx < config.warm_epochs:
                    im_estim = fold(im_out).reshape(1, -1, H, W)
                    loss = criterion(im_estim, imten[0,...])
                else:
                    loss=criterion(im_estim, imten)#*grad_map[:,None,...]
                
                # compute entropy loss
                if idx > config.entropy_epoch:
                    entr_loss, num_bits = entropy_loss(latents,model.module.prob_models,False,config.entropy_loss_weight)
                    net_loss = entr_loss + loss
                    bpp= num_bits/(1920*960*21)
                else:
                    bpp = total_params/(1920*960*21)
                    net_loss = loss
                
                #net_loss=loss

                net_loss.backward()
                optim.step()
            
                prob_optimizer.step()
                
                with torch.no_grad():
                    lossval = loss.item()
                    psnr_list.append(-10*math.log10(lossval))

               
            
            avg_psnr = sum(psnr_list) / len(psnr_list)
            if rank == 0:
                tbar.set_description(('Train PSNR: %.3f, BPP: %.4f')%(avg_psnr,bpp))
                tbar.refresh()
    

    if rank==0:
        # Test for either intermediate coordinates or whole data
        pred_videos = []
        with torch.no_grad():
            error_test = []
            rec_test = []
            mssim_list = []
            for sample in test_loader: 
                t_coords = sample["t"].cuda(rank).permute(1,0,2)
                imten = sample["img"].cuda(rank)
                model_idx = sample["model_idx"].cuda(rank)
                t_coords = (t_coords,model_idx)
                im_out = model(coords_chunked,learn_indices,t_coords)#, toy visualization 
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
                with open("{}/rank0.txt".format(save_path),"a") as f:
                    f.write("Average Test PSNR : {:.4f} \n".format(avg_test_psnr))
                    f.write("Average Test SSIM : {:.4f} \n".format(mssim_mean))
                    if not config.inference:
                        f.write("Average Train PNSR: {:.4f} \n".format(avg_psnr))

                
        # COMPRESSION WITH ARITHMETIC ENCODER
            latents= model.module.get_latents()
            ac_bytes_diff_emp,byte_stream_diff_emp= compute_ac_bytes(latents,False,False,False)
            bpp=ac_bytes_diff_emp/(960*1920*21)
            with open("{}/rank0.txt".format(save_path),"a") as f:
                f.write("BPP : {:.4f} \n".format(bpp))
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
    



def compute_ac_bytes(weights, use_diff_ac, use_prob_model, fit_linear):
    ac_bytes = 0
    overhead = []
    with torch.no_grad():
        for weight_tag,cur_weight in weights.items():
            weight = torch.round(cur_weight)
            if use_diff_ac:
                assert not use_prob_model, "Not implemented prob model for diff"
                if fit_linear:
                    X = torch.round(self.previous_latents[group_name])
                    block_size = self.model.weight_decoders[group_name].block_size
                    for m in self.model.modules():
                        if isinstance(m, layers.Linear):
                            if self.model.groups[m.name] == group_name:
                                out_features = m.out_features
                                break
                    X = rearrange(X, '(b c) (b1 c1) -> (b b1) (c c1)', b1=block_size[0], 
                                        c1=block_size[1], b=out_features//block_size[0])
                    X = torch.cat((X.unsqueeze(-1),torch.ones_like(X).unsqueeze(-1)),dim=-1)
                    Y = weights[group_name]
                    Y = rearrange(Y, '(b c) (b1 c1) -> (b b1) (c c1)', b1=block_size[0], 
                                        c1=block_size[1], b=out_features//block_size[0]).unsqueeze(-1)
                    try:
                        out = torch.linalg.inv(torch.matmul(X.permute(0,2,1),X))
                        out = torch.matmul(torch.matmul(out,X.permute(0,2,1)),Y)
                        overhead += [out.detach().cpu()]
                        pred_Y = torch.matmul(X,out)
                        weight = torch.round(Y-pred_Y).reshape(weights[group_name].size())
                    except Exception as e:
                        weight = torch.round(weights[group_name]) - torch.round(self.previous_latents[group_name])
                else:
                    weight = weight - torch.round(self.previous_latents[group_name])
            # Loop over each channel
            for dim in range(weight.size(1)):
                weight_pos = weight[:,dim] - torch.min(weight[:,dim]) # bring minimum to 0
                unique_vals, counts = torch.unique(weight[:,dim], return_counts = True)
                if use_prob_model:
                    unique_vals = torch.cat((torch.Tensor([unique_vals.min()-0.5]).to(unique_vals),\
                                            (unique_vals[:-1]+unique_vals[1:])/2,
                                            torch.Tensor([unique_vals.max()+0.5]).to(unique_vals)))
                    if "_" in weight_tag:
                        layer_no,layer_type = weight_tag.split("_")
                        cur_prob_model= self.model.prob_models[layer_no][layer_type]
                    else:
                        cur_prob_model = self.model.prob_models[layer_no]

                    cdf = cur_prob_model(unique_vals,single_channel=dim)
                    cdf = cdf.detach().cpu().unsqueeze(0).repeat(weight.size(0),1)
                else:
                    cdf = torch.cumsum(counts/counts.sum(),dim=0).detach().cpu()
                    cdf = torch.cat((torch.Tensor([0.0]),cdf))
                    cdf = cdf/cdf[-1]
                    cdf = cdf.unsqueeze(0).repeat(weight.size(0),1)
                weight_pos = weight_pos.long()
                unique_vals = torch.unique(weight_pos)
                mapping = torch.zeros((weight_pos.max().item()+1))
                mapping[unique_vals] = torch.arange(unique_vals.size(0)).to(mapping)
                weight_pos = mapping[weight_pos.cpu()]
                byte_stream = torchac.encode_float_cdf(cdf.clamp(min=0.0,max=1.0).detach().cpu(), weight_pos.detach().cpu().to(torch.int16), \
                                                check_input_bounds=True)
                ac_bytes += len(byte_stream)
    return ac_bytes+sum([torch.finfo(t.dtype).bits/8*t.numel() for t in overhead]),byte_stream


def msssim_fn_single(output, gt):
    msssim = ms_ssim(output.float().detach(), gt.detach(), data_range=1, size_average=False)
    return msssim.cpu()

def msssim_fn_batch(output_list, gt):
    msssim_list = [msssim_fn_single(output.detach(), gt.detach()) for output in output_list]
    # for output in output_list:
    #     msssim = ms_ssim(output.float().detach(), gt.detach(), data_range=1, size_average=False)
    #     msssim_list.append(msssim)
    return torch.stack(msssim_list, 0).cpu()