#!/usr/bin/env python

import os
import sys
from numpy.lib.twodim_base import mask_indices
from modules.quantize import quant_model
import tqdm
import importlib
import time
import re
import pdb
import copy
import configparser
import argparse
import ast
import math
import difflib
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
from dahuffman import HuffmanCodec
from entropy import DiffEntropyModel
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
from modules.ewc import ewc
from modules.ewc import hessian_trace
from customlinear import custom_linear
from customlinear import Log_T 
from customlinear import Scale_T
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
    save_path = os.path.join(config.save_path, config.dataset_name, "gop_{}_{}".format(config.start,config.end))
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
    
    # Instead of loading bias series , we propose to load last bias to minimize error in
    if config.diff_encoding:
        for param_name, param_tensor in pretrained_models[0].items():
            if param_name.endswith("bias") and param_tensor.dim() >= 1:  # Check if name includes "bias"
                last_value = param_tensor[-1]  # Extract last entry along the first dimension
                pretrained_models[0][param_name] = last_value.expand_as(param_tensor).clone()  # Expand & update

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
            else:
                model.load_state_dict(pretrained_models[idx],strict=False)
            
        model_list.append(model)
    
    #hyper_parameters = [param for name, param in model.named_parameters() if 'hyper' in name]
    #other_parameters = [param for name, param in model.named_parameters() if 'hyper' not in name]
    #train_dataset.edge_init(model)
    #from optimizer import Adan
    #optim = Adan(params=params,lr=config.lr)
    # optim = torch.optim.Adam([
    # {'params': hyper_parameters, 'lr': 1e-3},  # Learning rate for 'hyper' parameters
    # {'params': other_parameters, 'lr': 5e-3}])   # Learning rate for other parameters
    # Freeze scales so that we compute residuals in consistent manner
    # if config.diff_encoding:
    #     for name, param in model.module.named_parameters():
    #         if name.endswith("scale") and (not "bias" in name):
    #             param.requires_grad = False
    optim = torch.optim.Adam(lr=config.lr, params=filter(lambda p: p.requires_grad, model.parameters()))
        


    # We compute weight importances with hope of better regularization, INR dictionaries are generalizable ? 
    # All process compute precision it is needed for all of them
    







    # Criteria
    criterion = losses.L2Norm()

    #entropy
    entropy_model = DiffEntropyModel(distribution="gaussian")
    
    # Create inputs
    coords_chunked = utils.get_coords((H,W),
                                     config.ksize,
                                     config.coordstype,
                                     unfold)
    



    learn_indices = torch.arange(nchunks).cuda(rank)
    coords_chunked = coords_chunked.cuda(rank)
    if config.precision_matrix:
        old_dataset = VideoDataset(config.path,config.freq, config.n_frames,
                                    True,config.partition_size, config.resize, unfold=unfold,start=config.start,end=config.end,config=config)
        old_dataset_loader = torch.utils.data.DataLoader(old_dataset, batch_size=config.bs, shuffle=False,
                num_workers=0, pin_memory=True, sampler=train_sampler, drop_last=False)
        
        precision = hessian_trace(model,old_dataset_loader,coords_chunked,rank,learn_indices,config, nchunks,fold)
       
        # for name, cur_module in model.module.named_modules():
        #     if isinstance(cur_module, Scale_T):
        #         matched_key = "module."+ remove_quantizer_suffix(name)
        #         bit_values = bit_assignment[matched_key]
        #         v_blocks,h_blocks = 1080 // config.ksize[0] , 1920 // config.ksize[1]
        #         reshaped_bits = bit_values.view(1, v_blocks, h_blocks, 1, 1, 1, 1)
        #         g_v, g_h = v_blocks//config.group_size[0], h_blocks // config.group_size[1]
        #         if cur_module.signed:
        #             cur_module.qmax = ((2**(reshaped_bits-1)).view(1,config.group_size[0],g_v,config.group_size[1],g_h,1,1).float().mean(dim=(2,4), keepdim=True)).permute(0,1,3,2,4,5,6).to(rank)
        #             cur_module.qmin = ((-2**(reshaped_bits-1)).view(1,config.group_size[0],g_v,config.group_size[1],g_h,1,1).float().mean(dim=(2,4),keepdim=True)).permute(0,1,3,2,4,5,6).to(rank)
        #         else:
        #             cur_module.qmax = ((2**(reshaped_bits)-1).view(1,config.group_size[0],g_v,config.group_size[1],g_h,1,1).float().mean(dim=(2,4), keepdim=True)).permute(0,1,3,2,4,5,6).to(rank)
        #             cur_module.qmin = torch.zeros_like(cur_module.qmax)                    
    
    else:
        precision=None

    # coords_chunked = (torch.linspace(-1,1,config.ksize[1])[None,:,None]).cuda(rank) # toy 1 d example
    mse_array = np.zeros(config.epochs)
    best_mse = float('inf')
    best_img = None
    master = (rank == 0)

    old_codes = None
    old_bias= None
    # Store Codebook from previous model
    with torch.no_grad():
        if config.diff_encoding:
            model.eval()
            lower_bound={}
            upper_bound ={}
            old_codes = {}
            old_weights = {}
            old_bias = {}
            for name,m in model.module.named_modules():
                if type(m) in [custom_linear]:
                    if m.complex: 
                        weight,bias = torch.view_as_complex(m.weight), torch.view_as_complex(m.bias)
                        w_mag, w_angle = torch.abs(weight), torch.angle(weight)
                        b_mag, b_angle = torch.abs(bias), torch.angle(bias)
                        #code_w_angle, quant_w_angle, dequant_w_angle = m.weight_angle_quantizer(w_angle)
                        #code_w_mag, quant_w_mag, dequant_w_mag = m.weight_mag_quantizer(w_mag)
                        #code_b_angle, quant_b_angle, dequant_b_angle = m.bias_angle_quantizer(b_angle)
                        #code_b_mag, quant_b_mag, dequant_b_mag = m.bias_mag_quantizer(b_mag)
                        
                        
                        #code_w = torch.cat([code_w_angle,code_w_mag],dim=0)
                        #quant_w = torch.cat([quant_w_angle,quant_w_mag],dim=0)
                        #code_b = torch.cat([code_b_angle,code_b_mag],dim=0)
                        #quant_b =torch.cat([quant_b_angle,quant_b_mag])

                        old_codes[name] = [w_mag,w_angle]
                        for old_code_idx,target_old_code in enumerate(old_codes[name]):
                            old_codes[name][old_code_idx] = target_old_code.detach().clone().requires_grad_(False)

                        #Residual Loss
                        old_weights[name] = weight.abs().detach().clone().requires_grad_(False)
                        #old_bias[name] = bias.abs().detach().clone().requires_grad_(False)
                        #Weight Clipping
                        #mag_scale = m.weight_mag_quantizer.scale.flatten(1,2)[:,:,:,:,0,0]
                        #ang_scale = m.weight_angle_quantizer.scale.flatten(1,2)[:,:,:,:,0,0]
                        #mag_scale_b = m.bias_mag_quantizer.scale.flatten(1,2)[:,:,:,:,0,0]
                        #ang_scale_b = m.bias_angle_quantizer.scale.flatten(1,2)[:,:,:,:,0,0]                        
                        #quant_duz_mag = torch.round(w_mag/mag_scale)
                        #quant_duz_angle = torch.round(w_angle/ang_scale)
                        #quant_duz_mag_b = torch.round(b_mag/mag_scale_b)
                        #quant_duz_angle_b = torch.round(b_angle/ang_scale_b)
                        #dt = config.delta_bit
                        #lower_bound[name] = [(quant_duz_mag-dt)*mag_scale,(quant_duz_angle-dt)*ang_scale,
                        #                     (quant_duz_mag_b-dt)*mag_scale_b,(quant_duz_angle_b-dt)*ang_scale_b]
                        #upper_bound[name] = [(quant_duz_mag+dt)*mag_scale,(quant_duz_angle+dt)*ang_scale,
                        #                     (quant_duz_mag_b+dt)*mag_scale_b,(quant_duz_angle_b+dt)*ang_scale_b]
                    else:   
                        weight, bias = m.weight, m.bias
                        old_codes[name] = [weight]
                        for old_code_idx,target_old_code in enumerate(old_codes[name]):
                            old_codes[name][old_code_idx] = target_old_code.detach().clone().requires_grad_(False)
                        #Residual Loss
                        old_weights[name] = weight.detach().clone().requires_grad_(False)
                        #old_bias[name]= bias.detach().clone().requires_grad_(False)
                        #weight clipping
                        # mag_scale = m.weight_quantizer.scale.flatten(1,2)[:,:,:,:,0,0]
                        # mag_scale_b = m.bias_quantizer.scale.flatten(1,2)[:,:,:,:,0,0]
                        # quant_duz = torch.round(weight/mag_scale)
                        # quant_duz_b = torch.round(bias/mag_scale_b)
                        # dt = config.delta_bit
                        # lower_bound[name] = [(quant_duz-dt)*mag_scale,(quant_duz_b-dt)*mag_scale_b]
                        # upper_bound[name] = [(quant_duz+dt)*mag_scale,(quant_duz_b+dt)*mag_scale_b]


    model.train()
    total_params = sum([p.data.nelement() for p in model.parameters()]) / 1e6
    target_bpp = config.target_bit * total_params * 1e6 / (config.w*config.h) / (config.n_frames)
    tbar = tqdm.tqdm(range(config.epochs),disable = not master)

    if not config.inference:
        #if not config.diff_encoding:
        model.module.init_data()
        for idx in tbar:
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

                optim.zero_grad()



                if idx >= config.compression_start_epoch:   
                    model.module.set_temperature(idx)  
                    model.module.cal_params(entropy_model,None)
                    bpp = model.module.get_bitrate_sum(name="bitrate") / (config.h*config.w*config.n_frames)
                im_out = model(coords_chunked, learn_indices,t_coords,epochs=idx)
                im_out = im_out.permute(0, 3, 2, 1).reshape(-1,config.out_features,config.ksize[0],config.ksize[1],nchunks)
                #im_out = im_out#*window_weights
                im_out = im_out.reshape(-1,config.out_features*config.ksize[0]*config.ksize[1],nchunks)
                im_estim = fold(im_out).reshape(-1, config.out_features, H, W)

                    

                if config.warm_start and idx < config.warm_epochs:
                    im_estim = fold(im_out).reshape(1, -1, H, W)
                    loss = criterion(im_estim, imten[0,...])
                else:
                    final_loss=criterion(im_estim, imten)#*grad_map[:,None,...]
                    loss = final_loss.item()

                if config.residual_loss:
                    deviation_loss = model.module.residual_sparsity(old_weights,old_bias,precision)
                    final_loss = final_loss + config.lambda_sparse*deviation_loss

                if idx >= config.compression_start_epoch: 
                    if bpp > target_bpp:
                        final_loss = final_loss + config.lambda_rate*bpp

                


                final_loss.backward()
                optim.step()
                

                # CLIP SO THAT DEVIATIONS ARE THRESHOLDED BY N BIT
                if config.delta_bit_clipping:
                    with torch.no_grad():
                        for name,cur_module in model.module.named_modules():
                            
                            if type(cur_module) in [custom_linear]:
                                if cur_module.complex:
                                    cur_weight_ = torch.view_as_complex(cur_module.weight.data)
                                    weight_angle = torch.angle(cur_weight_)
                                    weight_angle = torch.clamp(weight_angle,min=lower_bound[name][1],max=upper_bound[name][1])
                                    weight_mag = torch.abs(cur_weight_)
                                    weight_mag = torch.clamp(weight_mag,min=lower_bound[name][0],max=upper_bound[name][0])
                                    weight_real,weight_imag = torch.cos(weight_angle)*weight_mag,torch.sin(weight_angle)*weight_mag
                                    weight = torch.stack([weight_real,weight_imag],dim=-1)
                                    cur_module.weight.data.copy_(weight)
                                    #cur_bias_ = torch.view_as_complex(cur_module.bias.data)
                                    #bias_angle = torch.angle(cur_bias_)
                                    #bias_angle = torch.clamp(bias_angle,min=lower_bound[name][3],max=upper_bound[name][3])
                                    #bias_mag = torch.abs(cur_bias_)
                                    #bias_mag =torch.clamp(bias_mag,min=lower_bound[name][2],max=upper_bound[name][2])
                                    #bias_real,bias_imag = torch.cos(bias_angle)*bias_mag,torch.sin(bias_angle)*bias_mag
                                    #bias = torch.stack([bias_real,bias_imag],dim=-1)
                                    #cur_module.bias.data.copy_(bias)

                                else:
                                    cur_weight_ = cur_module.weight.data
                                    cur_weight_ = torch.clamp(cur_weight_,min=lower_bound[name][0],max=upper_bound[name][0])
                                    #cur_bias_ = cur_module.bias.data
                                    #cur_bias_ = torch.clamp(cur_bias_,min=lower_bound[name][1],max=upper_bound[name][1])
                                    cur_module.weight.data.copy_(cur_weight_)
                                    #cur_module.bias.data.copy_(cur_bias_)



                with torch.no_grad():
                    psnr_list.append(-10*math.log10(loss))

                
            avg_psnr = sum(psnr_list) / len(psnr_list)

            if rank == 0:
                tbar.set_description(('%.3f')%(avg_psnr) if not (idx >= config.compression_start_epoch) else ('%.3f,%.3f')%(avg_psnr,bpp))
                tbar.refresh()
    

    if rank==0:

        # Test for intermediate coordinates 
        pred_videos = []
        model.eval()
        with torch.no_grad():
            error_test = []
            rec_test = []
            mssim_list = []
            time_list= []
            quant_params, trans_params = [], []
            entropy_params = []

            for name,m in model.module.named_modules():
                if type(m) in [custom_linear]:
                    if m.complex:
                        weight,bias = torch.view_as_complex(m.weight), torch.view_as_complex(m.bias)
                        w_mag, w_angle = torch.abs(weight), torch.angle(weight)
                        b_mag, b_angle = torch.abs(bias), torch.angle(bias)
                        code_w_angle, quant_w_angle, dequant_w_angle = m.weight_angle_quantizer(w_angle)
                        code_b_angle, quant_b_angle, dequant_b_angle = m.bias_angle_quantizer(b_angle)
                        # if config.abs_log:
                        #     code_w_mag, quant_w_mag, dequant_w_mag,exp_params_w = m.weight_mag_quantizer(w_mag)
                        #     code_b_mag, quant_b_mag, dequant_b_mag,exp_params_b= m.bias_mag_quantizer(b_mag)
                        # else:
                        code_w_mag, quant_w_mag, dequant_w_mag = m.weight_mag_quantizer(w_mag)                                
                        code_b_mag, quant_b_mag, dequant_b_mag = m.bias_mag_quantizer(b_mag)                
                        
                        m.dequant_w_mag , m.dequant_w_angle = dequant_w_mag , dequant_w_angle
                        m.dequant_b_mag , m.dequant_b_angle = dequant_b_mag, dequant_b_angle
                        #code_w = torch.cat([code_w_angle,code_w_mag],dim=0)
                        #quant_w = torch.cat([quant_w_angle,quant_w_mag],dim=0)
                        #code_b = torch.cat([code_b_angle,code_b_mag],dim=0)
                        #quant_b =torch.cat([quant_b_angle,quant_b_mag])
                    else:   
                        weight, bias = m.weight, m.bias
                        code_w, quant_w, dequant_w = m.weight_quantizer(weight)
                        code_b, quant_b, dequant_b = m.bias_quantizer(bias)
                        m.dequant_w = dequant_w
                        m.dequant_b = dequant_b                    

                    quant_params.extend(quant_w.int().flatten().tolist())
                    if m.complex:
                        for p in m.weight_angle_quantizer.parameters():
                            trans_params.extend(p.flatten().tolist())
                        for p in m.weight_mag_quantizer.parameters():
                            trans_params.extend(p.flatten().tolist())
                        for p in m.bias_angle_quantizer.parameters():
                            trans_params.extend(p.flatten().tolist())
                        for p in m.bias_mag_quantizer.parameters():
                            trans_params.extend(p.flatten().tolist())                                
                    else:    
                        for p in m.weight_quantizer.parameters():
                            trans_params.extend(p.flatten().tolist())
                        for p in m.bias_quantizer.parameters():
                            trans_params.extend(p.flatten().tolist())
                    if config.tv_loss_bias:
                        if m.complex:
                            code_b_list = [code_b_mag[:,0,...],code_b_mag[:,1:,...] - code_b_mag[:,:-1,...]]
                            quant_b_list = [quant_b_mag[:,0,...],quant_b_mag[:,1:,...] - quant_b_mag[:,:-1,...]]
                            code_diff_ang = code_b_angle[:,1:,...] - code_b_angle[:,:-1,...]
                            code_diff_ang = torch.atan2(torch.sin(code_diff_ang), torch.cos(code_diff_ang))
                            #quant_diff_ang = quant_b_angle[:,1:,...] - quant_b_angle[:,:-1,...]
                            quant_diff_ang = torch.round(code_diff_ang)
                            code_b_list_ang = [code_b_angle[:,0,...],code_diff_ang]
                            quant_b_list_ang = [quant_b_angle[:,0,...],quant_diff_ang]                                
                        else:
                            code_b_list = [code_b[:,0,...],code_b[:,1:,...] - code_b[:,:-1,...]]
                            quant_b_list = [quant_b[:,0,...],quant_b[:,1:,...] - quant_b[:,:-1,...]]


                    if entropy_model is not None:

                        if m.complex:
                            # if config.abs_log and (not config.diff_encoding):
                            #     entropy_model.distribution = "gaussian"
                            # else:
                            exp_params_w = None
                            exp_params_b = None
                            entropy_model.distribution = "rayleigh"
                            m.bitrate_w_mag_dict.update(entropy_model.cal_bitrate(code_w_mag, quant_w_mag, False))
                            entropy_model.distribution = "gaussian"
                            if not config.tv_loss_bias:
                                entropy_model.distribution = "rayleigh"
                                m.bitrate_b_mag_dict.update(entropy_model.cal_bitrate(code_b_mag, quant_b_mag, False))
                                entropy_model.distribution = "gaussian"
                            else:
                                entropy_model.distribution = "rayleigh"
                                a= entropy_model.cal_bitrate(code_b_list[0], quant_b_list[0], False)
                                code_b_list= code_b_list[1].flatten()
                                quant_b_list= quant_b_list[1].flatten()
                                code_b_list = code_b_list[quant_b_list!=0].unsqueeze(0)
                                quant_b_list = quant_b_list[quant_b_list!=0].unsqueeze(0)
                                entropy_model.distribution="categorical"
                                b = entropy_model.cal_bitrate(code_b_list,quant_b_list,False)
                                entropy_model.distribution="gaussian"
                                merged_dict={}
                                for k in set(a.keys()).union(b.keys()):
                                    if k in a and k in b:
                                        if k == "bitrate" or k == "real_bitrate":
                                            merged_dict[k] = a[k] + b[k]
                                        else:
                                            merged_dict[k] = torch.cat([a[k], b[k]], dim=0)
                                m.bitrate_b_mag_dict.update(merged_dict)                                
                            
                            m.bitrate_w_angle_dict.update(entropy_model.cal_bitrate(code_w_angle, quant_w_angle, False))
                            if not config.tv_loss_bias:
                                m.bitrate_b_angle_dict.update(entropy_model.cal_bitrate(code_b_angle, quant_b_angle, False))
                            else:
                                a = entropy_model.cal_bitrate(code_b_list_ang[0],quant_b_list_ang[0],False)
                                code_b_list_ang= code_b_list_ang[1].flatten()
                                quant_b_list_ang= quant_b_list_ang[1].flatten()
                                code_b_list_ang = code_b_list_ang[quant_b_list_ang!=0].unsqueeze(0)
                                quant_b_list_ang = quant_b_list_ang[quant_b_list_ang!=0].unsqueeze(0)
                                entropy_model.distribution="categorical"                                
                                b = entropy_model.cal_bitrate(quant_b_list_ang,quant_b_list_ang,False)
                                entropy_model.distribution="gaussian" 
                                merged_dict={}
                                for k in set(a.keys()).union(b.keys()):
                                    if k in a and k in b:
                                        if k == "bitrate" or k == "real_bitrate":
                                            merged_dict[k] = a[k] + b[k]
                                        else:
                                            merged_dict[k] = torch.cat([a[k], b[k]], dim=0)
                                m.bitrate_b_angle_dict.update(merged_dict)
                                
                        else:
                            
                            m.bitrate_w_dict.update(entropy_model.cal_bitrate(code_w, quant_w, False))
                            entropy_params.extend(m.bitrate_w_dict["mean"].flatten().tolist())
                            entropy_params.extend(m.bitrate_w_dict["std"].flatten().tolist())
                            if m.bias is not None:
                                
                                if not config.tv_loss_bias:
                                    m.bitrate_b_dict.update(entropy_model.cal_bitrate(code_b, quant_b, False))
                                else:
                                    a= entropy_model.cal_bitrate(code_b_list[0], quant_b_list[0], False)
                                    code_b_list= code_b_list[1].flatten()
                                    quant_b_list= quant_b_list[1].flatten()
                                    code_b_list = code_b_list[quant_b_list!=0].unsqueeze(0)
                                    quant_b_list = quant_b_list[quant_b_list!=0].unsqueeze(0)
                                    entropy_model.distribution="categorical"
                                    b = entropy_model.cal_bitrate(code_b_list,quant_b_list,False)
                                    entropy_model.distribution="gaussian"
                                    merged_dict={}
                                    for k in set(a.keys()).union(b.keys()):
                                        if k in a and k in b:
                                            if k == "bitrate" or k == "real_bitrate":
                                                merged_dict[k] = a[k] + b[k]
                                            else:
                                                merged_dict[k] = torch.cat([a[k], b[k]], dim=0)
                                    m.bitrate_b_dict.update(merged_dict)

                                entropy_params.extend(m.bitrate_b_dict["mean"].flatten().tolist())
                                entropy_params.extend(m.bitrate_b_dict["std"].flatten().tolist())


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
                

            #print(time_list)
            best_img =  torch.cat(rec_test,dim=0).permute(0, 2, 3, 1).numpy()
            pred_videos.append(best_img)
            if not config.slowmo:
                mse_list_test = (torch.cat(error_test,dim=0)).mean([1, 2, 3])
                mse_list_test = tuple(mse_list_test.numpy().tolist())
                mssim_mean = torch.cat(mssim_list,dim=0).mean()
                psnr_array_test = -10*np.log10(np.array(mse_list_test))
                avg_test_psnr = np.average(psnr_array_test)
                print('test psnr: {:.3f}'.format(avg_test_psnr))
                with open("{}/rank0.txt".format(save_path),"a") as f:
                    f.write("Average Test PSNR : {:.4f} \n".format(avg_test_psnr))
                    f.write("Average Test SSIM : {:.4f} \n".format(mssim_mean))
                    if not config.inference:
                        f.write("Average Train PNSR: {:.4f} \n".format(avg_psnr))

            total_pixels= (config.h*config.w*config.n_frames)    
            trans_params_len = len(trans_params)
            estimate_bits = model.module.get_bitrate_sum(name="bitrate")
            data_bits = model.module.get_bitrate_sum(name="real_bitrate")
            meta_bits = len(entropy_params) * 32
            meta_bits += trans_params_len * 32
            estimate_total_bits = meta_bits + estimate_bits.item()
            total_bits = data_bits + meta_bits
            total_bpp = total_bits / total_pixels
            estimate_bpp = estimate_total_bits / total_pixels
            print_str = f'Gaussian Entropy Model real bpp: {round(total_bpp, 6)}, estimated bpp:{round(estimate_bpp, 6)}, target_bpp:{round(target_bpp,6)} \n'          
            print(print_str, flush=True)  
            with open("{}/rank0.txt".format(save_path),"a") as f:
                f.write(print_str + '\n')


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

def remove_quantizer_suffix(s):
    # This pattern matches any of the suffixes at the end of the string
    return re.sub(r'(_angle_quantizer|_mag_quantizer|_quantizer)$', '', s)