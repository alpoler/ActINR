#!/usr/bin/env python

import os
import sys
import tqdm
import pdb
import math
import numpy as np
import torch
from torch import nn
from modules.siren_hyperinr import AdaptiveMultiSiren
import torch.nn.functional as F
import rff
import matplotlib.pyplot as plt
from customlinear import custom_linear
#from siren import PosEncoding


class AdaptiveLinearWithChannel(nn.Module):
    '''
        Implementation from https://github.com/pytorch/pytorch/issues/36591
        
        Evaluate only selective channels
    '''
    def __init__(self, input_size, output_size, channel_size,
                 is_first=False, share_weights=False, const=1.0,
                 bias=True, data_type=None, add_dim=False,n_img=None,rank=None,config=None):
        super(AdaptiveLinearWithChannel, self).__init__()
        
        if is_first or data_type is torch.float:
            dtype = torch.float
        else:
            if data_type is None:
                dtype = torch.cfloat
            else:
                dtype = data_type #torch.cfloat
                
        if share_weights:
            w_nchan = 1
        else:
            w_nchan = channel_size
        self.complex_layer = (dtype == torch.cfloat)
        self.share_weights = share_weights
        self.add_bias = bias
        self.input_size = input_size
        self.output_size = output_size
        self.rank = rank
        self.w_nchan = w_nchan
        self.customlinear = custom_linear(config.group_size,w_nchan,input_size,output_size,self.complex_layer,is_first,rank,config)
        self.tiles_no = config.no_tiles
        self.total_epochs=0.75*config.epochs
        self.final = False
        self.no_frames = config.n_frames - 1
        self.const = const

    def forward(self, x, indices,t=None, ex_biases=None,coords=None,epochs=None):        
        t, model_idx = t
        bias_idx = (t[0,:,0]*self.no_frames).long()
        output = self.customlinear(x,model_idx,bias_idx,t)
        return  output 
        
class AdaptiveMultiGaborLayer(nn.Module):
    '''
        Implements WIRE activations with multiple channel input.

    '''
    def __init__(self, in_features, out_features, n_channels,
                 is_first=False, share_weights=False,
                 omega_0=30, scale_0=5.0,n_img=None,rank=None,config=None):
        
        super().__init__()        
        self.in_features = in_features
        self.omega_0 = omega_0
        self.scale_0 = scale_0
        self.linear = AdaptiveLinearWithChannel(in_features,
                                                out_features,
                                                n_channels,
                                                is_first,
                                                share_weights,n_img=n_img,rank=rank,config=config)
        
    def forward(self, input, indices,t=None,coords=None,epochs=None):
        lin = self.linear(input, indices,t=t,coords=coords,epochs=epochs)
        omega = self.omega_0 * lin
        scale = self.scale_0 * lin
        
        return torch.exp(1j*omega - scale.abs().square())
        #return torch.sin(omega)*torch.exp(-scale.abs().square())


    
class AdaptiveMultiGabor2DLayer(nn.Module):
    '''
        Implements WIRE2D activations with multiple channel input.

    '''
    
    def __init__(self, in_features, out_features, n_channels,
                 is_first=False, share_weights=False,
                 omega_0=30, scale_0=5.0,n_img=None,rank=None,config=None):
        super().__init__()        
        # Divide features by 2 since we have two linear layers
        self.in_features = in_features
        self.out_features = out_features
        self.omega_0 = omega_0#nn.Parameter(omega_0*torch.ones(1).to(rank), True)
        self.scale_0 = scale_0#nn.Parameter(scale_0*torch.ones(1).to(rank), True)

        
        linlayer = AdaptiveLinearWithChannel


        self.linear = linlayer(in_features,
                                out_features,
                                n_channels,
                                is_first,
                                share_weights,n_img=n_img,rank=rank,config=config)
        
        self.orth_scale = linlayer(in_features,
                                    out_features,
                                    n_channels,
                                    is_first,
                                    share_weights,n_img=n_img,rank=rank,config=config)

        
    def forward(self, input, indices,t=None,coords=None,epochs=None):
        lin = self.linear(input, indices,t,coords=coords,epochs=epochs)        
        lin2 = self.orth_scale(input, indices,t,coords=coords,epochs=epochs)

        omega = self.omega_0 * lin
        scale = self.scale_0 * lin
        orth_scale = self.scale_0 * lin2
        
        arg = 1j*omega-scale.abs().square()-orth_scale.abs().square()
        
        return torch.exp(arg)

class AdaptiveMultiWIRE(nn.Module):
    def __init__(self, in_features, hidden_features, 
                 hidden_layers, out_features, n_channels, outermost_linear=False, share_weights=False,
                 first_omega_0=30, hidden_omega_0=30.,
                 scale=10.0, const=1.0, mode='2d',n_img=None,pos_encode=False,rank=None,config=None):
        super().__init__()
        
        self.mode = mode
        self.rank = rank
        self.tv_loss_bias = config.tv_loss_bias
        self.abs_log = config.abs_log
        self.diff_encoding = config.diff_encoding
        if self.mode == '1d':
            self.nonlin = AdaptiveMultiGaborLayer
        elif self.mode == '2d':
            self.nonlin = AdaptiveMultiGabor2DLayer
            

        if type(hidden_features) is int:
            hidden_features = [hidden_features]*hidden_layers
        
        '''
        self.pos_encode = False
        if in_features==1:
            print('POS encoding')
            self.positional_encoding = PosEncoding(in_features=in_features,
                                                   sidelength=480,
                                                   fn_samples=None,
                                                   use_nyquist=True)
            in_features = self.positional_encoding.out_dim
            self.pos_encode = Trued
        '''
        self.pos_encode = pos_encode
        if self.pos_encode:
            print('POS encoding')
            self.positional_encoding = PositionalEncoding('1.25_80',n_chunks=n_channels)
            #in_features = self.positional_encoding.embed_length
        
        hidden_features = [in_features] + hidden_features
        self.w_nchan = n_channels\
        
        self.net = []
        for i in range(len(hidden_features)-1):
            feat1 = hidden_features[i]
            feat2 = hidden_features[i+1]
            is_first = (i == 0)
            sw = (i > 0) and share_weights
            
            if is_first:
                omega_0 = first_omega_0
            else:
                omega_0 = hidden_omega_0
                
            self.net.append(self.nonlin(
                                feat1,
                                feat2, 
                                n_channels, 
                                is_first=is_first,
                                share_weights=sw, 
                                omega_0=omega_0,
                                scale_0=scale,n_img=n_img,rank=rank,config=config))

        if outermost_linear:
            feat = hidden_features[-1]
            final_linear = AdaptiveLinearWithChannel(
                                    feat, 
                                    out_features,
                                    n_channels,
                                    is_first=False,
                                    share_weights=share_weights,data_type=torch.float if self.mode == "complete_real" else None,
                                    const=const,
                                    bias=True,n_img=n_img,rank=rank,config=config)
                    
            self.net.append(final_linear)
        else:
            feat = hidden_features[-1]
            self.net.append(self.nonlin(
                                feat,
                                out_features, 
                                n_channels,
                                is_first=False, 
                                share_weights=share_weights,
                                omega_0=hidden_omega_0,
                                const=const))
        
        self.net = nn.ModuleList(self.net)

        
    def reset_bias(self):
        for module in self.net:
            try:
                module.linear.reset_bias()
                
                if self.mode == '2d':
                    module.orth_scale.reset_bias()
            except AttributeError:
                module.reset_bias()
        
    def bias_parameters(self):
        '''
            Wrapper to get only bias parameters
        '''
        params = []
        nparams = 0
        for module in self.net:
            try:
                params.append(module.linear.bias)
                nparams += params[-1].numel()
                if self.mode == '2d':
                    params.append(module.orth_scale.bias)
                    nparams += params[-1].numel()
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
                    self.net[idx].orth_scale.weight = \
                        ref_model.net[idx].orth_scale.weight
            except AttributeError:
                self.net[idx].weight = \
                    ref_model.net[idx].weight

    def set_bias(self):
        '''
            Wrapper to set the weights of one model to another
        '''
        for idx in range(len(self.net)):
            try:
                self.net[idx].linear.bias = \
                    self.net[idx].orth_scale.bias
            except AttributeError:
                tmp=10
    
    def forward(self, inp, indices,t=None, ex_biases=None,epochs=None,gt_data=None):            
        output = inp[None,indices, ...]
        coords = output.clone()
        for i_mod, mod in enumerate(self.net):
            if ex_biases is not None and i_mod==0:
                output = mod(output, indices, ex_biases)
            else:
                output = mod(output, indices, t,coords=coords,epochs=epochs)

        if output.dtype==torch.complex64:
            return output.real
        else:   
            return output
    
    def init_data(self):
        for m in self.modules():
            if type(m) in [custom_linear]:
                if m.complex:
                    weight,bias = torch.view_as_complex(m.weight), torch.view_as_complex(m.bias)
                    w_mag, w_angle = torch.abs(weight), torch.angle(weight)
                    b_mag, b_angle = torch.abs(bias), torch.angle(bias)
                    m.weight_angle_quantizer.init_data(w_angle)
                    m.weight_mag_quantizer.init_data(w_mag)
                    m.bias_angle_quantizer.init_data(b_angle)
                    m.bias_mag_quantizer.init_data(b_mag)
                else:
                    m.weight_quantizer.init_data(m.weight)
                    m.bias_quantizer.init_data(m.bias)
    def cal_params(self, entropy_model=None,old_codes=None):

        for name,m in self.named_modules():
            if type(m) in [custom_linear]:
                if m.complex:
                    weight,bias = torch.view_as_complex(m.weight), torch.view_as_complex(m.bias)
                    w_mag, w_angle = torch.abs(weight), torch.angle(weight)
                    b_mag, b_angle = torch.abs(bias), torch.angle(bias)
                    code_w_angle, quant_w_angle, dequant_w_angle = m.weight_angle_quantizer(w_angle)
                    code_b_angle, quant_b_angle, dequant_b_angle = m.bias_angle_quantizer(b_angle)
                    code_w_mag, quant_w_mag, dequant_w_mag = m.weight_mag_quantizer(w_mag)
                    code_b_mag, quant_b_mag, dequant_b_mag = m.bias_mag_quantizer(b_mag)
                    if self.tv_loss_bias:
                        code_b_list = [code_b_mag[:,0,...],code_b_mag[:,1:,...] - code_b_mag[:,:-1,...]]
                        quant_b_list = [quant_b_mag[:,0,...],quant_b_mag[:,1:,...] - quant_b_mag[:,:-1,...]]
                        code_diff_ang = code_b_angle[:,1:,...] - code_b_angle[:,:-1,...]
                        code_diff_ang = torch.atan2(torch.sin(code_diff_ang), torch.cos(code_diff_ang))
                        quant_diff_ang = torch.round(code_diff_ang)
                        #quant_diff_ang = torch.atan2(torch.sin(quant_diff_ang), torch.cos(quant_diff_ang))
                        code_b_list_ang = [code_b_angle[:,0,...],code_diff_ang]
                        quant_b_list_ang = [quant_b_angle[:,0,...],quant_diff_ang]                        

                    m.dequant_w_mag , m.dequant_w_angle = dequant_w_mag , dequant_w_angle
                    m.dequant_b_mag , m.dequant_b_angle = dequant_b_mag, dequant_b_angle

                    

                else:   
                    weight, bias = m.weight, m.bias
                    code_w, quant_w, dequant_w = m.weight_quantizer(weight)
                    code_b, quant_b, dequant_b = m.bias_quantizer(bias)
                    if self.tv_loss_bias:
                        code_b_list = [code_b[:,0,...],code_b[:,1:,...] - code_b[:,:-1,...]]
                        quant_b_list = [quant_b[:,0,...],quant_b[:,1:,...] - quant_b[:,:-1,...]]
                    m.dequant_w = dequant_w
                    m.dequant_b = dequant_b



                if entropy_model is not None:
                    if m.complex:
                        m.bitrate_w_mag_dict.update(entropy_model.cal_bitrate(code_w_mag, quant_w_mag, self.training))
                        if not self.tv_loss_bias:
                            m.bitrate_b_mag_dict.update(entropy_model.cal_bitrate(code_b_mag, quant_b_mag, self.training))
                        else:
                            a= entropy_model.cal_bitrate(code_b_list[0], quant_b_list[0], self.training)
                            b = entropy_model.cal_bitrate(code_b_list[1].flatten().unsqueeze(0),quant_b_list[1].flatten().unsqueeze(0),self.training)
                            merged_dict={}
                            for k in set(a.keys()).union(b.keys()):
                                if k in a and k in b:
                                    if k == "bitrate" or k == "real_bitrate":
                                        merged_dict[k] = a[k] + b[k]
                                    else:
                                        merged_dict[k] = torch.cat([a[k], b[k]], dim=0)
                            m.bitrate_b_mag_dict.update(merged_dict)
                        
                        m.bitrate_w_angle_dict.update(entropy_model.cal_bitrate(code_w_angle, quant_w_angle, self.training))
                        if not self.tv_loss_bias:
                            m.bitrate_b_angle_dict.update(entropy_model.cal_bitrate(code_b_angle, quant_b_angle, self.training))
                        else:
                            a = entropy_model.cal_bitrate(code_b_list_ang[0],quant_b_list_ang[0],self.training)
                            b = entropy_model.cal_bitrate(code_b_list_ang[1].flatten().unsqueeze(0),quant_b_list_ang[1].flatten().unsqueeze(0),self.training)
                            merged_dict={}
                            for k in set(a.keys()).union(b.keys()):
                                if k in a and k in b:
                                    if k == "bitrate" or k == "real_bitrate":
                                        merged_dict[k] = a[k] + b[k]
                                    else:
                                        merged_dict[k] = torch.cat([a[k], b[k]], dim=0)
                            m.bitrate_b_angle_dict.update(merged_dict)
                    else:
                        
                        m.bitrate_w_dict.update(entropy_model.cal_bitrate(code_w, quant_w, self.training))
                        if not self.tv_loss_bias:
                            m.bitrate_b_dict.update(entropy_model.cal_bitrate(code_b, quant_b, self.training))
                        else:
                            a= entropy_model.cal_bitrate(code_b_list[0], quant_b_list[0], self.training)
                            
                            b = entropy_model.cal_bitrate(code_b_list[1].flatten().unsqueeze(0),quant_b_list[1].flatten().unsqueeze(0),self.training)
                            merged_dict={}
                            for k in set(a.keys()).union(b.keys()):
                                if k in a and k in b:
                                    if k == "bitrate" or k == "real_bitrate":
                                        merged_dict[k] = a[k] + b[k]
                                    else:
                                        merged_dict[k] = torch.cat([a[k], b[k]], dim=0)
                            m.bitrate_b_dict.update(merged_dict)

    def get_bitrate_sum(self, name="bitrate"):
        sum = 0
        for m in self.modules():
            if type(m) in [custom_linear]:
                if m.complex:
                    sum += m.bitrate_w_mag_dict[name]
                    sum += m.bitrate_w_angle_dict[name]    
                    sum += m.bitrate_b_mag_dict[name]
                    sum += m.bitrate_b_angle_dict[name]
                else:
                    sum += m.bitrate_w_dict[name]
                    sum += m.bitrate_b_dict[name]
        return sum
    
    
        
    def residual_sparsity(self, old_weights,old_bias,precision_matrix):
        # Unstructured sparsity encouragement
        l2_norm = 0
        for name,m in self.named_modules():
            if type(m) in [custom_linear]:
                if m.complex:
                    cur_weight_mag = torch.view_as_complex(m.weight).abs()
                    #cur_bias_mag = torch.view_as_complex(m.bias).abs()
                else:
                    cur_weight_mag = m.weight
                    #cur_bias_mag = m.bias
                
                delta_weight = (cur_weight_mag-old_weights[name])
                #delta_bias = (cur_bias_mag - old_bias[name])
                delta_weight = torch.square(delta_weight)
                #delta_bias = torch.square(delta_bias)
                if precision_matrix is not None:
                    cur_precision = precision_matrix["module."+ name+ ".weight"].view(1,-1,1,1)
                    delta_weight = (cur_precision**2)*delta_weight
                l2_norm+=torch.sqrt(torch.sum(delta_weight)+1e-20)#+torch.sum(delta_bias)
        return l2_norm        
    
    def set_temperature(self,epoch):
        for name,cur_module in self.named_modules():
            if type(cur_module) in [custom_linear]:
                if cur_module.complex:
                    cur_module.weight_angle_quantizer.set_temperature(epoch)
                    cur_module.weight_mag_quantizer.set_temperature(epoch)
                    cur_module.bias_angle_quantizer.set_temperature(epoch)
                    cur_module.bias_mag_quantizer.set_temperature(epoch)
                else:
                    cur_module.weight_quantizer.set_temperature(epoch)
                    cur_module.bias_quantizer.set_temperature(epoch)

    @torch.no_grad()
    def update_old_parameters(self,old_parameters):
        old_codes={}
        for name,m in self.named_modules():
            if type(m) in [custom_linear]:
                if m.complex:
                    w_mag,w_angle=old_parameters[name][0],old_parameters[name][1]
                    code_w_mag,quant_w_mag,_ = m.weight_mag_quantizer(w_mag)
                    code_w_ang,quant_w_ang,_ = m.weight_angle_quantizer(w_angle)
                    old_codes[name]=[code_w_mag,quant_w_mag,code_w_ang,quant_w_ang]
                else:
                    weight = old_parameters[name][0]
                    code_w,quant_w,_ = m.weight_quantizer(weight)
                    old_codes[name]= [code_w,quant_w]
        return old_codes
    
    def smoothness_loss(self,p):
        tv_loss=0
        #phase_loss=0
        for name,m in self.named_modules():
            if type(m) in [custom_linear]:
                if m.complex:
                    bias = torch.view_as_complex(m.bias)
                    bias_mag = bias.abs()
                    bias_angle = bias.angle()
                    n_fr,w_nchan,in_size,out_size = bias.shape
                    bias_mag = bias_mag.view(n_fr,w_nchan,out_size).permute(1,0,2)
                    diff_mag = (bias_mag[:,1:,:] - bias_mag[:,:-1,:]) 
                    diff_mag_abs = torch.abs(diff_mag)
                    bias_angle = bias_angle.view(n_fr,w_nchan,out_size).permute(1,0,2)
                    diff_ang = (bias_angle[:,1:,:] - bias_angle[:,:-1,:])
                    diff_ang = torch.atan2(torch.sin(diff_ang), torch.cos(diff_ang))
                    diff_angle = torch.abs(diff_ang)
                    mask = torch.zeros_like(diff_mag)
                    mask.bernoulli_(p)
                    cur_tv_loss_mag = torch.sum(mask*diff_mag_abs)
                    cur_tv_loss_ang = 0.01*torch.sum(mask*diff_angle)
                    cur_tv_loss = cur_tv_loss_mag + cur_tv_loss_ang
                else:
                    bias = m.bias
                    n_fr,w_nchan,in_size,out_size = bias.shape
                    bias = bias.view(n_fr,w_nchan,out_size).permute(1,0,2)
                    diff = (bias[:,1:,:] - bias[:,:-1,:]) 
                    diff_abs = torch.abs(diff)
                    mask = torch.zeros_like(diff_abs)
                    mask.bernoulli_(p)
                    cur_tv_loss = torch.sum(mask*diff_abs)
                tv_loss = tv_loss + cur_tv_loss
        return tv_loss #,phase_loss