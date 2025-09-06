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
from customlinear_nirvana import custom_linear
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
                
        w_nchan = channel_size
        self.complex_layer = (dtype == torch.cfloat)
        self.customlinear = custom_linear(w_nchan,input_size,output_size,self.complex_layer,rank,is_first,config)
        self.no_frames = config.n_frames - 1

        
        
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
        self.omega_0 = omega_0
        self.scale_0 = scale_0

        
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
        if self.mode == '1d':
            self.nonlin = AdaptiveMultiGaborLayer
        elif self.mode == '2d':
            self.nonlin = AdaptiveMultiGabor2DLayer
            hidden_features = int(hidden_features/np.sqrt(2))

        if type(hidden_features) is int:
            hidden_features = [hidden_features]*hidden_layers

        
        hidden_features = [in_features] + hidden_features
        self.w_nchan = n_channels
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
    
    @torch.no_grad()
    def set_divider(self):
        for m in self.modules():
            if type(m) in [custom_linear]:
                if m.complex:
                    w_min = torch.abs(torch.amin(m.weight,dim=[0,2,3],keepdim=True))
                    w_max = torch.abs(torch.amax(m.weight,dim=[0,2,3],keepdim=True))
                    m.div.copy_(torch.maximum(w_min,w_max))
                else:
                    w_min = torch.abs(torch.amin(m.weight,dim=[0,2,3],keepdim=True))
                    w_max = torch.abs(torch.amax(m.weight,dim=[0,2,3],keepdim=True))
                    m.div.copy_(torch.maximum(w_min,w_max))
                m.div[m.div==0] += 1

    def get_latents(self):
        latents = None  
        for m in self.modules():
            if type(m) in [custom_linear]:
                if latents is None:
                    latents = m.get_weight_group()
                else:
                    latents = torch.cat([latents,m.get_weight_group()])

    def calculate_bit_per_parameter(self):
        bits = num_elements = 0
        for m in self.modules():
            if type(m) in [custom_linear]:
                latents=m.get_weight_group()
                prob_models=m.cdf
                for idx,prob_model in enumerate(prob_models):
                    cur_bit, prob = self_information(latents[idx],prob_model,False,is_val=False)
                    bits+=cur_bit
                    num_elements+=prob.numel()
        loss = bits / num_elements
        total_bit= bits.float().item()
        return loss,total_bit


def self_information(weight, prob_model, is_single_model=False, is_val=False, g=None):
    weight = (weight + torch.rand(weight.shape, generator=g).to(weight)-0.5) if not is_val else torch.round(weight)
    weight_p = weight + 0.5
    weight_n = weight - 0.5
    if not is_single_model:
        prob = prob_model(weight_p) - prob_model(weight_n)
    else:
        prob = prob_model(weight_p.reshape(-1,1))-prob_model(weight_n.reshape(-1,1))
    total_bits = torch.sum(torch.clamp(-1.0 * torch.log(prob + 1e-10) / np.log(2.0), 0, 50))
    return total_bits, prob