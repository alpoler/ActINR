#!/usr/bin/env python

import os
import sys
import tqdm
import pdb

import numpy as np
import torch
from torch import nn

import torch.nn.functional as F
    
class AdaptiveLinearWithChannel(nn.Module):
    '''
        Implementation from https://github.com/pytorch/pytorch/issues/36591
        
        Evaluate only selective channels
    '''
    def __init__(self, input_size, output_size, channel_size,
                 is_first=False, share_weights=False, const=1.0,
                 bias=True, data_type=None, add_dim=False, rank=None):
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
        self.share_weights = share_weights
        self.add_bias = bias
        
        #initialize weights
        if add_dim:
            self.weight = torch.nn.Parameter(torch.zeros(
                            w_nchan,
                            1,
                            input_size,
                            output_size,
                            dtype=dtype))
        else:
            self.weight = torch.nn.Parameter(torch.zeros(
                                w_nchan,
                                input_size,
                                output_size,
                                dtype=dtype))
        if bias:
            if add_dim:
                self.bias = torch.nn.Parameter(torch.zeros(channel_size,
                                                   1,
                                                   output_size,
                                                   1,
                                                   dtype=dtype))
            else:
                self.bias = torch.nn.Parameter(torch.zeros(channel_size,
                                                   1,
                                                   output_size,
                                                   dtype=dtype))
        else:
            self.bias =  torch.nn.Parameter(torch.zeros(1, 1, 1))
        
        #change weights to kaiming
        self.reset_parameters(self.weight, self.bias, const)
        self.const = const
        
    @torch.no_grad()
    def reset_parameters(self, weights, bias, const):
        _, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(weights)
        bound = np.sqrt(const*3/fan_out)
        self.weight.uniform_(-bound, bound)
        
        bound = np.sqrt(const/fan_out)
        
        if self.add_bias:
            self.bias.uniform_(-bound, bound)
    
    @torch.no_grad()
    def reset_bias(self):
        _, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
        
        bound = np.sqrt(self.const/fan_out)
        self.bias.uniform_(-bound, bound)
        
    def forward(self, x, indices):        
        if self.share_weights:
            weight = self.weight
        else:
            weight = self.weight[indices, ...]
            
        output = torch.matmul(x, weight)
        
        if self.add_bias:
            output = output + self.bias[indices, ...]
        
        return output
    
class AdaptiveLRLinearWithChannel(nn.Module):
    '''
        Implementation from https://github.com/pytorch/pytorch/issues/36591
        
        Evaluate only selective channels
    '''
    def __init__(self, input_size, output_size, channel_size,
                 is_first=False, share_weights=False, const=1.0,
                 bias=True, data_type=None, add_dim=False, rank=2):
        super(AdaptiveLRLinearWithChannel, self).__init__()
        
        if is_first or data_type is torch.float:
            dtype = torch.float
        else:
            if data_type is None:
                dtype = torch.cfloat
            else:
                dtype = data_type #torch.cfloat
       
        w_nchan = channel_size
        self.share_weights = share_weights
        self.add_bias = bias

        self.inp_sz = input_size
        self.out_sz = output_size

        self.rank = min(rank, self.inp_sz*self.out_sz)
        
        #initialize weights
        self.weights_U = torch.nn.Parameter(torch.zeros(
            w_nchan,
            self.rank,
            dtype=dtype
        ))
        self.weights_V = torch.nn.Parameter(torch.zeros(
            self.rank,
            input_size*output_size,
            dtype=dtype
        ))
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(
                channel_size,
                1,
                output_size,
                dtype=dtype))
        else:
            self.bias =  torch.nn.Parameter(torch.zeros(1, 1, 1))
        
        #change weights to kaiming
        self.reset_parameters(const)
        self.const = const
        
    @torch.no_grad()
    def reset_parameters(self, const):
        _, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(self.weights_U)
        bound = np.sqrt(const*3/fan_out)
        self.weights_U.uniform_(-bound, bound)

        _, fan_out2 = torch.nn.init._calculate_fan_in_and_fan_out(self.weights_V)
        bound = np.sqrt(const/fan_out2)
        self.weights_V.uniform_(-bound, bound)
        
        bound = np.sqrt(const/fan_out)
        
        if self.add_bias:
            self.bias.uniform_(-bound, bound)
    
    @torch.no_grad()
    def reset_bias(self):
        _, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(self.weights_U)
        
        bound = np.sqrt(self.const/fan_out)
        self.bias.uniform_(-bound, bound)
        
    def forward(self, x, indices):  
        self.weight = (self.weights_U@self.weights_V).reshape(-1, self.inp_sz, self.out_sz)
        weight = self.weight[indices, ...]
            
        output = torch.matmul(x, weight)
        
        if self.add_bias:
            output = output + self.bias[indices, ...]
        
        return output
        
class AdaptiveMultiGaborLayer(nn.Module):
    '''
        Implements WIRE activations with multiple channel input.

    '''
    def __init__(self, in_features, out_features, n_channels,
                 is_first=False, share_weights=False,
                 omega_0=30, scale_0=5.0):
        super().__init__()        
        self.in_features = in_features
        self.omega_0 = omega_0
        self.scale_0 = scale_0
        self.linear = AdaptiveLinearWithChannel(in_features,
                                                out_features,
                                                n_channels,
                                                is_first,
                                                share_weights)
        
    def forward(self, input, indices):
        lin = self.linear(input, indices)
        omega = self.omega_0 * lin
        scale = self.scale_0 * lin
        
        return torch.exp(1j*omega - scale.abs().square())
        #return torch.sin(omega)*torch.exp(-scale.abs().square())

class AdaptiveMultiGabor2DLayer_real(nn.Module):
    '''

        Implements WIRE2D activations with multiple channel input.

    '''
    
    def __init__(self, in_features, out_features, n_channels,
                 is_first=False, share_weights=False,
                 omega_0=30, scale_0=5.0):
        super().__init__()        
        # Divide features by 2 since we have two linear layers
        self.in_features = in_features
        self.omega_0 = omega_0
        self.scale_0 = scale_0
        self.linear = AdaptiveLinearWithChannel(in_features,
                                                out_features,
                                                n_channels,
                                                is_first,
                                                share_weights)
        
        self.orth_scale = AdaptiveLinearWithChannel(in_features,
                                                out_features,
                                                n_channels,
                                                is_first,
                                                share_weights, data_type=torch.float)
    def forward(self, input, indices):
        lin = self.linear(input, indices)
        if input.dtype == torch.complex64:
            lin2 = self.orth_scale(input.abs(), indices)
        else:
            lin2 = self.orth_scale(input, indices)
            
        omega = self.omega_0 * lin
        scale = self.scale_0 * lin
        orth_scale = self.scale_0 * lin2
        
        arg = 1j*omega-scale.abs().square()-orth_scale.square()
        
        return torch.exp(arg)
    
class AdaptiveMultiGabor2DLayer(nn.Module):
    '''
        Implements WIRE2D activations with multiple channel input.

    '''
    
    def __init__(self, in_features, out_features, n_channels,
                 is_first=False, share_weights=False,
                 omega_0=30, scale_0=5.0, rank=2):
        super().__init__()        
        # Divide features by 2 since we have two linear layers
        self.in_features = in_features
        self.omega_0 = omega_0
        self.scale_0 = scale_0

        if is_first:
            linlayer = AdaptiveLinearWithChannel
        else:
            linlayer = AdaptiveLRLinearWithChannel

        self.linear = linlayer(in_features,
                                out_features,
                                n_channels,
                                is_first,
                                share_weights,
                                rank=rank)
        
        self.orth_scale = linlayer(in_features,
                                    out_features,
                                    n_channels,
                                    is_first,
                                    share_weights,
                                    rank=rank)
    def forward(self, input, indices):
        lin = self.linear(input, indices)
        lin2 = self.orth_scale(input, indices)
        
        omega = self.omega_0 * lin
        scale = self.scale_0 * lin
        orth_scale = self.scale_0 * lin2
        
        arg = 1j*omega-scale.abs().square()-orth_scale.abs().square()
        
        return torch.exp(arg)
    
class AdaptiveMultiWIRELR(nn.Module):
    def __init__(self, in_features, hidden_features, 
                 hidden_layers, out_features, n_channels, outermost_linear=False, share_weights=False,
                 first_omega_0=30, hidden_omega_0=30.,
                 scale=10.0, const=1.0, mode='2d', rank=2):
        super().__init__()
        
        self.mode = mode
        
        if self.mode == '1d':
            self.nonlin = AdaptiveMultiGaborLayer
        elif self.mode == '2d':
            self.nonlin = AdaptiveMultiGabor2DLayer
            hidden_features = int(hidden_features/np.sqrt(2))

        if type(hidden_features) is int:
            hidden_features = [hidden_features]*hidden_layers
        
        hidden_features = [in_features] + hidden_features
        
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
                                scale_0=scale,
                                rank=rank))

        if outermost_linear:
            feat = hidden_features[-1]
            final_linear = AdaptiveLinearWithChannel(
                                    feat, 
                                    out_features,
                                    n_channels,
                                    is_first=False,
                                    share_weights=share_weights,
                                    const=const,
                                    bias=True)
                    
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
    
    def forward(self, inp, indices):            
        output = inp[indices, ...]

        for mod in self.net:
            output = mod(output, indices)
        return output.real
    
