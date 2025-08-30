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
from decoders import ConvDecoder
from bitEstimator import BitEstimator
from math import sqrt
#from siren import PosEncoding

class AdaptiveLinearWithChannel(nn.Module):
    '''
        Implementation from https://github.com/pytorch/pytorch/issues/36591
        
        Evaluate only selective channels
    '''
    def __init__(self, input_size, output_size, channel_size,weight_decoder,
                 is_first=False, share_weights=False, const=1.0,
                 bias=True, data_type=None, add_dim=False,n_img=None,rank=None,config=None,name=None):
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
        self.input_size = input_size
        self.output_size = output_size
        self.rank = rank
        self.w_nchan = w_nchan
        self.tiles_no = config.no_tiles
        self.name = name
        self.fan_in = self.output_size*self.w_nchan
        self.complex_layer = (dtype == torch.cfloat)
        self.fan_out = self.output_size*self.w_nchan
        self.target_var = sqrt(1/self.fan_out)
        self.rff = rff.layers.GaussianEncoding(sigma=10.0,input_size=1,encoded_size=40)        
        self.last_element = ((config.n_frames-1)//config.freq)*config.freq
        hyper_network = AdaptiveMultiSiren
        
        #initialize weights
        self.final = False
        
        # Weight 
        if dtype == torch.cfloat:
            self.decoder_real = weight_decoder[0]
            self.decoder_imag = weight_decoder[1]
            self.weight_real = nn.Parameter(torch.empty(w_nchan*input_size*output_size, 1,dtype=torch.float).to(rank))
            self.weight_imag =  nn.Parameter(torch.empty(w_nchan*input_size*output_size, 1,dtype=torch.float).to(rank))
        else:
            self.weight_decoder = weight_decoder
            self.weight = nn.Parameter(torch.empty(w_nchan*input_size*output_size, 1,dtype=torch.float).to(rank))
        if bias:
            if dtype == torch.cfloat:
                bias = (0.1*torch.rand(config.no_tiles,channel_size,1,10, dtype=torch.float)-0.1).to(rank) # interpolation
                self.bias = nn.Parameter(bias)
            else:
                bias = (0.1*torch.rand(config.no_tiles,channel_size,1,10, dtype=torch.float)-0.1).to(rank)
                self.bias = nn.Parameter(bias)

        else:
            self.bias =  torch.nn.Parameter(torch.zeros(1, 1, 1))     

        if dtype == torch.cfloat:    
            self.hyper_shift = hyper_network(in_features=80+10,
                        out_features=output_size*2,
                        n_channels=w_nchan,
                        hidden_features=self.output_size,
                        hidden_layers=1,
                        nonlin='relu',
                        outermost_linear=True,
                        share_weights=True,
                        first_omega_0=30,
                        hidden_omega_0=30,
                        n_img=n_img,
                        hyper=True,pos_encode=False,gating=input_size,scale=False
                        )
                
        else:
            self.hyper_dist = hyper_network(in_features=80+10,
                        out_features=output_size,
                        n_channels=w_nchan,
                        hidden_features=self.output_size,
                        hidden_layers=1,
                        nonlin='relu',
                        outermost_linear=True,
                        share_weights=True,
                        first_omega_0=30,
                        hidden_omega_0=30,
                        n_img=n_img,
                        hyper=True,pos_encode=False,gating=input_size,scale=False
                        )
        self.reset_parameters(self.bias)
        self.const = const
        
    @torch.no_grad()
    def reset_parameters(self,bias):
        bound = np.sqrt(3/(self.fan_out))
        for i in range(self.w_nchan*self.input_size*self.output_size):
            if self.complex_layer:
                self.weight_real[i,...].uniform_(-bound, bound)
                self.weight_imag[i,...].uniform_(-bound, bound)
            else:
                self.weight[i,...].uniform_(-bound, bound)
        for i in range(bias.shape[0]):
            if self.complex_layer:
                bound = np.sqrt(1/(self.fan_out))
                bias[i,...].uniform_(-bound,bound)
            else:
                bound = np.sqrt(1/(self.fan_out))
                bias[i,...].uniform_(-bound,bound)

    @torch.no_grad()
    def reset_bias(self):
        _, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
        
        bound = np.sqrt(self.const/fan_out)
        self.bias.uniform_(-bound, bound)
        
    def forward(self, x, indices,t=None, ex_biases=None,coords=None,epochs=None):        
        t, model_idx = t
        
        if self.complex_layer:
            w_real = self.decoder_real(self.weight_real).squeeze(-1)
            w_imag = self.decoder_imag(self.weight_imag).squeeze(-1)
            weight = torch.stack([w_real,w_imag],-1)
            weight = torch.view_as_complex(weight)
        else:
            weight = self.weight
            weight= self.weight_decoder(weight)
            weight = weight.squeeze(-1)

        weight = weight.view(1,self.w_nchan,self.input_size,self.output_size)
        bias = self.bias

        weight = weight[model_idx,...] 
        bias = bias[model_idx,...]

        t = self.rff(t)
        t= torch.cat([t,bias.squeeze(-2).permute(1,0,2)],dim=-1)
        if self.complex_layer:
            hyper_shift = self.hyper_shift(t,indices).permute(2, 1, 0, 3)
            size = hyper_shift.shape[-1]
            hyper_shift_real = hyper_shift[:,:,:,:(size//2)]
            hyper_shift_imag = hyper_shift[:,:,:,(size//2):]
            hyper_shift = torch.complex(hyper_shift_real,hyper_shift_imag)
        else:
            hyper_shift = self.hyper_dist(t,indices,t=None).permute(2, 1, 0, 3)  # Global coordinates 

        output = torch.matmul(x, weight)
        return  output + hyper_shift  
    
class AdaptiveMultiGabor2DLayer(nn.Module):
    '''
        Implements WIRE2D activations with multiple channel input.

    '''
    
    def __init__(self, in_features, out_features, n_channels,weight_decoder,
                 is_first=False, share_weights=False,
                 omega_0=30, scale_0=5.0,n_img=None,rank=None,config=None,name=None):
        super().__init__()        
        # Divide features by 2 since we have two linear layers
        self.in_features = in_features
        self.out_features = out_features
        self.omega_0 = omega_0 #nn.Parameter(omega_0*torch.ones(1),True)
        self.scale_0 =  scale_0 #.Parameter(scale_0*torch.ones(1),True)

        
        linlayer = AdaptiveLinearWithChannel


        self.linear = linlayer(in_features,
                                out_features,
                                n_channels,
                                weight_decoder["linear"],
                                is_first,
                                share_weights,
                                n_img=n_img,rank=rank,config=config,name=name)
        
        self.orth_scale = linlayer(in_features,
                                    out_features,
                                    n_channels,
                                    weight_decoder["orthogonal"],
                                    is_first,
                                    share_weights,
                                    n_img=n_img,rank=rank,config=config,name=name)

        
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

        if self.mode == '2d':
            self.nonlin = AdaptiveMultiGabor2DLayer
            hidden_features = int(hidden_features/np.sqrt(2))


        if type(hidden_features) is int:
            hidden_features = [hidden_features]*hidden_layers
        
        self.pos_encode = pos_encode

        
        #
        decoder_cfg = {'init_type': 'random', 'no_shift': True, 'decode_norm': 'min_max', 'decode_type': 'layer', 'decode_matrix': 'sq', 
         'num_hidden': 0, 'hidden_sizes': 12, 'nonlinearity': 'sine', 'boundary': 3.0, 'boundary_first': 200.0, 'unique_last': False,
           'unique_first': False, 'compress_first': True, 'std': 1.0}
        
        # Define weight decoders and entropy estimator

        weight_decoders, prob_models = {}, {}
        groups = {f'layer{k+1}': f'layer{k+1}' for k in range(0,hidden_layers+1) }        
        self.groups = groups
        self.unique_groups = list(groups.values())
        last_layer = "layer{}".format(hidden_layers+1)
        self.last_layer = last_layer
        for group in self.unique_groups:
            if "layer1" in group:
                # Real layer decoder initialization
                weight_decoders[group]= {"linear": ConvDecoder((1,1), -1, rank, **decoder_cfg),
                                         "orthogonal": ConvDecoder((1,1), -1, rank, **decoder_cfg)}

                prob_models[group]={"linear": BitEstimator(1,rank,False,num_layers=1),
                                    "orthogonal": BitEstimator(1,rank,False,num_layers=1)}
            elif last_layer in group:
                # COOOMPLEX LAST
                weight_decoders[group] = [ConvDecoder((1,1), -1,rank, **decoder_cfg),ConvDecoder((1,1), -1,rank, **decoder_cfg)]
                prob_models[group] = [BitEstimator(1,rank,False,num_layers=1),BitEstimator(1,rank,False,num_layers=1)]
            else:
                # Complex Layer initialization
                # We have orthogonal and linear layer with complex weights so our channel size is 2
                weight_decoders[group]= {"linear": [ConvDecoder((1,1), -1,rank, **decoder_cfg),ConvDecoder((1,1), -1,rank, **decoder_cfg)],
                                         "orthogonal": [ConvDecoder((1,1), -1,rank, **decoder_cfg),ConvDecoder((1,1), -1,rank, **decoder_cfg)]}

                prob_models[group]={"linear": [BitEstimator(1,rank,False,num_layers=1),BitEstimator(1,rank,False,num_layers=1)],
                                    "orthogonal": [BitEstimator(1,rank,False,num_layers=1),BitEstimator(1,rank,False,num_layers=1)]}

        self.weight_decoders = weight_decoders
        self.prob_models = prob_models

        hidden_features = [in_features] + hidden_features
        self.w_nchan = n_channels
        self.net = []
        for i in range(len(hidden_features)-1):
            group_name = "layer{}".format(i+1)
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
                                weight_decoder=self.weight_decoders[self.groups[group_name]],
                                is_first=is_first,
                                share_weights=sw, 
                                omega_0=omega_0,
                                scale_0=scale,n_img=n_img,rank=rank,config=config,name=self.groups[group_name]))

        if outermost_linear:
            group_name = last_layer
            feat = hidden_features[-1]
            final_linear = AdaptiveLinearWithChannel(
                                    feat, 
                                    out_features,
                                    n_channels,
                                    weight_decoder=self.weight_decoders[self.groups[group_name]],
                                    is_first=False,
                                    share_weights=share_weights,data_type=torch.float if self.mode == "complete_real" else None,
                                    const=const,
                                    bias=True,n_img=n_img,rank=rank,config=config,name=self.groups[group_name])
                    
            self.net.append(final_linear)
        
        self.net = nn.ModuleList(self.net)
        boundaries= self.calc_boundaries()
        self.reset_parameters(boundaries,config)


    def reset_parameters(self,boundaries,config):

        # Initialize scale such that rounded and transformed weights being kicked off with Kaiming
        # Assuming variance of b=1 where is the div take place just after following this block
        #-----------------------------------------------------------------------------------------------#
        for group_name in self.weight_decoders:
            weight_decoder = self.weight_decoders[group_name]
            min_std = boundaries[group_name]
            boundary = config.boundary_first if "layer1" in group_name else config.boundary
            init_type = 'random'
            if not (self.last_layer in group_name):
               
                lin = weight_decoder["linear"]
                ortho = weight_decoder["orthogonal"]
                if isinstance(lin,list):
                    for cur_lin,cur_ortho in zip(lin,ortho):
                        cur_lin.reset_parameters(init_type,min_std) 
                        cur_ortho.reset_parameters(init_type,min_std)
                else:
                    lin.reset_parameters(init_type,min_std)
                    ortho.reset_parameters(init_type,min_std)
            else:
                weight_decoder[0].reset_parameters(init_type,min_std)
                weight_decoder[1].reset_parameters(init_type,min_std)
        #------------------------------------------------------------------------------------------------#
        # Initialize contionous weight surrogate between pre-defined interval [-b,b]
        # Since its variance does not equal to 1, division follow infinity norm of weight to standardize before scaling
        for m in self.modules():
            if isinstance(m,AdaptiveLinearWithChannel):
                group_name = m.name
                if m.name == "layer1":
                    mult = config.boundary_first
                else:
                    mult = config.boundary
                if m.complex_layer:
                    nn.init.uniform_(m.weight_real,-mult,mult)
                    nn.init.uniform_(m.weight_imag,-mult,mult)
                else:
                    nn.init.uniform_(m.weight,-mult,mult)
        


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
                #self.net[idx].weight = \
                #    ref_model.net[idx].weight
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
    
    def calc_boundaries(self):
        boundaries={}
        for m in self.modules():
            if isinstance(m,AdaptiveLinearWithChannel):
                group_name = self.groups[m.name]
                w_std = m.target_var
                if group_name not in boundaries:
                    boundaries[group_name] = w_std
                else:
                    boundaries[group_name] = min(boundaries[group_name],w_std)
        return boundaries

    def get_latents(self):
        weights = {}
        for n,m in self.named_modules():
            if isinstance(m,AdaptiveLinearWithChannel):
                group_name = self.groups[m.name]
                
                if self.last_layer in group_name:
                    up_group_name = group_name
                    weights[group_name+"_real"] = m.weight_real
                    weights[group_name+"_imag"] = m.weight_imag
                else:
                    layer_type ="orthogonal" if n.endswith("orth_scale") else "linear"
                    up_group_name = group_name+"_"+layer_type
                    if "layer1" in group_name:
                        weights[up_group_name] = m.weight
                    else:
                        weights[up_group_name+"_real"] = m.weight_real
                        weights[up_group_name+"_imag"] = m.weight_imag

        return weights
