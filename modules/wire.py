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
    
#from siren import PosEncoding

class PosEncoding(nn.Module):
    '''Module to add positional encoding as in NeRF [Mildenhall et al. 2020].'''
    def __init__(self, in_features, sidelength=None, fn_samples=None, use_nyquist=True):
        super().__init__()

        self.in_features = in_features

        if self.in_features == 3:
            self.num_frequencies = 10
        elif self.in_features == 2:
            assert sidelength is not None
            if isinstance(sidelength, int):
                sidelength = (sidelength, sidelength)
            self.num_frequencies = 4
            if use_nyquist:
                self.num_frequencies = self.get_num_frequencies_nyquist(min(sidelength[0], sidelength[1]))
        elif self.in_features == 1:
            #assert fn_samples is not None
            self.num_frequencies = 4
            #if use_nyquist:
            #    self.num_frequencies = self.get_num_frequencies_nyquist(fn_samples)
        else:
            self.num_frequencies = 4

        self.out_dim = in_features + 2 * in_features * self.num_frequencies

    def get_num_frequencies_nyquist(self, samples):
        nyquist_rate = 1 / (2 * (2 * 1 / samples))
        return int(math.floor(math.log(nyquist_rate, 2)))

    def forward(self, coords):
        coords = coords.view(coords.shape[0], -1, self.in_features)

        coords_pos_enc = coords
        for i in range(self.num_frequencies):
            for j in range(self.in_features):
                c = coords[..., j]

                sin = torch.unsqueeze(torch.sin((2 ** i) * np.pi * c), -1)
                cos = torch.unsqueeze(torch.cos((2 ** i) * np.pi * c), -1)

                coords_pos_enc = torch.cat((coords_pos_enc, sin, cos), axis=-1)

        return coords_pos_enc.reshape(coords.shape[0], -1, self.out_dim)


class AdaptiveLinearWithChannel(nn.Module):
    '''
        Implementation from https://github.com/pytorch/pytorch/issues/36591
        
        Evaluate only selective channels
    '''
    def __init__(self, input_size, output_size, channel_size,
                 is_first=False, share_weights=False, const=1.0,
                 bias=True, data_type=None, add_dim=False,n_img=None):
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
        hyper_network = AdaptiveMultiSiren
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
        
        # if input_size !=2 and output_size!=3:
        #     weights_t = 0.01*torch.randn((n_img, 10),dtype=torch.float)
        #     matrix_t = 0.01*torch.randn((10, input_size*output_size),dtype=torch.cfloat)
        #     self.register_parameter('weights_t', torch.nn.Parameter(weights_t))
        #     self.register_parameter('matrix_t', torch.nn.Parameter(matrix_t))
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
            
        if dtype == torch.cfloat:
            # self.hyper_scale = hyper_network(in_features=1,
            #                         out_features=output_size*2,
            #                         n_channels=w_nchan,
            #                         hidden_features=64,
            #                         hidden_layers=3,
            #                         nonlin='relu',
            #                         outermost_linear=True,
            #                         share_weights=False,
            #                         first_omega_0=5,
            #                         hidden_omega_0=5,
            #                         n_img=n_img,
            #                         hyper=True,pos_encode=False,final_regularizer=input_size
            #                         )
            #self.hyper_shift = torch.nn.LSTM(input_size=1,hidden_size=32,num_layers=1,batch_first=True)
            #self.convertor = nn.Linear(in_features=32,out_features=output_size*2)
            self.hyper_shift = hyper_network(in_features=1,
                        out_features=output_size*2,
                        n_channels=w_nchan,
                        hidden_features=64,
                        hidden_layers=2,
                        nonlin='relu',
                        outermost_linear=True,
                        share_weights=False,
                        first_omega_0=5,
                        hidden_omega_0=5,
                        n_img=n_img,
                        hyper=True,pos_encode=False,final_regularizer=input_size
                        )
        else:
            
            #self.hyper_dist = torch.nn.LSTM(input_size=1,hidden_size=32,num_layers=1,batch_first=True)
            #self.convertor = nn.Linear(in_features=32,out_features=output_size)
            self.hyper_dist = hyper_network(in_features=1,
                        out_features=output_size,
                        n_channels=w_nchan,
                        hidden_features=64,
                        hidden_layers=2,
                        nonlin='relu',
                        outermost_linear=True,
                        share_weights=False,
                        first_omega_0=5,
                        hidden_omega_0=5,
                        n_img=n_img,
                        hyper=True,pos_encode=False,final_regularizer=input_size
                        )

    
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
        
    def forward(self, x, indices,t=None, ex_biases=None):        
        if self.share_weights:
            weight = self.weight
        else:
            weight = self.weight[None,indices, ...]
        
        # if hasattr(self,'weights_t'):
        #     output = torch.matmul(x, weight + self._get_delta_weight(t))
        # else:
        output = torch.matmul(x, weight)

        if ex_biases is None:
            if self.add_bias:
                if self.bias.dtype == torch.cfloat:
                    bias = self.bias[None,...]
                    #angle = bias.angle()
                    #abs_c = bias.abs()
                    #delta_angle = self.hyper_bias(t,indices).permute(1, 0, 2).unsqueeze(-2)
                    # size = delta_angle.shape[-1]
                    # real = delta_angle[:,:,:,:(size//2)]
                    # comp = delta_angle[:,:,:,(size//2):]
                    #hyper_scale = self.hyper_scale(t,indices).permute(1, 0, 2).unsqueeze(-2)
                    #size = hyper_scale.shape[-1]
                    #hyper_scale_real = hyper_scale[:,:,:,:(size//2)]
                    #hyper_scale_imag = hyper_scale[:,:,:,(size//2):]
                    #hyper_scale = torch.complex(hyper_scale_real,hyper_scale_imag)
                    hyper_shift = self.hyper_shift(t,indices).permute(1, 0, 2).unsqueeze(-2)
                    size = hyper_shift.shape[-1]
                    hyper_shift_real = hyper_shift[:,:,:,:(size//2)]
                    hyper_shift_imag = hyper_shift[:,:,:,(size//2):]
                    hyper_shift = torch.complex(hyper_shift_real,hyper_shift_imag)
                    #bias = torch.complex(hyper_real,hyper_complex)
                    #bias = bias + delta_bias
                    #angle_ = delta_angle + angle
                    #bias = torch.polar(abs_c,angle_)
                else:
                   
                    bias = self.bias[None,...]
                    hyper_shift = self.hyper_dist(t,indices).permute(1, 0, 2).unsqueeze(-2)
                    #size = hyper_mixed.shape[-1]
                    #hyper_scale = hyper_mixed[:,:,:,:(size//2)]
                    #hyper_shift = hyper_mixed[:,:,:,(size//2):]

                    #bias = bias + delta_bias
                

                output = output + bias + hyper_shift
        else:
            #print(output.shape)
            #print(self.bias.shape)
            #print(indices.shape, indices[:10])
            #print(ex_biases.shape)
            
            output = output + torch.unsqueeze(ex_biases, 1)[indices, ...]
            
        return output  # (1+hyper_scale)*
    
    def _get_delta_weight(self,t):
        
        weights_t = self.weights_t
        grid_query = t.view(1,-1,1,1)
        weights_t = torch.nn.functional.grid_sample(weights_t.transpose(0, 1).unsqueeze(0).unsqueeze(-1),
                                                    torch.cat([torch.zeros_like(grid_query), grid_query], dim=-1),
                                                    padding_mode='border',
                                                    mode='bilinear',
                                                    align_corners=True).squeeze(0).squeeze(-1).transpose(0,1)
        
        return (torch.complex(weights_t,torch.zeros_like(weights_t)) @ self.matrix_t).view(-1,1,self.weight.shape[1],self.weight.shape[2])
        
    
class AdaptiveLRLinearWithChannel(nn.Module):
    '''
        Implementation from https://github.com/pytorch/pytorch/issues/36591
        
        Evaluate only selective channels
    '''
    def __init__(self, input_size, output_size, channel_size,
                 is_first=False, share_weights=False, const=1.0,
                 bias=True, data_type=None, add_dim=False):
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

        self.rank = min(4, self.inp_sz*self.out_sz)
        
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
    
class AdaptiveLRLinearWithChannel(nn.Module):
    '''
        Implementation from https://github.com/pytorch/pytorch/issues/36591
        
        Evaluate only selective channels
    '''
    def __init__(self, input_size, output_size, channel_size,
                 is_first=False, share_weights=False, const=1.0,
                 bias=True, data_type=None, add_dim=False):
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

        self.rank = min(4, self.inp_sz*self.out_sz)
        
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
                                                share_weights,n_img=n_img)
        
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
                 omega_0=30, scale_0=5.0,n_img=None):
        super().__init__()        
        # Divide features by 2 since we have two linear layers
        self.in_features = in_features
        self.omega_0 = omega_0
        self.scale_0 = scale_0

        
        linlayer = AdaptiveLinearWithChannel


        self.linear = linlayer(in_features,
                                out_features,
                                n_channels,
                                is_first,
                                share_weights,n_img=n_img)
        
        self.orth_scale = linlayer(in_features,
                                    out_features,
                                    n_channels,
                                    is_first,
                                    share_weights,n_img=n_img)
    def forward(self, input, indices,t=None):
        lin = self.linear(input, indices,t)
        lin2 = self.orth_scale(input, indices,t)
        
        omega = self.omega_0 * lin
        scale = self.scale_0 * lin
        orth_scale = self.scale_0 * lin2
        
        arg = 1j*omega-scale.abs().square()-orth_scale.abs().square()
        
        return torch.exp(arg)
        
class AdaptiveMultiGabor2DLayer_Guha(nn.Module):
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
        self.is_first = is_first
        self.linear = AdaptiveLinearWithChannel(in_features,
                                                out_features,
                                                n_channels,
                                                is_first,
                                                share_weights)

        orth_nchan = 1 if self.is_first else 2
        self.orth_scale = AdaptiveLinearWithChannel(orth_nchan,
                                                    1,
                                                    n_channels,
                                                    is_first,
                                                    data_type=torch.float,
                                                    add_dim=True)
    def forward(self, input, indices):
        lin = self.linear(input, indices)
    
        if self.is_first:
            freq_stack = lin.real[..., None]
        else:
            freq_stack = torch.stack((lin.real, lin.imag), dim=-1)
        lin2 = self.orth_scale(freq_stack, indices)[..., 0]
        
        omega = self.omega_0 * lin
        scale = self.scale_0 * lin
        orth_scale = self.scale_0 * lin2
        
        arg = 1j*omega-scale.abs().square()-orth_scale.abs().square()
        
        return torch.exp(arg)
        
class AdaptiveMultiWIRE(nn.Module):
    def __init__(self, in_features, hidden_features, 
                 hidden_layers, out_features, n_channels, outermost_linear=False, share_weights=False,
                 first_omega_0=30, hidden_omega_0=30.,
                 scale=10.0, const=1.0, mode='2d',n_img=None,pos_encode=False):
        super().__init__()
        
        self.mode = mode
        
        if self.mode == '1d':
            self.nonlin = AdaptiveMultiGaborLayer
        elif self.mode == '2d':
            self.nonlin = AdaptiveMultiGabor2DLayer
            hidden_features = int(hidden_features/np.sqrt(2))
        elif self.mode == '2db':
            self.nonlin = AdaptiveMultiGabor2DLayer_real

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
            self.pos_encode = True
        '''
        self.pos_encode = pos_encode
        if self.pos_encode:
            print('POS encoding')
            self.positional_encoding = PositionalEncoding('1.25_80',n_chunks=n_channels)
            #in_features = self.positional_encoding.embed_length
        
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
                                scale_0=scale,n_img=n_img))

        if outermost_linear:
            feat = hidden_features[-1]
            final_linear = AdaptiveLinearWithChannel(
                                    feat, 
                                    out_features,
                                    n_channels,
                                    is_first=False,
                                    share_weights=share_weights,
                                    const=const,
                                    bias=True,n_img=n_img)
                    
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
                #self.net[idx].weight = \
                #    ref_model.net[idx].weight
                tmp=10
    
    def forward(self, inp, indices,t=None, ex_biases=None):            
        output = inp[None,indices, ...]
        
        #if self.pos_encode:
        #    output = self.positional_encoding(output)
        if self.pos_encode:
            t = self.positional_encoding(t)

        output_ls = []
        for i_mod, mod in enumerate(self.net):
            if ex_biases is not None and i_mod==0:
                output = mod(output, indices, ex_biases)
            else:
                output = mod(output, indices, t)
            
            output_ls.append(output)
        
        if output.dtype==torch.complex64:
            return output.real
        else:   
            return output
    
class PositionalEncoding(nn.Module):
    def __init__(self, pe_embed,n_chunks=None):
        super(PositionalEncoding, self).__init__()
        self.pe_embed = pe_embed.lower()
        self.n_chunks = n_chunks
        if self.pe_embed == 'none':
            self.embed_length = 1
        else:
            self.lbase, self.levels = [float(x) for x in pe_embed.split('_')]
            self.levels = int(self.levels)
            self.embed_length = 2 * self.levels

    def forward(self, pos):
        pos = pos[0,:,0]
        if self.pe_embed == 'none':
            return pos[:,None]
        else:
            pe_list = []
            for i in range(self.levels):
                temp_value = pos * self.lbase **(i) * math.pi
                pe_list += [torch.sin(temp_value), torch.cos(temp_value)]
            return torch.stack(pe_list, 1).unsqueeze(0)