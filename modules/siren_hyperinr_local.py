#!/usr/bin/env python

import os
import sys
from torch.functional import align_tensors
from torch.nn.modules.linear import Linear
import tqdm
import pdb
import math

import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from einops import rearrange
from PIL import Image
#from torchvision.transforms import Resize, Compose, ToTensor, Normalize
#from pytorch_wavelets import DWTForward, DWTInverse

import skimage
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import cv2
    
#from wire import AdaptiveMultiWIRE
class TanhLayer(nn.Module):
    '''
        Drop in repalcement for SineLayer but with Tanh nonlinearity
    '''
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        '''
            is_first, and omega_0 are not used.
        '''
        super().__init__()
        self.in_features = in_features
        self.omega_0 = omega_0
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features, bias=bias)
    
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
    def forward(self, input):
        return torch.tanh(self.linear(input))

class ReLULayer(nn.Module):
    '''
        Drop in replacement for SineLayer but with ReLU non linearity
    '''
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        '''
            is_first, and omega_0 are not used.
        '''
        super().__init__()
        self.in_features = in_features
        self.omega_0 = omega_0
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features, bias=bias)
    
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        return nn.functional.relu(self.linear(input))
    
class SincLayer(nn.Module):
    '''
        Instead of a sinusoid, utilize a sync nonlinearity
    '''
    
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.eps = 1e-3
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        denom = self.omega_0*self.linear(input)
        numer = torch.cos(denom)
        
        return numer/(1 + abs(denom).pow(2) )
    
class SineLayer(nn.Module):
    '''
        See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for
        discussion of omega_0.
    
        If is_first=True, omega_0 is a frequency factor which simply multiplies
        the activations before the nonlinearity. Different signals may require
        different omega_0 in the first layer - this is a hyperparameter.
    
        If is_first=False, then the weights will be divided by omega_0 so as to
        keep the magnitude of activations constant, but boost gradients to the
        weight matrix (see supplement Sec. 1.5)
    '''
    
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))
    
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
            assert fn_samples is not None
            self.num_frequencies = 10
            if use_nyquist:
                self.num_frequencies = self.get_num_frequencies_nyquist(fn_samples)
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
    
class Siren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, 
                 out_features, nonlinearity='sine', outermost_linear=False, first_omega_0=30, hidden_omega_0=30., pos_encode=False, 
                 sidelength=512, fn_samples=None, use_nyquist=True):
        super().__init__()
        self.pos_encode = pos_encode
        
        if nonlinearity == 'sine':
            self.nonlin = SineLayer
        elif nonlinearity == 'tanh':
            self.nonlin = TanhLayer
        elif nonlinearity == 'sinc':
            self.nonlin = SincLayer
        else:
            self.nonlin = ReLULayer
            
        if pos_encode:
            self.positional_encoding = PosEncoding(in_features=in_features,
                                                   sidelength=sidelength,
                                                   fn_samples=fn_samples,
                                                   use_nyquist=use_nyquist)
            in_features = self.positional_encoding.out_dim
            
        self.net = []
        self.net.append(self.nonlin(in_features, hidden_features, 
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(self.nonlin(hidden_features,
                                        hidden_features,
                                        is_first=False,
                                        omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, 
                                     out_features)
            
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0, 
                                            np.sqrt(6 / hidden_features) / hidden_omega_0)
                    
            self.net.append(final_linear)
        else:
            self.net.append(self.nonlin(hidden_features, out_features, 
                                      is_first=False, omega_0=hidden_omega_0))
        
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords):
        # allows to take derivative w.r.t. input
        #coords = coords.clone().detach().requires_grad_(True) 
        if self.pos_encode:
            coords = self.positional_encoding(coords)
            
        output = self.net(coords)
        #return output, coords    
        return output
    
class LinearWithChannel(nn.Module):
    '''
        Implementation from https://github.com/pytorch/pytorch/issues/36591
    '''
    def __init__(self, input_size, output_size, channel_size):
        super(LinearWithChannel, self).__init__()
        
        #initialize weights
        self.weight = torch.nn.Parameter(torch.zeros(channel_size,
                                                     input_size,
                                                     output_size))
        self.bias = torch.nn.Parameter(torch.zeros(channel_size,
                                                   1,
                                                   output_size))
        
        #change weights to kaiming
        self.reset_parameters(self.weight, self.bias)
        
    def reset_parameters(self, weights, bias):
        
        torch.nn.init.kaiming_uniform_(weights, a=math.sqrt(3))
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(weights)
        bound = 1 / math.sqrt(fan_in)
        torch.nn.init.uniform_(bias, -bound, bound)
    
    def forward(self, x):
        mul_output = torch.bmm(x, self.weight)
        return mul_output + self.bias
    
class AdaptiveLinearWithChannel(nn.Module):
    '''
        Implementation from https://github.com/pytorch/pytorch/issues/36591
        
        Evaluate only selective channels
    '''
    def __init__(self, input_size, output_size, channel_size,
                 share_weights=False,n_img=None):
        super(AdaptiveLinearWithChannel, self).__init__()
        

        
        if share_weights:
            w_nchan = 1
        else:
            w_nchan = channel_size
        self.share_weights = share_weights
        
        hyper_network = AdaptiveMultiSiren
        self.output_size=output_size
        self.input_size = input_size
        self.w_nchan = w_nchan
        self.n_img = n_img

        #initialize weights
        self.weight = torch.nn.Parameter(torch.zeros(w_nchan,
                                                     input_size,
                                                     output_size))
        self.bias = torch.nn.Parameter(torch.zeros(channel_size,
                                                   1,
                                                   output_size))
        
        #self.transformer= TransformerBlock(dim=output_size,heads=4,dim_head=output_size,mlp_dim=output_size,dropout=0,prenorm=False)
        # self.latents = torch.nn.Parameter(torch.zeros(channel_size,
        #                                     n_img,
        #                                     output_size))

        self.hyper_bias = hyper_network(in_features=1,
                                                    out_features=output_size*2,
                                                    n_channels=w_nchan,
                                                    hidden_features=10,
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
        self.reset_parameters(self.weight, self.bias)
        
    def reset_parameters(self, weights, bias):
        
        torch.nn.init.kaiming_uniform_(weights, a=math.sqrt(3))
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(weights)
        bound = 1 / math.sqrt(fan_in)
        torch.nn.init.uniform_(bias, -bound, bound)
        #torch.nn.init.uniform_(self.latents, -1, 1)
    
    def forward(self, x, indices,t):
        if self.share_weights:
            weight = self.weight
        else:
            weight = self.weight[None,indices, ...]
        # x_size = 144 (n_chunks) x 640 (n_coordinates) x  20 (output_size) 
        # 1 ,    144,  640,  20
        # n_img, 144,    1,  20   
        # bias _ size = chunks , 1, out_features
        
        # Unnormalize t
        # t_unnorm = (t+1)*(self.n_img-1)*0.5
        # t_l = torch.clip(torch.floor(t_unnorm+1e-6).long(),min=0,max=self.n_img-1)
        # t_r = torch.clip(t_l+1,min=0,max=self.n_img-1)
        # t_r_margin = torch.where(t_l>=(self.n_img-1),1,t_r - t_unnorm)
        # t_l_margin= torch.where(t_l>=(self.n_img-1),0,t_unnorm - t_l)
        # latents_floor=self.latents[[[i] for i in range(self.w_nchan)],t_l.squeeze(-1).tolist(),:]
        # latents_ceil=self.latents[[[i] for i in range(self.w_nchan)],t_r.squeeze(-1).tolist(),:]
        # latents_eff = latents_floor*t_r_margin + latents_ceil*t_l_margin
        #eff_inp = torch.cat([t,latents_eff],dim=-1)
        dist_char = self.hyper_bias(t,indices).permute(1, 0, 2).unsqueeze(-2) # n_img , n_chunks , out_features
        f_dim = dist_char.shape[-1] //2
        alpha= dist_char[...,:f_dim]
        beta= dist_char[...,f_dim:] # n_img x n_chunks, 1, out_features # self.bias = 1x n_chunks x 1 x 20
        #N,_,C = self.bias.shape
        #bias_at = self.transformer(self.bias.view(_,N,C)).view(N,_,C)

        out = torch.matmul(x, weight) + (self.bias[None,indices,:]+beta)
        #B,W,N,C = out.shape
        #out = nn.functional.instance_norm(out.view(B*W,N,C).permute(0,2,1)).permute(0,2,1).view(B,W,N,C)
        out = (1+alpha)*out

        #delta_bias = ((self.delta_bias[t[0],...])*t[3])+((self.delta_bias[t[1],...])*t[2])
        return out #torch.matmul(x, weight) + (self.bias[None, indices, ...])
    
class AdaptiveLinearWithChannelHyper(nn.Module):
    '''
        Implementation from https://github.com/pytorch/pytorch/issues/36591
        
        Evaluate only selective channels
    '''
    def __init__(self, input_size, output_size, channel_size,
                 share_weights=False,n_img=None,config=None):
        super(AdaptiveLinearWithChannelHyper, self).__init__()
        
        if share_weights:
            w_nchan = 1
        else:
            w_nchan = channel_size
        self.share_weights = share_weights
        
        #initialize weights
        self.weight = torch.nn.Parameter(torch.zeros(w_nchan,config.no_tiles,
                                                     input_size,
                                                     output_size))
        self.bias = torch.nn.Parameter(torch.zeros(channel_size,config.no_tiles,
                                                   1,
                                                   output_size))
        
        #change weights to kaiming
        self.reset_parameters(self.weight, self.bias)
    @torch.no_grad()
    def reset_parameters(self, weights, bias):
        
        weights.uniform_(-0.05,0.05)
        bias.uniform_(-0.05,0.05)
        #for l in range(weights.shape[1]):
        #    torch.nn.init.kaiming_uniform_(weights[:,l,...], a=math.sqrt(3))
        #    fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(weights[:,l,...])
        #    bound = 1 / math.sqrt(fan_in)
        #    torch.nn.init.uniform_(bias[:,l,...], -bound, bound)
    
    def forward(self, x, indices,t=None):
        if self.share_weights:
            weight = self.weight
        else:
            weight = self.weight[:,t,...][indices, ...]
        
        return torch.matmul(x, weight) + self.bias[:,t,...][indices, ...]
    

    
class AdaptiveLRLinearWithChannel(nn.Module):
    '''
        Implementation from https://github.com/pytorch/pytorch/issues/36591
        
        Evaluate only selective channels
    '''
    def __init__(self, input_size, output_size, channel_size,
                 share_weights=False):
        super(AdaptiveLRLinearWithChannel, self).__init__()
        
        
        dtype = torch.float
       
        w_nchan = channel_size
        self.share_weights = share_weights

        self.inp_sz = input_size
        self.out_sz = output_size

        self.rank = min(60, self.inp_sz*self.out_sz)
        
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
        
        self.bias = torch.nn.Parameter(torch.zeros(
            channel_size,
            1,
            output_size,
            dtype=dtype))
                
        #change weights to kaiming
        self.reset_parameters(1.0)
        self.const = 1.0
        
    @torch.no_grad()
    def reset_parameters(self, const):
        _, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(self.weights_U)
        bound = np.sqrt(const*3/fan_out)
        self.weights_U.uniform_(-bound, bound)

        _, fan_out2 = torch.nn.init._calculate_fan_in_and_fan_out(self.weights_V)
        bound = np.sqrt(const/fan_out2)
        self.weights_V.uniform_(-bound, bound)
        
        bound = np.sqrt(const/fan_out)
        
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
        
        output = output + self.bias[indices, ...]
        
        return output
    
class MultiSineLayer(nn.Module):
    '''
        Implements sinusoidal activations with multiple channel input
    '''
    
    def __init__(self, in_features, out_features, n_channels, is_first=False, 
                 omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.in_features = in_features
        
        
        self.linear = LinearWithChannel(in_features, out_features, n_channels)
        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))
    
class AdaptiveMultiSineLayer(nn.Module):
    '''
        Implements sinusoidal activations with multiple channel input
    '''
    
    def __init__(self, in_features, out_features, n_channels,
                 is_first=False, share_weights=False,
                 omega_0=30, const=1.0,n_img=None,hyper=False):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.const = const
        
        self.in_features = in_features
        
       # if is_first:
        if hyper==False:       
            linlayer = AdaptiveLinearWithChannel
        if hyper==True:
            linlayer = AdaptiveLinearWithChannelHyper
       # else:
       #     linlayer = AdaptiveLRLinearWithChannel

        self.linear = linlayer(in_features,
                                out_features,
                                n_channels,
                                share_weights,n_img=n_img)
        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                bound = self.const/self.in_features
                self.linear.weight.uniform_(-bound, bound)      
                self.linear.bias.uniform_(-bound, bound)
                
            else:
                bound = np.sqrt(self.const*6 / self.in_features) / self.omega_0
                self.linear.weight.uniform_(-bound, bound)
                self.linear.bias.uniform_(-bound, bound)
                
        
    def forward(self, input, indices,t):
        return torch.sin(self.omega_0 * self.linear(input, indices,t))    
        
        
class AdaptiveMultiReLULayer(nn.Module):
    '''
        Implements ReLU activations with multiple channel input.
        
        The parameters is_first, and omega_0 are not relevant.
    '''
    
    def __init__(self, in_features, out_features, n_channels, is_first=False, 
                 omega_0=30, share_weights=None, const=None,n_img=None,hyper=False,config=None):
        super().__init__()        
        self.in_features = in_features
        self.linear = AdaptiveLinearWithChannelHyper(in_features,
                                                out_features,
                                                n_channels,n_img=n_img,config=config)
        self.scale = 10
        #self.relu = torch.nn.GELU()
        
    def forward(self, input, indices,t):
        return torch.exp(-(self.scale * self.linear(input,indices,t))**2)
        #return self.relu(self.linear(input, indices,t))    
    
class MultiSiren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, 
                 out_features, n_channels, outermost_linear=False, first_omega_0=30, hidden_omega_0=30.):
        super().__init__()
        
        self.nonlin = MultiSineLayer
            
        self.net = []
        self.net.append(self.nonlin(in_features, hidden_features, n_channels,
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(self.nonlin(hidden_features,
                                        hidden_features, 
                                        n_channels, is_first=False, 
                                        omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = LinearWithChannel(hidden_features,
                                             out_features,
                                             n_channels)
            
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0, 
                                            np.sqrt(6 / hidden_features) / hidden_omega_0)
                    
            self.net.append(final_linear)
        else:
            self.net.append(self.nonlin(hidden_features, out_features, 
                                        n_channels, is_first=False, 
                                        omega_0=hidden_omega_0))
        
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords):            
        output = self.net(coords) 
        return output
    
class MultiSequential(nn.Sequential):
    '''
        https://github.com/pytorch/pytorch/issues/19808#
    '''
    def forward(self, *input):
        for module in self._modules.values():
            input = module(*input)
        return input
    
class AdaptiveMultiSiren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, 
                 out_features, n_channels, outermost_linear=False, share_weights=False,
                 first_omega_0=30, hidden_omega_0=30., nonlin='sine', pos_encode=False,
                 const=1.0,n_img=None,hyper=False,gating=None,config=None):
        super().__init__()
        
        if nonlin == 'sine':
            self.nonlin = AdaptiveMultiSineLayer
        elif nonlin == 'relu':
            self.nonlin = AdaptiveMultiReLULayer
        self.n_img = n_img
        self.hyper= hyper
        self.pos_encode = pos_encode
        if self.pos_encode:
            print('POS encoding')
            self.positional_encoding = PositionalEncoding('1.25_80',n_chunks=n_channels)
            in_features = self.positional_encoding.embed_length
        
        self.net = []
        self.net.append(self.nonlin(in_features, hidden_features, n_channels,
                                  is_first=True, omega_0=first_omega_0,
                                  share_weights=share_weights,
                                  const=const,n_img=n_img,hyper=hyper,config=config))

        for i in range(hidden_layers):
            self.net.append(self.nonlin(hidden_features,
                                        hidden_features, 
                                        n_channels, is_first=False, 
                                        share_weights=share_weights,
                                        omega_0=hidden_omega_0,
                                        const=const,n_img=n_img,hyper=hyper,config=config))

        if outermost_linear:
            feat = hidden_features
            if hyper == False:
                final_linear = AdaptiveLinearWithChannel(
                                    feat, 
                                    out_features,
                                    n_channels,
                                    share_weights=share_weights,n_img=n_img,config=config)
            if hyper == True:
                final_linear = AdaptiveLinearWithChannelHyper(
                    feat, 
                    out_features,
                    n_channels,
                    share_weights=share_weights,n_img=n_img,config=config)
            
            
            with torch.no_grad():
                if hyper==False:
                    if nonlin=='sine':
                        bound = np.sqrt(const / hidden_features) / hidden_omega_0
                        final_linear.weight.uniform_(-bound, bound)
                        final_linear.bias.uniform_(-bound, bound)
                        
                else:
                    #bound = np.sqrt(6/(hidden_features*(final_regularizer))) / hidden_omega_0
                    #target_var = (uni_bound**2)/3
                    #bound = np.sqrt((const*0.5) / hidden_features) / hidden_omega_0
                    pass
                    #bound = np.sqrt(const / hidden_features) / hidden_omega_0
                    #final_linear.weight.uniform_(-bound, bound)
                    #final_linear.bias.uniform_(-bound, bound)
                    
                    #final_linear.weight.data = 1e-2*final_linear.weight.data
                    #final_linear.bias.data = 1e-2*final_linear.bias.data

                    
            self.net.append(final_linear)
        else:
            self.net.append(self.nonlin(
                hidden_features, out_features, 
                n_channels, is_first=False,
                share_weights=share_weights, 
                omega_0=hidden_omega_0,
                const=const))
        
        self.net = nn.ModuleList(self.net)

    def hyper_weight_init(self,m, in_features_main_net):
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='linear', mode='fan_in')
            m.weight.data = m.weight.data / 1.e1

        if hasattr(m, 'bias'):
            with torch.no_grad():
                m.bias.uniform_(-1/in_features_main_net, 1/in_features_main_net)    

    def bias_parameters(self):
        '''
            Wrapper to get only bias parameters
        '''
        params = []
        nparams = 0
        for module in self.net:
            try:
                params.append(module.linear.bias)
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
        
            except AttributeError:
                self.net[idx].weight = \
                    ref_model.net[idx].weight
    
    
    def forward(self, inp, indices,t=None): 
        if self.hyper:           
            output = inp[:,:,None,:] # 1, n_chunks, coordinates, output_size
        else:
            output = inp[None,indices,...]
        #left = torch.where(torch.floor(t+1e-6).long()<0,0,torch.floor(t+1e-6).long()) 
        #right= torch.where((left+1)>(self.n_img-1),self.n_img-1,left+1) # limit time coordinate to number of images
        #left_res = torch.where(left>=(self.n_img-1),1.,t-left)
        #right_res = torch.where(left>=(self.n_img-1),0.,right-t)
        #time = [left,right,left_res.view(-1,1,1,1),right_res.view(-1,1,1,1)]
        if self.pos_encode:
            output = self.positional_encoding(output)

        for mod in self.net:
            output = mod(output, indices, t=t)
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
            return torch.repeat_interleave(torch.stack(pe_list, 1).unsqueeze(0),repeats=self.n_chunks,dim=0)
        

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = heads * dim_head
        project_out = not(heads==1 and dim_head==dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
    
    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0., prenorm=False):
        super(TransformerBlock, self).__init__()
        if prenorm:
            self.attn = PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))
            self.ffn = PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
        else:
            self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
            self.ffn = FeedForward(dim, mlp_dim, dropout=dropout)
    def forward(self, x):
        x = self.attn(x) + x
        x = self.ffn(x) + x
        return x