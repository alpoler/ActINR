#!/usr/bin/env python

import os
import sys
from torch.functional import align_tensors
from torch.nn.modules.linear import Linear
import tqdm
import pdb
import math
from modules.siren_hyperinr import AdaptiveMultiSiren as hyper
import rff
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from modules.siren_hyperinr import AdaptiveMultiSiren as hyper_net
from PIL import Image
#from torchvision.transforms import Resize, Compose, ToTensor, Normalize
#from pytorch_wavelets import DWTForward, DWTInverse

import skimage
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import cv2
    
from modules.wire import AdaptiveMultiWIRE

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
    def __init__(self, in_features, sidelength=None, fn_samples=None, use_nyquist=True,nchunks=None,n_img=None):
        super().__init__()

        self.in_features = in_features
        hyper_network = hyper_net

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
            self.num_frequencies = 4
            if use_nyquist:
                self.num_frequencies = self.get_num_frequencies_nyquist(fn_samples)
        else:
            self.num_frequencies = 4

        self.out_dim =  2 * in_features * self.num_frequencies
        self.hyper_dist = hyper_network(in_features=1,
            out_features=self.out_dim,
            n_channels=nchunks,
            hidden_features=20,
            hidden_layers=3,
            nonlin='sine',
            outermost_linear=True,
            share_weights=False,
            first_omega_0=30,
            hidden_omega_0=30,
            n_img=n_img,
            hyper=True,pos_encode=True,scale=False
            )


    def get_num_frequencies_nyquist(self, samples):
        nyquist_rate = samples
        return int(math.floor(math.log(nyquist_rate, 2))+1)

    def forward(self, coords,t,indices=None):
        tile_n,chunk_no,coord_no,inp_dim= coords.shape
        coords = coords.view(coords.shape[0], -1, self.in_features)
        phase = self.hyper_dist(t,indices,t=None).permute(2,1,0,3)
        pe_list = []
        for i in range(self.num_frequencies):
            for j in range(self.in_features):
                c = coords[..., j]
                c = c.view(1,chunk_no,-1)
                sin = torch.unsqueeze(torch.sin((2 ** i) * np.pi * c + phase[...,j]),-1)
                cos = torch.unsqueeze(torch.cos((2 ** i) * np.pi * c + + phase[...,j]), -1)
                #sin = torch.unsqueeze(torch.sin((2 ** i) * np.pi * c + phase), -1)
                #cos = torch.unsqueeze(torch.cos((2 ** i) * np.pi * c), -1)
                pe_list += [sin,cos]
                #coords_pos_enc = torch.cat((coords_pos_enc, sin, cos), axis=-1)

        return torch.cat(pe_list, -1).reshape(-1,chunk_no,coord_no,self.out_dim) #coords_pos_enc.reshape(coords.shape[0], -1, self.out_dim)
    
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
                 is_first=False, share_weights=False, const=1.0,
                 bias=True, data_type=None, add_dim=False,n_img=None,rank=None,config=None):
        super(AdaptiveLinearWithChannel, self).__init__()
        

                
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
        #self.rff = rff.layers.GaussianEncoding(sigma=1.0,input_size=1,encoded_size=20)
       
    
        #self.total_epochs=0.75*config.epochs


        #hyper_network = hyper

        self.final = False
        weight = (0.1*torch.rand(1,w_nchan,input_size,output_size, dtype=torch.float)-0.1).to(rank) # interpolation
        self.weight = nn.Parameter(weight)

        bias = (0.1*torch.rand(1,channel_size,1,output_size, dtype=torch.float)-0.1).to(rank)  # interpolation


        self.bias = nn.Parameter(bias)
                        

        #self.reset_parameters(self.weight,self.bias)
        self.const = const
        
    @torch.no_grad()
    def reset_parameters(self,weight,bias):


        for i in range(weight.shape[0]):
            fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(weight[0,...])
            bound = np.sqrt(3/(fan_out))
            self.weight[i,...].uniform_(-bound, bound)
    
        for i in range(bias.shape[0]):
            fan_in , fan_out = torch.nn.init._calculate_fan_in_and_fan_out(weight[0,...])
            bound = np.sqrt(1/(fan_out))
            bias[i,...].uniform_(-bound,bound)

    @torch.no_grad()
    def reset_bias(self):
        _, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
        
        bound = np.sqrt(self.const/fan_out)
        self.bias.uniform_(-bound, bound)
        
    def forward(self, x, indices,t=None, ex_biases=None,coords=None,epochs=None):        

        t, model_idx = t

        weight = self.weight
        bias = self.bias

        #weight = weight[model_idx,...] # Interpolation
        #bias = bias[model_idx,...]


       # t = self.rff(t) #*0.1

        #t= torch.cat([t,bias.squeeze(-2).permute(1,0,2)],dim=-1)

        #hyper_shift = self.hyper_dist(t,indices,t=bias.squeeze(-2).permute(1,0,2)).permute(1, 0, 2).unsqueeze(-2)   # Global coordinates 

        output = torch.matmul(x, weight)


        return  output + bias  
    
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
                 omega_0=30, const=1.0,n_img=None,config=None):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.const = const
        
        self.in_features = in_features
        
       # if is_first:
        linlayer = AdaptiveLinearWithChannel
       # else:
       #     linlayer = AdaptiveLRLinearWithChannel

        self.linear = linlayer(in_features,
                                out_features,
                                n_channels,
                                share_weights,n_img=n_img,config=config)
        
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
                 omega_0=30, share_weights=None, const=None):
        super().__init__()        
        self.in_features = in_features
        self.linear = AdaptiveLinearWithChannel(in_features,
                                                out_features,
                                                n_channels)
        self.relu = torch.nn.LeakyReLU()
        
    def forward(self, input, indices):
        return self.relu(self.linear(input, indices))    
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
                 const=1.0,n_img=None,config=None):
        super().__init__()
        
        if nonlin == 'sine':
            self.nonlin = AdaptiveMultiSineLayer
        elif nonlin == 'relu':
            self.nonlin = AdaptiveMultiReLULayer
        self.n_img = n_img
                
        self.pos_encode = pos_encode
        if self.pos_encode:
            print('POS encoding')
            self.positional_encoding = PosEncoding(in_features=in_features,
                                                   sidelength=270,
                                                   fn_samples=None,
                                                   use_nyquist=True,
                                                   nchunks=n_channels,n_img=config.n_frames)
            in_features = self.positional_encoding.out_dim
        
        self.net = []
        self.net.append(self.nonlin(in_features, hidden_features, n_channels,
                                  is_first=True, omega_0=first_omega_0,
                                  share_weights=share_weights,
                                  const=const,n_img=n_img,config=config))

        for i in range(hidden_layers):
            self.net.append(self.nonlin(hidden_features,
                                        hidden_features, 
                                        n_channels, is_first=False, 
                                        share_weights=share_weights,
                                        omega_0=hidden_omega_0,
                                        const=const,n_img=n_img,config=config))

        if outermost_linear:
            feat = hidden_features
            final_linear = AdaptiveLinearWithChannel(
                                feat, 
                                out_features,
                                n_channels,
                                share_weights=share_weights,n_img=n_img,config=config)
            
            if nonlin == 'sine':
                with torch.no_grad():
                    bound = np.sqrt(const / hidden_features) / hidden_omega_0
                    final_linear.weight.uniform_(-bound, bound)
                    final_linear.bias.uniform_(-bound, bound)
                    
            self.net.append(final_linear)
        else:
            self.net.append(self.nonlin(
                hidden_features, out_features, 
                n_channels, is_first=False,
                share_weights=share_weights, 
                omega_0=hidden_omega_0,
                const=const))
        
        self.net = nn.ModuleList(self.net)
        
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
        output = inp[None, indices, ...] # 1, n_chunks, coordinates, output_size
        #left = torch.where(torch.floor(t+1e-6).long()<0,0,torch.floor(t+1e-6).long()) 
        #right= torch.where((left+1)>(self.n_img-1),self.n_img-1,left+1) # limit time coordinate to number of images
        #left_res = torch.where(left>=(self.n_img-1),1.,t-left)
        #right_res = torch.where(left>=(self.n_img-1),0.,right-t)
        #time = [left,right,left_res.view(-1,1,1,1),right_res.view(-1,1,1,1)]
        if self.pos_encode:
            output = self.positional_encoding(output,t[0],indices)

        for mod in self.net:
            output = mod(output, indices, t=t)
        return output
    

