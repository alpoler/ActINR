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
import time
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
        self.share_weights = share_weights
        self.add_bias = bias
        self.input_size = input_size
        self.output_size = output_size
        self.rank = rank
        self.w_nchan = w_nchan
        self.tiles_no = config.no_tiles
        self.rff = rff.layers.GaussianEncoding(sigma=1.0,input_size=1,encoded_size=40)
       
        #self.rff_s = rff.layers.GaussianEncoding(sigma=1,input_size=1,encoded_size=40)

        
        self.motion_smoothing = config.smoothing
        self.total_epochs=0.75*config.epochs
        #self.t_enc = PositionalEncoding('1.05_5',n_chunks=w_nchan)
        #if self.output_size!=3:
        #    self.affine = nn.Parameter(torch.ones(w_nchan,self.tiles_no,12,12))
        self.last_element = ((config.n_frames-1)//config.freq)*config.freq
        hyper_network = AdaptiveMultiSiren
        self.complex_layer = (dtype == torch.cfloat)
        #initialize weights
        self.final = False
        if add_dim:
            self.weight = torch.nn.Parameter(torch.zeros(
                            w_nchan,
                            1,
                            input_size,
                            output_size,
                            dtype=dtype))
        else:
            if dtype == torch.cfloat:
                if output_size !=3:
                    weight = (0.1*torch.rand(config.no_tiles,w_nchan,input_size,output_size, dtype=torch.cfloat)-0.1).to(rank) # interpolation
                else:
                    weight = (0.1*torch.rand(config.no_tiles,w_nchan,input_size,output_size, dtype=torch.cfloat)-0.1).to(rank) # interpolation
                # weight = (0.1*torch.rand(1,w_nchan,input_size,output_size, dtype=torch.cfloat)-0.1).to(rank) # Regression



                #if self.output_size==3:
                #    self.final = True
                #    weight = (0.1*torch.rand(5,w_nchan,input_size,output_size, dtype=torch.cfloat)-0.1).to(rank)
                #    self.weight = nn.Parameter(torch.view_as_real(weight))
                #else:
                #    weight_q = torch.rand(w_nchan*5,w_nchan, dtype=torch.cfloat).to(rank)
                #    self.weight_q = nn.Parameter(torch.view_as_real(weight_q))
                #    weight_r = torch.rand(w_nchan,input_size*output_size,dtype=torch.cfloat).to(rank)
                #    self.weight_r = nn.Parameter(torch.view_as_real(weight_r))
                # 400 x 1440 = 400 x 200 @ 200x 1440 -> (chunksX t) x low_rank_no) @ (low_rank_no x 400)

                self.weight = nn.Parameter(torch.view_as_real(weight))
            else:
                if output_size !=3:
                    weight = (0.1*torch.rand(config.no_tiles,w_nchan,input_size,output_size, dtype=torch.float)-0.1).to(rank) # interpolation
                else:
                    weight = (0.1*torch.rand(config.no_tiles,w_nchan,input_size,output_size, dtype=torch.float)-0.1).to(rank) # interpolation
                # weight = (0.1*torch.rand(1,w_nchan,input_size,output_size, dtype=torch.float)-0.1).to(rank) # regression
                #weight_q = torch.rand(w_nchan*5,w_nchan, dtype=torch.float).to(rank)
                #self.weight_q = nn.Parameter(weight_q)
                #weight_r = torch.rand(w_nchan,input_size*output_size,dtype=torch.float).to(rank)
                #self.weight_r = nn.Parameter(weight_r)


                self.weight = nn.Parameter(weight)
            # self.weight = torch.nn.Parameter(torch.zeros(2,
            #                     w_nchan,
            #                     input_size,
            #                     output_size,
            #                     dtype=dtype))
        
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
                if dtype == torch.cfloat:
                    #pass
                    #if output_size!=3 and input_size!=2:
                    #if input_size==2:
                    bias = (0.1*torch.rand(config.no_tiles,channel_size,1,10, dtype=torch.float)-0.1).to(rank) # interpolation
                    self.bias = nn.Parameter(bias)
                    #else:
                    
                    #bias = (0.1*torch.rand(config.no_tiles,channel_size,1,output_size, dtype=torch.cfloat)-0.1).to(rank) # interpolation
                    #bias = (0.1*torch.rand(config.no_tiles,channel_size,1,10, dtype=torch.float)-0.1).to(rank) # interpolation
                    #self.bias = nn.Parameter(torch.view_as_real(bias))
                    #bias2 = (torch.randn(channel_size,config.no_tiles,self.output_size, dtype=torch.float)).to(rank) # interpolation 
                    #self.bias = nn.Parameter(bias)
                    #self.bias2 = nn.Parameter(bias2)
                    #else:
                    #    bias = (0.1*torch.rand(w_nchan,1,output_size, dtype=torch.float)-0.1).to(rank)# interpolation
                    #    self.bias = nn.Parameter(bias)
                    # bias = (0.1*torch.rand(config.no_tiles,channel_size,1,output_size, dtype=torch.cfloat)-0.1).to(rank) # regresiion
                    # self.bias = nn.Parameter(torch.view_as_real(bias))
                    #
                    
                else:
                    #pass
                    #if output_size!=3 and input_size!=2:
                    if input_size!=2:
                        bias = (0.1*torch.rand(config.no_tiles,channel_size,1,10, dtype=torch.float)-0.1).to(rank)  # interpolation
                    else:
                        bias = (0.1*torch.rand(config.no_tiles,channel_size,1,10, dtype=torch.float)-0.1).to(rank)
                        #bias2 = (torch.randn(channel_size,config.no_tiles,self.output_size, dtype=torch.float)).to(rank) 
                        #bias = (0.1*torch.rand(config.no_tiles,channel_size,1,output_size, dtype=torch.float)-0.1).to(rank) # regression
                        self.bias = nn.Parameter(bias)
                        #self.  bias2 = nn.Parameter(bias2)
                    #else:
                    #bias = (0.1*torch.rand(w_nchan,1,output_size, dtype=torch.float)-0.1).to(rank)# interpolation
                    self.bias = nn.Parameter(bias)
                        
                    
                # self.bias = torch.nn.Parameter(torch.zeros(2,
                #                                    channel_size,
                #                                    1,
                #                                    output_size,
                #                                    dtype=dtype))
        else:
            self.bias =  torch.nn.Parameter(torch.zeros(1, 1, 1))
            
        if dtype == torch.cfloat:
            # pass
            # self.hyper_scale = hyper_network(in_features=90,
            #                         out_features=output_size*5,
            #                         n_channels=w_nchan,
            #                         hidden_features=self.output_size,
            #                         hidden_layers=1,
            #                         nonlin='relu',
            #                         outermost_linear=True,
            #                         share_weights=True, 
            #                         first_omega_0=5,
            #                         hidden_omega_0=5,
            #                         n_img=n_img,
            #                         hyper=True,pos_encode=False,gating=input_size,scale=True
            #                         )
            



            #self.hyper_shift = torch.nn.LSTM(input_size=1,hidden_size=32,num_layers=1,batch_first=True)
            #self.convertor = nn.Linear(in_features=32,out_features=output_size*2)

            #self.hyper_shift = nn.Sequential(nn.Linear())

            #pass
            
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
            # pass
            #self.hyper_dist = torch.nn.LSTM(input_size=1,hidden_size=32,num_layers=1,batch_first=True)
            #self.convertor = nn.Linear(in_features=32,out_features=output_size)
            
            #pass

            # self.hyper_scale = hyper_network(in_features=90,
            #             out_features=output_size,
            #             n_channels=w_nchan,
            #             hidden_features=self.output_size,
            #             hidden_layers=1,
            #             nonlin='relu',
            #             outermost_linear=True,
            #             share_weights=True, 
            #             first_omega_0=5,
            #             hidden_omega_0=5,
            #             n_img=n_img,
            #             hyper=True,pos_encode=False,gating=input_size,scale=True
            #             )


            
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
           
        # if output_size!=3 and input_size!=2:
        #     self.rff_p = rff.layers.GaussianEncoding(sigma=10,input_size=output_size,encoded_size=output_size*5)
        #     self.hyper_low_a = hyper_network(in_features=output_size*10,
        #                     out_features=output_size*10,
        #                     n_channels=w_nchan,
        #                     hidden_features=self.output_size,
        #                     hidden_layers=1,
        #                     nonlin='relu',
        #                     outermost_linear=True,
        #                     share_weights=True, 
        #                     first_omega_0=5,
        #                     hidden_omega_0=5,
        #                     n_img=n_img,
        #                     hyper=True,pos_encode=False,gating=input_size,scale=False
        #                     )
            
        #     self.hyper_low_b = hyper_network(in_features=output_size*10,
        #             out_features=output_size*10,
        #             n_channels=w_nchan,
        #             hidden_features=self.output_size,
        #             hidden_layers=1,
        #             nonlin='relu',
        #             outermost_linear=True,
        #             share_weights=True, 
        #             first_omega_0=5,
        #             hidden_omega_0=5,
        #             n_img=n_img,
        #             hyper=True,pos_encode=False,gating=input_size,scale=False
        #             )


        #latent = (0.1*torch.rand(config.no_tiles,channel_size,1,25, dtype=torch.float)-0.1).to(rank) # interpolation
        #self.latent = nn.Parameter(latent)
        #change weights to kaiming
        self.reset_parameters(self.weight,self.bias)
        self.const = const
        
    @torch.no_grad()
    def reset_parameters(self,weight,bias):
        # if not self.complex_layer:
        #     m,l,k,n = weight.shape
        #     identity = torch.eye(k,n).unsqueeze(0).unsqueeze(0).expand(m,l,k,n)
        # else:
        #     m,l,k,n,r = weight.shape
        #     identity = torch.eye(k,n,dtype=torch.cfloat).unsqueeze(0).unsqueeze(0).expand(m,l,k,n)
        #     identity = torch.view_as_real(identity)
       
        # weight.data = identity
        # if self.input_size == 2:
        #     bias.uniform_(-1,1)
        # else: 
        #     bias.zero_()

        #_, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(weight[0,...] if not self.complex_layer else weight[0,...,0])

        
        
        #if self.input_size!=2:
        #bound = np.sqrt(1*3/fan_out)
        #weight.uniform_(-bound, bound)
        #bound = np.sqrt(1/fan_out)
        #self.bias.uniform_(-bound, bound)

        # else:
        #     m,l,k,n = weight.shape
        #     self.bias.uniform_(-1,1)
        #     identity = torch.eye(k,n).unsqueeze(0).unsqueeze(0).expand(m,l,k,n)
        #     weight.data = identity


        for i in range(weight.shape[0]):
            if self.complex_layer:
                fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(torch.view_as_complex(weight[0,...]))
                bound = np.sqrt(3/(fan_out))
                self.weight[i,...].uniform_(-bound, bound)
            else:
                fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(weight[0,...])
                bound = np.sqrt(3/(fan_out))
                self.weight[i,...].uniform_(-bound, bound)
        
        for i in range(bias.shape[0]):
            if self.complex_layer:
                fan_in , fan_out = torch.nn.init._calculate_fan_in_and_fan_out(torch.view_as_complex(weight[0,...]))
                bound = np.sqrt(1/(fan_out))
     
                bias[i,...].uniform_(-bound,bound)

            else:
                fan_in , fan_out = torch.nn.init._calculate_fan_in_and_fan_out(weight[0,...])
                bound = np.sqrt(1/(fan_out))
                bias[i,...].uniform_(-bound,bound)
                #if i==0:
                #if self.input_size==2:
                #    bias[i,...].uniform_(-1,1)

    @torch.no_grad()
    def reset_bias(self):
        _, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
        
        bound = np.sqrt(self.const/fan_out)
        self.bias.uniform_(-bound, bound)
        
    def forward(self, x, indices,t=None, ex_biases=None,coords=None,epochs=None):        

        t, model_idx = t
        #t = self.t_enc(t)
        #t= torch.repeat_interleave(t,repeats=self.w_nchan,dim=0)
        ##t= torch.cat([t,t_emb],dim=-1)
        #t_s = self.rff_s(t)
        #freq_mask = get_freq_mask(40,epochs,self.total_epochs,self.rank)
        #t = t*freq_mask.unsqueeze(0).unsqueeze(0)

        
        # if self.output_size!=3:
        #     affine = nn.functional.grid_sample(self.affine, coords, padding_mode='border', align_corners=True)
        #     affine = affine.view(self.w_nchan,self.tiles_no,96*96,1)
        #     affine = affine[:,model_idx,...].permute(1,0,2,3)
        #affine = nn.functional.grid_sample(self.affine, coords, padding_mode='border', align_corners=True).view(2,-1,1)
        if self.complex_layer:
            #if not self.final:
            #    weight_q = torch.view_as_complex(self.weight_q)
            #    weight_r = torch.view_as_complex(self.weight_r)
            #    weight = torch.matmul(weight_q,weight_r).view(-1,self.w_nchan,self.input_size,self.output_size)
            weight = torch.view_as_complex(self.weight)
            bias = self.bias
            # bias = torch.view_as_complex(self.bias)
            #else:
            #    weight = torch.view_as_complex(self.weight)
        else:
            
            weight = self.weight
            bias = self.bias

        # weight = weight[0,...] # Regression
        weight = weight[model_idx,...] # Interpolation
        if self.motion_smoothing:
            t,model_idx = self.expand_time(t)
        bias = bias[model_idx,...]
        #if self.output_size!=3 and self.input_size!=2:
            #bias2 = self.rff_p(self.bias2[:,model_idx,...])
            #a = self.hyper_low_a(bias2,indices).view(self.w_nchan,len(model_idx),self.output_size,-1).permute(1,0,2,3)
            #b = self.hyper_low_b(bias2,indices).view(self.w_nchan,len(model_idx),-1,self.output_size).permute(1,0,2,3)
            #modulator = torch.matmul(a,b)
            #weight = weight + modulator
        
        
        #slatent = self.latent[model_idx,...]
        t = self.rff(t) #*0.1

        t= torch.cat([t,bias.squeeze(-2).permute(1,0,2)],dim=-1)
        if self.complex_layer:
            #hyper_shift = self.hyper_shift(t_,indices,t=model_idx).permute(1,0,2,3) # Local Coordinates 
            hyper_shift = self.hyper_shift(t,indices).permute(2, 1, 0, 3) # Global Coordinates
            size = hyper_shift.shape[-1]
            hyper_shift_real = hyper_shift[:,:,:,:(size//2)]
            hyper_shift_imag = hyper_shift[:,:,:,(size//2):]
            hyper_shift = torch.complex(hyper_shift_real,hyper_shift_imag)

        else:

            #hyper_shift = self.hyper_dist(t_,indices,t=model_idx).permute(1,0,2,3)     # Local coordinates
            hyper_shift = self.hyper_dist(t,indices,t=None).permute(2, 1, 0, 3)  # Global coordinates 


        output = torch.matmul(x, weight)
    
        #if self.complex_layer:
        #   bias = torch.view_as_complex(bias)


        #hyper_scale = self.hyper_scale(t_,indices).permute(1, 0, 2).unsqueeze(-2)
        #hyper_scale = torch.complex(hyper_scale,torch.zeros_like(hyper_scale))
        return  output + hyper_shift  #+ bias # affine*(output +  hyper_shift) if self.output_size!=3 else   + bias 

    def expand_time(self,t):
        delta_t = 1/self.last_element
        expanded_t = torch.cat([t-2*delta_t,t-delta_t,t,t+delta_t,t+2*delta_t],dim=1)
        model_idx = torch.clamp(torch.floor((self.tiles_no)*expanded_t[0,:,0]),min=0,max=self.tiles_no-1).long()
        return expanded_t,model_idx

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



class AdaptiveMultiGabor2DLayer_complete_real(nn.Module):
    '''

        Implements WIRE2D activations with multiple channel input.

    '''
    
    def __init__(self, in_features, out_features, n_channels,
                 is_first=False, share_weights=False,
                 omega_0=30, scale_0=5.0,n_img=None,rank=None,config=None):
        super().__init__()        
        # Divide features by 2 since we have two linear layers
        self.in_features = in_features
        self.omega_0 = omega_0
        self.scale_0 = scale_0
        self.linear = AdaptiveLinearWithChannel(in_features,
                                                out_features,
                                                n_channels,
                                                is_first,
                                                share_weights,data_type=torch.float,
                                                n_img=n_img,rank=rank,config=config)
        
        self.orth_scale = AdaptiveLinearWithChannel(in_features,
                                                out_features,
                                                n_channels,
                                                is_first,
                                                share_weights, data_type=torch.float
                                                ,n_img=n_img,rank=rank,config=config)
    def forward(self, input, indices,t=None):

        lin = self.linear(input, indices,t=t)
        lin2 = self.orth_scale(input, indices,t=t)
            
        omega = self.omega_0 * lin
        scale = self.scale_0 * lin
        orth_scale = self.scale_0 * lin2
        
        arg = torch.cos(omega)*torch.exp(-scale.square()-orth_scale.square())
        
        return arg





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
                 omega_0=30, scale_0=5.0,n_img=None,rank=None,config=None):
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
        

        # Higher order interactions :
        #if self.in_features != 2 and self.out_features!=3:
        #    lin  = lin*input  + input
        #    lin2 = lin2*input + input

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
                 scale=10.0, const=1.0, mode='2d',n_img=None,pos_encode=False,rank=None,config=None):
        super().__init__()
        
        self.mode = mode
        self.rank = rank
        if self.mode == '1d':
            self.nonlin = AdaptiveMultiGaborLayer
        elif self.mode == '2d':
            self.nonlin = AdaptiveMultiGabor2DLayer
            hidden_features = int(hidden_features/np.sqrt(2))
        elif self.mode == '2db':
            self.nonlin = AdaptiveMultiGabor2DLayer_real
        elif self.mode=="complete_real":
                self.nonlin = AdaptiveMultiGabor2DLayer_complete_real

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
        self.fold = torch.nn.Fold(output_size=(480, 480),
                kernel_size=[480,480],
                stride=[480,480])
        
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

        # output_ls = []

        for i_mod, mod in enumerate(self.net):
            if ex_biases is not None and i_mod==0:
                output = mod(output, indices, ex_biases)
            else:
                output = mod(output, indices, t,coords=coords,epochs=epochs)
                # output_ls.append(output)
      

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
        

def get_freq_mask(embedded_shape, current_iter, total_iter,rank):
    # len(embedded_shape) == 2:
    #    embedded_shape = embedded_shape[1]
    if current_iter > total_iter:
        return torch.ones(embedded_shape).cuda(rank).float()
    else:
        alpha_t = np.zeros(embedded_shape)
        ptr = embedded_shape * current_iter / total_iter + 1 
        ptr = ptr if ptr < embedded_shape else embedded_shape 
        int_ptr = int(ptr)
        alpha_t[: int_ptr * 1] = 1.0
        alpha_t[int_ptr * 1: int_ptr * 1 + 1] = (ptr - int_ptr)
        return torch.from_numpy(np.clip(np.array(alpha_t), 1e-8, 1 - 1e-8)).cuda(rank).float()