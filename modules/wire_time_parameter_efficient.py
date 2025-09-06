#!/usr/bin/env python

import os
import sys
import tqdm
import pdb
import math
import numpy as np
import torch
from torch import nn
from einops import rearrange
from siren_hyperinr import AdaptiveMultiSiren
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
                 bias=True, data_type=None, add_dim=False,n_img=None,rank=None):
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
        self.w_nchan = w_nchan
        self.pos_encoding = False
        if self.pos_encoding:
            self.t_enc =  PositionalEncoding('1.05_36',n_chunks=None)
            self.t_enc_b =  PositionalEncoding('1.05_9',n_chunks=None)
        hyper_network = AdaptiveMultiSiren
        self.complex_layer = (dtype == torch.cfloat)
        self.nearest = True
        #self.transformer = True
        #initialize weights
        self.final = False
        
        self.weight_grid = nn.ParameterList().to(rank)
        self.bias_grid = nn.ParameterList().to(rank)
        if add_dim:
            self.weight = torch.nn.Parameter(torch.zeros(
                            w_nchan,
                            1,
                            input_size,
                            output_size,
                            dtype=dtype))
        else:
            if dtype == torch.cfloat:
                feat_size= 0
                for no,t in enumerate([20,10,5]):
                    
                    #scale = int(pow(2,no))
                    #feat_size += 24*scale
                    weight = (0.1*torch.rand(t,12,24,24, dtype=torch.cfloat)-0.1).to(rank)
                    self.weight_grid.append(nn.Parameter(torch.view_as_real(weight)))

                feat_size = int(feat_size)
                stem_additive = torch.rand(1,input_size*output_size,dtype=torch.cfloat).to(rank)
                stem_additive = stem_additive.uniform_(-1/math.sqrt(72),1/math.sqrt(72))
                self.stem_additive = nn.Parameter(torch.view_as_real(stem_additive))
                #stem_additive2= torch.rand(1,input_size*output_size,dtype=torch.cfloat).to(rank)
                #stem_additive2= stem_additive2.uniform_(-1/math.sqrt(input_size*output_size),1/math.sqrt(input_size*output_size))
                #self.stem_additive2 = nn.Parameter(torch.view_as_real(stem_additive2))
                stem_weight = torch.rand(72,input_size*output_size,dtype=torch.cfloat).to(rank)
                #q,r = torch.linalg.qr(stem_weight.permute(1,0) if input_size*output_size>72 else stem_weight)
                #q = (q*2/(input_size*output_size)).permute(1,0) if input_size*output_size>72 else stem_weight
                stem_weight = torch.nn.init.kaiming_uniform_(stem_weight,nonlinearity="relu",mode="fan_out")
                self.stem = nn.Parameter(torch.view_as_real(stem_weight))
                #self.transformer_r= TransformerBlock(dim=72,heads=4,dim_head=18,mlp_dim=36,dropout=0,prenorm=False)
                #self.transformer_i= TransformerBlock(dim=72,heads=4,dim_head=18,mlp_dim=36,dropout=0,prenorm=False)
                #stem_weight2 = torch.rand(input_size*output_size,input_size*output_size,dtype=torch.cfloat).to(rank)
                #stem_weight2 = torch.nn.init.kaiming_uniform_(stem_weight2,nonlinearity="relu",mode="fan_out")
                #self.stem2 = nn.Parameter(torch.view_as_real(stem_weight2))
                #stem_weight3 = torch.rand(input_size*output_size,input_size*output_size,dtype=torch.cfloat).to(rank)
                #stem_weight3 = torch.nn.init.kaiming_uniform_(stem_weight3,nonlinearity="relu",mode="fan_out")
                #self.stem3 = nn.Parameter(torch.view_as_real(stem_weight3))
                self.complex_relu = ModReLU(True).to(rank)
            else:
                feat_size= 0
                for no,t in enumerate([20,10,5]):
                    #scale = int(pow(2,no))
                    #feat_size += 24*scale
                    weight = (0.1*torch.rand(t,12,24,24, dtype=torch.float)-0.1).to(rank)
                    self.weight_grid.append(nn.Parameter(weight))
                #self.transformer= TransformerBlock(dim=72,heads=4,dim_head=18,mlp_dim=36,dropout=0,prenorm=False)
                feat_size = int(feat_size)
                stem_additive = torch.rand(1,input_size*output_size,dtype=torch.float).to(rank)
                stem_additive = stem_additive.uniform_(-1/math.sqrt(72),1/math.sqrt(72))
                self.stem_additive = nn.Parameter(stem_additive)
                #stem_additive2 = torch.rand(1,input_size*output_size,dtype=torch.float).to(rank)
                #stem_additive2 = stem_additive2.uniform_(-1/math.sqrt(input_size*output_size),1/math.sqrt(input_size*output_size))
                #self.stem_additive2 = nn.Parameter(stem_additive2)
                stem_weight = torch.rand(72,input_size*output_size,dtype=torch.float).to(rank)
                #q,r = torch.linalg.qr(stem_weight.permute(1,0) if input_size*output_size>72 else stem_weight)
                #q = (q*2/(input_size*output_size)).permute(1,0) if input_size*output_size>72 else q
                stem_weight = torch.nn.init.kaiming_uniform_(stem_weight,nonlinearity="relu",mode="fan_out")
                self.stem = nn.Parameter(stem_weight)
                #stem_weight2 = torch.rand(input_size*output_size,input_size*output_size,dtype=torch.float).to(rank)
                #stem_weight2 = torch.nn.init.kaiming_uniform_(stem_weight2,nonlinearity="relu",mode="fan_out")
                #self.stem2 = nn.Parameter(stem_weight2)
                #stem_weight3 = torch.rand(input_size*output_size,input_size*output_size,dtype=torch.float).to(rank)
                #stem_weight3 = torch.nn.init.kaiming_uniform_(stem_weight3,nonlinearity="relu",mode="fan_out")
                #self.stem3 = nn.Parameter(stem_weight3)

        #self.stem_act = nn.ReLU()
        if bias:
            if add_dim:
                self.bias = torch.nn.Parameter(torch.zeros(channel_size,
                                                   1,
                                                   output_size,
                                                   1,
                                                   dtype=dtype))
            else:
                if dtype == torch.cfloat:
                    for no,t in enumerate([20,10,5]):
                        #scale = int(pow(2,no))
                        bias = (0.1*torch.rand(t,12,24,6, dtype=torch.cfloat)-0.1).to(rank)
                        self.bias_grid.append(nn.Parameter(torch.view_as_real(bias)))
                    stem_bias = torch.rand(6*3,output_size,dtype=torch.cfloat).to(rank)
                    #q,r = torch.linalg.qr(stem_bias.permute(1,0) if output_size>18 else stem_bias)
                    #q = (q*2/(input_size*output_size)).permute(1,0)  if output_size>18 else q
                    stem_bias = torch.nn.init.kaiming_uniform_(stem_bias,nonlinearity="relu",mode="fan_out")
                    self.stem_bias = nn.Parameter(torch.view_as_real(stem_bias))
                    self.complex_relu_b = ModReLU(True).to(rank)
                    #stem_bias2 = torch.rand(output_size,output_size,dtype=torch.cfloat).to(rank)
                    #stem_bias2 = torch.nn.init.kaiming_uniform_(stem_bias2,nonlinearity="relu",mode="fan_out")
                    #self.stem_bias2 = nn.Parameter(torch.view_as_real(stem_bias2))
                    #stem_bias3 = torch.rand(output_size,output_size,dtype=torch.cfloat).to(rank)
                    #stem_bias3 = torch.nn.init.kaiming_uniform_(stem_bias3,nonlinearity="relu",mode="fan_out")
                    #self.stem_bias3 = nn.Parameter(torch.view_as_real(stem_bias3))
                    stem_additive_bias = torch.rand(1,output_size,dtype=torch.cfloat).to(rank)
                    stem_additive_bias = stem_additive_bias.uniform_(-1/math.sqrt(6*3),1/math.sqrt(6*3))
                    self.stem_additive_bias = nn.Parameter(torch.view_as_real(stem_additive_bias))
                    #stem_additive_bias2 = torch.rand(1,output_size,dtype=torch.cfloat).to(rank)
                    #stem_additive_bias2 = stem_additive_bias2.uniform_(-1/math.sqrt(output_size),1/math.sqrt(output_size))
                    #self.stem_additive_bias2 = nn.Parameter(torch.view_as_real(stem_additive_bias2))
                else:
                    for no,t in enumerate([20,10,5]):
                        #scale = int(pow(2,no))
                        bias = (0.1*torch.rand(t,12,24,6, dtype=torch.float)-0.1).to(rank)
                        self.bias_grid.append(nn.Parameter(bias))
                    stem_bias = torch.rand(6*3,output_size,dtype=torch.float).to(rank)
                    #q,r = torch.linalg.qr(stem_bias.permute(1,0) if output_size>18 else stem_bias)
                    #q = (q*2/(input_size*output_size)).permute(1,0) if output_size>18 else q
                    stem_bias = torch.nn.init.kaiming_uniform_(stem_bias,nonlinearity="relu",mode="fan_out")
                    self.stem_bias = nn.Parameter(stem_bias)
                    #stem_bias2 = torch.rand(output_size,output_size,dtype=torch.float).to(rank)
                    #stem_bias2 = torch.nn.init.kaiming_uniform_(stem_bias2,nonlinearity="relu",mode="fan_out")
                    #self.stem_bias2 = nn.Parameter(stem_bias2)
                    #stem_bias3 = torch.rand(output_size,output_size,dtype=torch.float).to(rank)
                    #stem_bias3 = torch.nn.init.kaiming_uniform_(stem_bias3,nonlinearity="relu",mode="fan_out")
                    #self.stem_bias3 = nn.Parameter(stem_bias3)
                    stem_additive_bias = torch.rand(1,output_size,dtype=torch.float).to(rank)
                    stem_additive_bias = stem_additive_bias.uniform_(-1/math.sqrt(6*3),1/math.sqrt(6*3))
                    self.stem_additive_bias = nn.Parameter(stem_additive_bias)
                    #stem_additive_bias2 = torch.rand(1,output_size,dtype=torch.float).to(rank)
                    #stem_additive_bias2 = stem_additive_bias2.uniform_(-1/math.sqrt(output_size),1/math.sqrt(output_size))
                    #self.stem_additive_bias2 = nn.Parameter(stem_additive_bias2)

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

            #self.hyper_shift = nn.Sequential(nn.Linear())
            
            self.hyper_shift = hyper_network(in_features=1,
                        out_features=output_size*2,
                        n_channels=w_nchan,
                        hidden_features=10,
                        hidden_layers=1,
                        nonlin='relu',
                        outermost_linear=True,
                        share_weights=False,
                        first_omega_0=5,
                        hidden_omega_0=5,
                        n_img=n_img,
                        hyper=True,pos_encode=False,gating=input_size
                        )
                
        else:
            
            #self.hyper_dist = torch.nn.LSTM(input_size=1,hidden_size=32,num_layers=1,batch_first=True)
            #self.convertor = nn.Linear(in_features=32,out_features=output_size)
            
            self.hyper_dist = hyper_network(in_features=1,
                        out_features=output_size,
                        n_channels=w_nchan,
                        hidden_features=10,
                        hidden_layers=1,
                        nonlin='relu',
                        outermost_linear=True,
                        share_weights=False,
                        first_omega_0=5,
                        hidden_omega_0=5,
                        n_img=n_img,
                        hyper=True,pos_encode=False,gating=input_size
                        )
           

    
        #change weights to kaiming
        #self.reset_parameters()
        self.const = const
        
    @torch.no_grad()
    def reset_parameters(self):
        #_, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(weights)
        #bound = np.sqrt(const/self.output_size)
        #self.weight.uniform_(-0.1, 0.1)
        if self.complex_layer:
            #weight = (0.1*torch.rand(5,self.w_nchan,self.input_size,self.output_size, dtype=torch.cfloat)-0.1)
            #weight = weight.view(5*self.w_nchan,-1)
            #Q_, R_ = torch.linalg.qr(weight)
            #self.weight_r.data = R_
            #  var (0.2 ^ 12)
            #self.weight_q.data = Q_
            if not self.final:
                self.weight_r.uniform_(-0.1,0.1)
                bound = np.sqrt(12/(self.w_nchan))/(2)
                self.weight_q.uniform_(-bound,bound)
            else:
                self.weight.uniform_(-0.1,0.1)
            
        else:
            self.weight.uniform_(-0.1,0.1)
        self.bias.uniform_(-0.1, 0.1)

        #bound = np.sqrt(const/self.output_size)
        
        #if self.add_bias:
        #    self.bias.uniform_(-bound, bound)
    
    @torch.no_grad()
    def reset_bias(self):
        _, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
        
        bound = np.sqrt(self.const/fan_out)
        self.bias.uniform_(-bound, bound)
        
    def forward(self, x, indices,t=None, ex_biases=None):        

        t, model_idx = t
        if self.pos_encoding:
            pos_encoding = 0.1*self.t_enc(t[0,:,0])
            pos_encoding_b = 0.1*self.t_enc_b(t[0,:,0])
        out_list=[]

        if self.complex_layer:
            for weight_g in self.weight_grid:
                weight_g = torch.view_as_complex(weight_g)
                size = weight_g.size(0)
                t_scaled = (t[0,:,0])*(size)
                if not self.nearest:
                    left = torch.floor(t_scaled+1e-6).long()
                    right = torch.min(left+1,torch.tensor(weight_g.size(0)-1))
                    d_left = (t_scaled - left).view(-1,1,1,1)
                    d_right = (right - t_scaled).view(-1,1,1,1)
                    out_list.append(d_right*weight_g[left]+d_left*weight_g[right] - ((right-left-1).view(-1,1,1,1))*weight_g[left])
                else:
                    nearest = torch.min(torch.round(t_scaled).long(),torch.tensor(weight_g.size(0)-1))
                    out_list.append(weight_g[nearest])
            stem_w = torch.view_as_complex(self.stem)
            stem_additive = torch.view_as_complex(self.stem_additive)
            
        else:
            for weight_g in self.weight_grid:
                size = weight_g.size(0)
                t_scaled = (t[0,:,0])*(size)
                if not self.nearest:
                    left = torch.floor(t_scaled+1e-6).long()
                    right = torch.min(left+1,torch.tensor(weight_g.size(0)-1))
                    d_left = (t_scaled - left).view(-1,1,1,1)
                    d_right = (right - t_scaled).view(-1,1,1,1)
                    out_list.append(d_right*weight_g[left]+d_left*weight_g[right] - ((right-left-1).view(-1,1,1,1))*weight_g[left])
                else:
                    nearest = torch.min(torch.round(t_scaled).long(),torch.tensor(weight_g.size(0)-1))
                    out_list.append(weight_g[nearest])

            stem_w = self.stem
            stem_additive = self.stem_additive

        stem_additive = stem_additive[None,...]

        output = out_list[0]
        for i in range(len(out_list)-1):
            output = torch.cat([output, out_list[i+1]],dim=-1)
        output = output.flatten(1,2)
        if self.pos_encoding:
            output = output + pos_encoding[:,None,:]
        #if self.complex_layer:
        #    output = self.transformer_r(output.real).type(torch.cfloat) + 1j*self.transformer_i(output.imag).type(torch.cfloat)
        #else:
        #    output = self.transformer(output)
        weight = torch.matmul(output ,stem_w[None,:]) + stem_additive
        if self.complex_layer:
            #weight2 = torch.matmul(weight,stem_w_2) + stem_additive
            weight = self.complex_relu(weight)
            #weight = weight + weight2
            #weight3 = torch.matmul(weight,stem_w_3) + stem_additive2
            #weight3 = F.gelu(weight3.real).type(torch.cfloat) + 1j*F.gelu(weight3.imag).type(torch.cfloat)
            #weight = weight + weight3
        else:
            weight =F.relu(weight)
            #weight = weight + F.gelu(torch.matmul(weight,stem_w_3) + stem_additive2)
        weight = weight.view(-1,self.w_nchan,self.input_size,self.output_size)
        out_list = []
        #weight = weight[model_idx,...]
        w_mul_x = torch.matmul(x, weight)
        #bias = self.bias[model_idx,...]

        if self.complex_layer:
            for bias_g in self.bias_grid:   
                bias_g = torch.view_as_complex(bias_g)
                size = bias_g.size(0)
                t_scaled = (t[0,:,0])*(size)
                if not self.nearest: 
                    left = torch.floor(t_scaled+1e-6).long()
                    right = torch.min(left+1,torch.tensor(bias_g.size(0)-1))
                    d_left = (t_scaled - left).view(-1,1,1,1)
                    out_list.append(d_right*bias_g[left]+d_left*bias_g[right] - ((right-left-1).view(-1,1,1,1))*bias_g[left]) 
                else:
                    nearest = torch.min(torch.round(t_scaled).long(),torch.tensor(bias_g.size(0)-1))
                    out_list.append(bias_g[nearest])
            stem_b = torch.view_as_complex(self.stem_bias)
            #stem_b2 = torch.view_as_complex(self.stem_bias2)
            #stem_b3 = torch.view_as_complex(self.stem_bias3)
            stem_additive_bias = torch.view_as_complex(self.stem_additive_bias)
            #stem_additive_bias2 = torch.view_as_complex(self.stem_additive_bias2)

        else:
            for bias_g in self.bias_grid:
                size = bias_g.size(0)
                t_scaled = (t[0,:,0])*(size)
                if not self.nearest:
                    left = torch.floor(t_scaled+1e-6).long()
                    right = torch.min(left+1,torch.tensor(bias_g.size(0)-1))
                    d_left = (t_scaled - left).view(-1,1,1,1)
                    d_right = (right - t_scaled).view(-1,1,1,1)
                    out_list.append(d_right*bias_g[left]+d_left*bias_g[right] - ((right-left-1).view(-1,1,1,1))*bias_g[left])
                else:
                    nearest = torch.min(torch.round(t_scaled).long(),torch.tensor(bias_g.size(0)-1))
                    out_list.append(bias_g[nearest])

            stem_additive_bias = self.stem_additive_bias
            #stem_additive_bias2 = self.stem_additive_bias2
            stem_b = self.stem_bias
            #stem_b2 = self.stem_bias2
            #stem_b3 = self.stem_bias3
        
        stem_additive_bias = stem_additive_bias[None,...]
        #stem_additive_bias2 = stem_additive_bias2[None,...]
        output_ = out_list[0]
        for i in range(len(out_list)-1):
            output_ = torch.cat([output_, out_list[i+1]],dim=-1)
        output_ = output_.flatten(1,2)
        if self.pos_encoding:
            output_ = output_ + pos_encoding_b[:,None,:]
        bias = torch.matmul(output_,stem_b) + stem_additive_bias
        if self.complex_layer:
            #bias2 = torch.matmul(bias,stem_b2) + stem_additive_bias
            bias = self.complex_relu_b(bias)
            #bias = bias + bias2
            #bias3 = torch.matmul(bias,stem_b3) + stem_additive_bias2
            #bias3 = F.gelu(bias3.real).type(torch.cfloat) + 1j*F.gelu(bias3.imag).type(torch.cfloat)
            #bias = bias + bias3
        else:
            bias = F.relu(bias)
            #xbias = bias + F.gelu(torch.matmul(bias,stem_b3)+  stem_additive_bias2)
            
        bias = bias.view(-1,self.w_nchan,1,self.output_size)
        out_list = []  
        if self.complex_layer:
            hyper_shift = self.hyper_shift(t,indices).permute(1, 0, 2).unsqueeze(-2)
            size = hyper_shift.shape[-1]
            hyper_shift_real = hyper_shift[:,:,:,:(size//2)]
            hyper_shift_imag = hyper_shift[:,:,:,(size//2):]
            hyper_shift = torch.complex(hyper_shift_real,hyper_shift_imag)

        else:
           hyper_shift = self.hyper_dist(t,indices).permute(1, 0, 2).unsqueeze(-2)   

        return  w_mul_x + bias + hyper_shift 
    

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
                 omega_0=30, scale_0=5.0,n_img=None,rank=None):
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
                                share_weights,n_img=n_img,rank=rank)
        
        self.orth_scale = linlayer(in_features,
                                    out_features,
                                    n_channels,
                                    is_first,
                                    share_weights,n_img=n_img,rank=rank)
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
                 scale=10.0, const=1.0, mode='2d',n_img=None,pos_encode=False,rank=None):
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
            self.pos_encode = Trued
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
                                scale_0=scale,n_img=n_img,rank=rank))

        if outermost_linear:
            feat = hidden_features[-1]
            final_linear = AdaptiveLinearWithChannel(
                                    feat, 
                                    out_features,
                                    n_channels,
                                    is_first=False,
                                    share_weights=share_weights,
                                    const=const,
                                    bias=True,n_img=n_img,rank=rank)
                    
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
            
            #output_ls.append(output)
        
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
        #pos = pos[0,:,0]
        if self.pe_embed == 'none':
            return pos[:,None]
        else:
            pe_list = []
            for i in range(self.levels):
                temp_value = pos * self.lbase **(i) * math.pi
                pe_list += [torch.sin(temp_value), torch.cos(temp_value)]
            return torch.stack(pe_list, 1)

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
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.,dtype=torch.cfloat):
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
    

class ModReLU(nn.Module):
    def __init__(self, relu_squared = False):
        super().__init__()
        self.pow = 2 if relu_squared else 1
        self.bias = nn.Parameter(torch.tensor(0.))

    def forward(self, x):
        real = F.relu(torch.abs(x) + self.bias) ** self.pow
        imag = torch.exp(1.j * torch.angle(x))
        return real + imag