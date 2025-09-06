import torch
import torch.nn as nn
import numpy as np
from bitEstimator import BitEstimator

class custom_linear(nn.Module):

    
    def __init__(self,w_nchan,input_size,output_size,complex,rank,is_first,args):
        super().__init__()
        self.no_shift= args.no_shift
        self.h,self.w = args.h//args.ksize[0],args.w//args.ksize[1]
        (self.g_h,self.g_w) = args.entropy_group
        if complex:
            weight = (0.1*torch.rand(1,w_nchan,input_size,output_size, dtype=torch.cfloat)-0.1).to(rank)
            bias = (0.1*torch.rand(args.n_frames,w_nchan,1,output_size, dtype=torch.cfloat)-0.1).to(rank) 
            self.bias = nn.Parameter(torch.view_as_real(bias))
            self.weight = nn.Parameter(torch.view_as_real(weight))
            self.scale =nn.Parameter(torch.empty((1, w_nchan,1,1,2),dtype=torch.float),requires_grad=True).to(rank)
            self.shift = nn.Parameter(torch.empty((1,w_nchan,1,1,2), dtype=torch.float)).to(rank) if not args.no_shift else 0
            self.register_buffer("div",torch.ones((1,w_nchan,1,1,2),dtype=torch.float).to(rank))
        else:
            weight = (0.1*torch.rand(args.no_tiles,w_nchan,input_size,output_size, dtype=torch.float)-0.1).to(rank) 
            bias = (0.1*torch.rand(args.n_frames,w_nchan,1,output_size, dtype=torch.float)-0.1).to(rank)
            self.weight = nn.Parameter(weight)
            self.bias = nn.Parameter(bias)
            self.scale =nn.Parameter(torch.empty((1, w_nchan,1,1),dtype=torch.float,requires_grad=True).to(rank))
            self.shift = nn.Parameter(torch.empty((1,w_nchan,1,1), dtype=torch.float).to(rank)) if not args.no_shift else 0
            self.register_buffer("div",torch.ones((1,w_nchan,1,1),dtype=torch.float).to(rank))

        
        self.cdf = nn.ModuleList([BitEstimator(1,rank,False,num_layers=1).to(rank) for _ in range(self.g_h*self.g_w)]).to(rank)

        if is_first:
            self.boundary= args.boundary_first
        else: 
            self.boundary= args.boundary

        self.complex = complex
        

        self.reset_parameters()


    @torch.no_grad()
    def reset_parameters(self):
 
        self.weight.uniform_(-self.boundary,self.boundary)
        if not self.no_shift:
            torch.nn.init.zeros_(self.shift)
        for i in range(self.weight.shape[0]):
            if self.complex:
                fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(torch.view_as_complex(self.weight[0,...]))
                bound = np.sqrt(3/(fan_out))
                torch.nn.init.constant_(self.scale,bound)
            else:
                fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(self.weight[0,...])
                bound = np.sqrt(3/(fan_out))
                torch.nn.init.constant_(self.scale,bound)
                

        for i in range(self.bias.shape[0]):
            if self.complex:
                fan_in , fan_out = torch.nn.init._calculate_fan_in_and_fan_out(torch.view_as_complex(self.weight[0,...]))
                bound = np.sqrt(1/(fan_out))
                self.bias[i,...].uniform_(-bound,bound)
            else:
                fan_in , fan_out = torch.nn.init._calculate_fan_in_and_fan_out(self.weight[0,...])
                bound = np.sqrt(1/(fan_out))
                self.bias[i,...].uniform_(-bound,bound)



    def forward(self,x,m_idx,fr_idx,t):

        if self.complex:
            bias = self.bias[fr_idx,...] 
            weight= self.weight[m_idx,...]
            weight = torch.mul((weight / self.div + self.shift),self.scale)
            weight = torch.view_as_complex(weight)
            bias = torch.view_as_complex(bias)
            #weight_abs = torch.abs(weight_s)
            #w_phase = torch.angle(weight_s)
            #weight_abs = (weight_abs.round() - weight_abs).detach() + weight_abs
            #w_phase = (w_phase.round() - w_phase).detach() + w_phase
            #w_real, w_imag = weight_abs*torch.cos(w_phase), weight_abs*torch.sin(w_phase)
            #weight = torch.stack([w_real,w_imag],dim=-1)
            #weight = torch.mul((weight / self.div + self.shift),self.scale)
            #weight = torch.view_as_complex(weight)
        else:   
            weight = self.weight 
            bias = self.bias 
            weight, bias = weight[m_idx,...], bias[fr_idx,...]
            #weight = (weight.round() - weight).detach() + weight
            weight = torch.mul((weight / self.div + self.shift),self.scale)
        
        return torch.matmul(x,weight) + bias

    def get_weight_group(self):
        g_h, g_w, h, w = self.g_h, self.g_w, self.h, self.w
        weight = self.weight
        weight_ = weight.flatten(start_dim=2)
        weight_ = weight_.view(g_h, h//g_h, g_w, w//g_w, -1)
        weight_ = weight_.permute(0,2,1,3,4)
        weight_ = weight_.reshape(g_h*g_w,-1)
        return weight_