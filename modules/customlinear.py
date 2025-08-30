import torch
import torch.nn as nn
import numpy as np
import rff
import math
from torch.nn import functional as F
from scipy.stats import boxcox
from scipy.special import inv_boxcox

def grad_scale(x, scale):
    y = x
    y_grad = x * scale
    return y.detach() - y_grad.detach() + y_grad

def myabs(x):
    return torch.where(x==0, x, torch.abs(x))

def mysign(x):
    return torch.where(x == 0, torch.ones_like(x), torch.sign(x))



def soft_round(x, temperature):
  if not temperature==0:
    temperature = torch.tensor([temperature],device=x.device,dtype=torch.float)
    m = torch.floor(x) + 0.5
    z = 2 * torch.tanh(0.5 / temperature)
    r = torch.tanh((x - m) / temperature) / z
    return m + r
  else:
    return torch.round(x)

def soft_round_inverse(x, temperature):

  m = torch.floor(x) + 0.5
  z = 2 * torch.tanh(0.5 / temperature)
  r = torch.arctanh((x - m) * z) * temperature
  return m + r

def soft_round_conditional_mean(x, temperature):

  return soft_round_inverse(x - 0.5, temperature) + 0.5



def init_ssf_scale_shift(dim,complex,rank):
    if complex:
        scale_abs = torch.normal(mean=1,std=0.02,size=(1,dim,1,1),dtype=torch.float,requires_grad=True).to(rank)
        scale_ang = torch.zeros_like(scale_abs)
        scale_real, scale_imag = scale_abs*torch.cos(scale_ang), scale_abs*torch.sin(scale_ang)
        scale = nn.Parameter(torch.stack([scale_real,scale_imag],dim=-1)).to(rank)
        shift_abs = torch.normal(mean=0,std=0.02,size=(1,dim,1,1),dtype=torch.float,requires_grad=True).to(rank)
        shift_angle = torch.empty_like(shift_abs).uniform_(-math.pi,math.pi)
        shift_real,shift_imag = shift_abs*torch.cos(shift_angle),shift_abs*torch.sin(shift_angle)
        shift = nn.Parameter(torch.stack([shift_real,shift_imag],dim=-1)).to(rank)
    else:
        scale = nn.Parameter(torch.ones((1,dim,1,1))).to(rank)
        shift = nn.Parameter(torch.zeros((1,dim,1,1))).to(rank)
        nn.init.normal_(scale, mean=1, std=.02)
        nn.init.normal_(shift, std=.02)

    nn.init.normal_(scale, mean=1, std=.02)
    nn.init.normal_(shift, std=.02)

    return scale, shift


class StraightThroughFloor(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        return torch.floor(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output
    

class ewgs(torch.autograd.Function):
    """
    x_in: continuous inputs within the range of [0,1]
    num_levels: number of discrete levels
    scaling_factor: backward scaling factor
    x_out: discretized version of x_in within the range of [0,1]
    """
    @staticmethod
    def forward(ctx, x,scale):
        code = x/scale
        quant = torch.round(code)
        dequant = scale*code
        error = dequant - x
        scale_der = quant - code
        ctx.save_for_backward(scale_der,error)
        ctx._scaling_factor = 0.001
        return code,quant,dequant
    
    @staticmethod
    def backward(ctx, code_g,quant_g,dequant_g):
        #print("Backward called")
        #breakpoint()
        scale_der,error = ctx.saved_tensors
        delta = ctx._scaling_factor
        scale = 1 + delta * torch.sign(dequant_g)*error
        return dequant_g * scale, scale_der

class ewgs2(torch.autograd.Function):
    """
    x_in: continuous inputs within the range of [0,1]
    num_levels: number of discrete levels
    scaling_factor: backward scaling factor
    x_out: discretized version of x_in within the range of [0,1]
    """
    @staticmethod
    def forward(ctx, code):
        
        quant = torch.round(code)
        
        ctx._scaling_factor = 0.1
        ctx.save_for_backward(code-quant) # save quantization error which modulates gradient
        return quant
    
    @staticmethod
    def backward(ctx, g):
        diff = ctx.saved_tensors[0]
        delta = ctx._scaling_factor
        scale = 1 + delta * torch.sign(g)*diff # if error direction is not aligned with gradient, downscale it (TRUST ISSUE)
        return g * scale



epsilon=1e-5
class custom_linear(nn.Module):
    ''' 
    group_size = (4 patch,4 patch) [share same scale, and entropy model]
    input_size and output size of linear layer
    args.quant dictate whether quantization is active
    args.quant_model_bit : number of bits used to quantize weights
    args.quant_bias_bit : number of bits used to quantize biases
    complex dictates whether layer is complex
    Weight Shape = [1,n_v,n_h,25,25]
    n_v and n_h stands for number of blocks in vertical and horizontal direction
    We decompose weight into g_h times g_v block size
    Decomposed weight = [1,(n_v//g_v),g_v,(n_h//g_h),g_h,25,25]
    We should have [n_v // g_v,n_h//g_h] unique scaler
    '''
    
    def __init__(self,group_size,w_nchan,input_size,output_size,complex,is_first,rank,args):
        super().__init__()
        n_h, n_v= args.h // args.ksize[1],args.w // args.ksize[0]
        self.no_win, self.out_size = w_nchan,output_size
        ent_group = args.entropy_group
        grid_size = (n_v,n_h)
        self.sga = args.sga
        self.soft_round = args.soft_round
        if complex:
            weight = (0.1*torch.rand(args.no_tiles,w_nchan,input_size,output_size, dtype=torch.cfloat)-0.1).to(rank)
            bias = (0.1*torch.rand(args.n_frames,w_nchan,1,output_size, dtype=torch.cfloat)-0.1).to(rank) 
            self.bias = nn.Parameter(torch.view_as_real(bias))
            self.weight = nn.Parameter(torch.view_as_real(weight))
        else:
            weight = (0.1*torch.rand(args.no_tiles,w_nchan,input_size,output_size, dtype=torch.float)-0.1).to(rank) 
            bias = (0.1*torch.rand(args.n_frames,w_nchan,1,output_size, dtype=torch.float)-0.1).to(rank)
            self.weight = nn.Parameter(weight)
            self.bias = nn.Parameter(bias)

        if args.quant:
            bits_w, bits_b = args.quant_model_bit, args.quant_bias_bit
            per_channel_w, per_channel_b = args.per_channel_w, args.per_channel_b
            if complex:
                if args.abs_log:
                    self.weight_mag_quantizer = Log_T(bits_w, group_size,ent_group, grid_size, signed=False, per_channel=per_channel_w,
                                                      rank=rank,args=args)
                else:
                    self.weight_mag_quantizer = Scale_T(bits_w, group_size,ent_group, grid_size, signed=False, per_channel=per_channel_w,args=args)

                self.weight_angle_quantizer = Scale_T(bits_w, group_size,ent_group, grid_size, signed=True, per_channel=per_channel_w,args=args)
                if args.abs_log:
                    self.bias_mag_quantizer = Log_T(bits_b,group_size, ent_group, grid_size,signed=False, per_channel=per_channel_b,
                                                    rank=rank,args=args)
                else:
                    self.bias_mag_quantizer = Scale_T(bits_b,group_size, ent_group, grid_size,signed=True, per_channel=per_channel_b,args=args)
                
                self.bias_angle_quantizer = Scale_T(bits_b,group_size,ent_group, grid_size,signed=True, per_channel=per_channel_b,args=args)
                self.bitrate_w_mag_dict = {}
                self.bitrate_b_mag_dict = {}
                self.bitrate_w_angle_dict = {}
                self.bitrate_b_angle_dict = {}
            else:
                self.weight_quantizer = Scale_T(bits_w, group_size, ent_group, grid_size, signed=True, per_channel=per_channel_w,args=args)
                self.bias_quantizer = Scale_T(bits_w, group_size, ent_group, grid_size, signed=True, per_channel=per_channel_w,args=args)
                self.bitrate_w_dict = {}
                self.bitrate_b_dict = {}
        self.complex = complex
        if self.complex:
            self.dequant_w_mag = None
            self.dequant_w_angle = None
            self.dequant_b_mag = None
            self.dequant_b_angle = None   
        else:       
            self.dequant_w = None
            self.dequant_b = None  
        self.diff_encoding = args.diff_encoding


        self.quant = args.quant
        self.reset_parameters()


    @torch.no_grad()
    def reset_parameters(self):
        
        for i in range(self.weight.shape[0]):
            if self.complex:
                fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(torch.view_as_complex(self.weight[0,...]))
                bound = np.sqrt(3/(fan_out))
                self.weight[i,...].uniform_(-bound, bound)
            else:
                fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(self.weight[0,...])
                bound = np.sqrt(3/(fan_out))
                self.weight[i,...].uniform_(-bound, bound)

    
        for i in range(self.bias.shape[0]):
            if self.complex:
                fan_in , fan_out = torch.nn.init._calculate_fan_in_and_fan_out(torch.view_as_complex(self.weight[0,...]))
                bound = np.sqrt(1/(fan_out))
                self.bias[i,:,:,:].uniform_(-bound,bound)
            else:
                fan_in , fan_out = torch.nn.init._calculate_fan_in_and_fan_out(self.weight[0,...])
                bound = np.sqrt(1/(fan_out))
                self.bias[i,:,:,:].uniform_(-bound,bound)
        #for i in range(self.bias.shape[0]):
           #for n in range(self.bias.shape[1]):
                #self.bias[i,n,:,:].copy_(self.bias[0,0,:,:].data.clone())


    def forward(self,x,m_idx,fr_idx,t):

        if self.complex:
            if (self.dequant_w_mag is not None):
                weight = torch.polar(self.dequant_w_mag,self.dequant_w_angle)
                bias = torch.polar(self.dequant_b_mag,self.dequant_b_angle)
            else:
                weight = torch.view_as_complex(self.weight)
                bias = torch.view_as_complex(self.bias)
            weight, bias = weight[m_idx,...], bias[fr_idx,...]
            
        else:   
            weight = self.weight if self.dequant_w is None else self.dequant_w
            bias = self.bias if self.dequant_b is None else self.dequant_b
            weight, bias = weight[m_idx,...], bias[fr_idx,...]

        

        return torch.matmul(x,weight) + bias




class Scale_T(nn.Module):
    def __init__(self, bits,group_size,ent_group, grid_size,signed=False, per_channel=False,args=None):
        super().__init__()
        g_v,g_h,n_v,n_h = group_size[0],  group_size[1],  grid_size[0], grid_size[1]
        self.g_v,self.g_h,self.n_v,self.n_h= g_v, g_h, n_v, n_h
        self.g_v_e, self.g_h_e = ent_group[0], ent_group[1]
        initial = torch.ones((1,g_v,g_h,1,1,1,1),dtype=torch.float)
        self.scale = nn.Parameter(initial, requires_grad=True)
        self.init = False
        self.epoch_offset= args.compression_start_epoch
        self.p_quant = args.quant_droupout
        self.signed = signed
        self.per_channel = per_channel
        self.egws= args.egws
        self.soft_round = args.soft_round
        self.sga = args.sga
        self.temperature = self.end = args.temp
        self.start = args.temp_initial
        self.decay_ratio = args.decay_ratio
        self.cur_temp = args.temp_initial
        self.total_steps = (args.epochs - self.epoch_offset)//2


        if self.signed:
            # signed weight/activation is quantized to [-2^(b-1), 2^(b-1)-1]
            self.qmin = - 2 ** (bits - 1)
            self.qmax = 2 ** (bits - 1) - 1
        else:
            # unsigned activation is quantized to [0, 2^b-1]
            self.qmin = 0
            self.qmax = 2 ** bits - 1
        

    def init_data(self, tensor):



        n_v,n_h,g_v,g_h = self.n_v,self.n_h,self.g_v,self.g_h
        if not self.init:
            device = tensor.device
            fr, no_w, inp_size, out_size = tensor.shape
            tensor = tensor.view(fr,n_v,n_h,inp_size,out_size)
            tensor = tensor.view(fr,g_v,n_v//g_v,g_h,n_h//g_h,inp_size,out_size)
            tensor = tensor.permute(0,1,3,2,4,5,6)
            min_val = torch.amin(tensor, dim=(0, 3, 4, 5, 6), keepdim=True)
            max_val = torch.amax(tensor, dim=(0, 3, 4, 5, 6), keepdim=True)                  
            with torch.no_grad():
                scale = (max_val - min_val) / (self.qmax -self.qmin)
                self.scale.data = scale.to(device)
            self.init = True





    def encode(self, x,scale):
        
        return x/scale

    def decode(self, x,scale):
        return x*scale

    # we give just 1,n_windows,n_features,n_features,2
    # code_size = n_fr,g_v,g_h,n_v//g_v,n_h//g_h,inp,out,2 
    # or n_fr,g_v,g_h,n_v//g_v,n_h//g_h,inp,out

    def forward(self, x):
        # Reshape tensor so that form groups:
        n_fr, n_win, inp_size, out_size = x.shape
        n_v, n_h, g_v, g_h = self.n_v, self.n_h, self.g_v, self.g_h
        g_v_e, g_h_e = self.g_v_e, self.g_h_e
        x = x.view(n_fr,n_v,n_h,inp_size,out_size)
        x = x.view(n_fr,g_v,n_v//g_v,g_h,n_h//g_h,inp_size,out_size)
        x = x.permute(0,1,3,2,4,5,6)
        #
        
        if (self.sga) and (self.training):
            code = self.encode(x,self.scale)
            codef = StraightThroughFloor.apply(code) 
            codec = codef+1
            logits_f = -torch.atanh(torch.clamp(code-codef, min=-1+epsilon, max=1-epsilon)).unsqueeze(-1)/self.cur_temp
            logits_c = -torch.atanh(torch.clamp(codec-code, min=-1+epsilon, max=1-epsilon)).unsqueeze(-1)/self.cur_temp
            logits = torch.cat((logits_f,logits_c),dim=-1)
            soft_quantized = F.gumbel_softmax(logits, tau=self.cur_temp, hard=True, dim=-1)

            #dist = torch.distributions.relaxed_categorical.RelaxedOneHotCategorical(self.cur_temp, logits=logits)
            #sample = dist.rsample() 
            quant = codef*soft_quantized[...,0]+codec*soft_quantized[...,1] 
            dequant = self.decode(quant,self.scale)        
        elif self.egws:
            code = self.encode(x,self.scale)
            quant = ewgs2.apply(code)
            dequant = self.decode(quant,self.scale)
        elif self.soft_round:
            code = self.encode(x,scale)
            noise = torch.empty_like(code).uniform_(-0.5, 0.5)
            code = soft_round(code, self.cur_temp)
            code = code + noise
            code = soft_round_conditional_mean(code, self.cur_temp)            
        else:
            g = 1.0 / math.sqrt(inp_size*out_size*1e4)

            scale = grad_scale(torch.abs(self.scale), g)
            
            code = self.encode(x,scale)
            # Random quantization
            mask = torch.zeros_like(code)
            mask.bernoulli_(self.p_quant if self.training else 0)
            noise = (code.round() - code).masked_fill(mask.bool(), 0)
            quant = code + noise.detach()
            #quant = (code.round() - code).detach() + code
            dequant = self.decode(quant,scale)


        dequant = dequant.permute(0,1,3,2,4,5,6)
        dequant = dequant.view(-1,n_v,n_h,inp_size,out_size)
        dequant = dequant.view(-1,n_v*n_h,inp_size,out_size)
        code = code.permute(0,1,3,2,4,5,6) 
        quant = quant.permute(0,1,3,2,4,5,6) 
        code = code.view(n_fr,g_v_e,n_v//g_v_e,g_h_e,n_h//g_h_e,inp_size,out_size)
        quant = quant.view(n_fr,g_v_e,n_v//g_v_e,g_h_e,n_h//g_h_e,inp_size,out_size)
        code = code.permute(0,1,3,2,4,5,6)
        quant = quant.permute(0,1,3,2,4,5,6)
        code = code.permute(1,2,0,3,4,5,6)
        quant = quant.permute(1,2,0,3,4,5,6)
        code = code.flatten(start_dim=0,end_dim=1)
        quant = quant.flatten(start_dim=0,end_dim=1)
        
        return code, quant, dequant

    def set_temperature(self,step):
            self.cur_temp = max(1 -(step-self.epoch_offset)/self.total_steps/self.decay_ratio,0)
            self.p_quant = max(self.start + (self.end - self.start) * (step-self.epoch_offset) / self.total_steps / self.decay_ratio,0)


class ScaleBeta_T(nn.Module):
    def __init__(self,bits, group_size, ent_group, grid_size ,signed=False, per_channel=False):
        super().__init__()
        g_v,g_h,n_v,n_h = group_size[0],  group_size[1],  grid_size[0], grid_size[1]
        self.g_v,self.g_h,self.n_v,self.n_h= g_v, g_h, n_v, n_h
        self.g_v_e, self.g_h_e = ent_group[0], ent_group[1]
        initial = torch.ones((1,g_v,g_h,1,1,1,1),dtype=torch.float)
        self.scale = nn.Parameter(initial, requires_grad=True)
        initial2 = torch.zeros((1,g_v,g_h,1,1,1,1),dtype=torch.float)
        self.beta = nn.Parameter(initial2, requires_grad=True)
        self.init = False
        self.signed = signed
        self.per_channel = per_channel
        if self.signed:
            # signed weight/activation is quantized to [-2^(b-1), 2^(b-1)-1]
            self.qmin = - 2 ** (bits - 1)
            self.qmax = 2 ** (bits - 1) - 1
        else:
            # unsigned activation is quantized to [0, 2^b-1]
            self.qmin = 0
            self.qmax = 2 ** bits - 1


    def init_data(self, tensor):
        n_v,n_h,g_v,g_h = self.n_v,self.n_h,self.g_v,self.g_h

        if not self.init:
            device = tensor.device
            fr, no_w, inp_size, out_size = tensor.shape
            tensor = tensor.view(fr,n_v,n_h,inp_size,out_size)
            tensor = tensor.view(fr,g_v,n_v//g_v,g_h,n_h//g_h,inp_size,out_size)
            tensor = tensor.permute(0,1,3,2,4,5,6)
            min_val = torch.amin(tensor, dim=(0, 3, 4, 5, 6), keepdim=True)
            max_val = torch.amax(tensor, dim=(0, 3, 4, 5, 6), keepdim=True)                  
            with torch.no_grad():
                scale = (max_val - min_val) / (self.qmax -self.qmin)
                self.beta.data = min_val.to(device)
                self.scale.data = scale.to(device)
            self.init = True

    def forward(self, x):
        n_fr, n_win, inp_size, out_size = x.shape
        n_v, n_h, g_v, g_h = self.n_v, self.n_h, self.g_v, self.g_h
        g_v_e, g_h_e = self.g_v_e, self.g_h_e
        x = x.view(n_fr,n_v,n_h,inp_size,out_size)
        x = x.view(n_fr,g_v,n_v//g_v,g_h,n_h//g_h,inp_size,out_size)
        x = x.permute(0,1,3,2,4,5,6)
        code = ((x - self.beta) / self.scale)
        quant = (code.round() - code).detach() + code
        dequant = quant*self.scale + self.beta
        dequant = dequant.permute(0,1,3,2,4,5,6)
        dequant = dequant.view(-1,n_v,n_h,inp_size,out_size)
        dequant = dequant.view(-1,n_v*n_h,inp_size,out_size)
        code = code.permute(0,1,3,2,4,5,6) 
        quant = quant.permute(0,1,3,2,4,5,6) 
        code = code.view(n_fr,g_v_e,n_v//g_v_e,g_h_e,n_h//g_h_e,inp_size,out_size)
        quant = quant.view(n_fr,g_v_e,n_v//g_v_e,g_h_e,n_h//g_h_e,inp_size,out_size)
        code = code.permute(0,1,3,2,4,5,6)
        quant = quant.permute(0,1,3,2,4,5,6)
        code = code.permute(1,2,0,3,4,5,6)
        quant = quant.permute(1,2,0,3,4,5,6)
        code = code.flatten(start_dim=0,end_dim=1)
        quant = quant.flatten(start_dim=0,end_dim=1)
        
        return code, quant, dequant



class Log_T(nn.Module):
    def __init__(self, bits, group_size, ent_group, grid_size, signed=False, per_channel=False,rank=None,args=None):
        super().__init__()
        g_v,g_h,n_v,n_h = group_size[0],  group_size[1],  grid_size[0], grid_size[1]
        self.g_v,self.g_h,self.n_v,self.n_h= g_v, g_h, n_v, n_h
        #self.g_v_e, self.g_h_e = ent_group[0], ent_group[1]
        #initial = torch.ones((1,g_v,g_h,1,1,1,1),dtype=torch.float)
        #self.scale = nn.Parameter(initial/(2**bits), requires_grad=True) # Learned online
        #self.x_max = torch.ones((1,g_v,g_h,1,1,1,1),dtype=torch.float) # Updated online
        #self.theta =  nn.Parameter(torch.log(torch.e*torch.ones((1,g_v,g_h,1,1,1,1),dtype=torch.float)-1.0),requires_grad=False)
        self.p_quant = args.quant_droupout
        initial = torch.ones((1,g_v,g_h,1,1,1,1),dtype=torch.float)
        self.scale = nn.Parameter(initial, requires_grad=True)
        initial2 = torch.zeros((1,g_v,g_h,1,1,1,1),dtype=torch.float)
        self.beta = nn.Parameter(initial2, requires_grad=True)
        self.lambdas = torch.ones((1,g_v,g_h,1,1,1,1),dtype=torch.float).to(rank)
        # thata is softplus reparametrization of nu => nu = log(1+exp(thea)) for better stability
        
        self.init = False
        self.signed = signed
        self.per_channel = per_channel  
        self.bits =bits
        if self.signed:
            # signed weight/activation is quantized to [-2^(b-1), 2^(b-1)-1]
            self.qmin = - 2 ** (bits - 1)
            self.qmax = 2 ** (bits - 1) - 1
        else:
            # unsigned activation is quantized to [0, 2^b-1]
            self.qmin = 0
            self.qmax = 2 ** bits - 1
        
    @torch.no_grad
    def init_data(self, tensor):
        n_v,n_h,g_v,g_h = self.n_v,self.n_h,self.g_v,self.g_h   
        if not self.init:
            device = tensor.device
            fr, no_w, inp_size, out_size = tensor.shape
            tensor = tensor.view(fr,n_v,n_h,inp_size,out_size)
            tensor = tensor.view(fr,g_v,n_v//g_v,g_h,n_h//g_h,inp_size,out_size)
            tensor = tensor.permute(0,1,3,2,4,5,6)   
            max_val = torch.amax(tensor.abs(), dim=(0, 3, 4, 5, 6), keepdim=True)          
            self.x_max = max_val
            ## seek for best_cox box
            tensor = tensor.permute(1,2,0,3,4,5,6).flatten(end_dim=1).flatten(1)
            total_error=0
            scale = self.scale.data.view(g_v*g_h)
            beta = self.beta.data.view(g_v*g_h)
            lambdas = self.lambdas.view(g_v*g_h)
            for idx,cur_tensor in enumerate(tensor):
                np_tensor = cur_tensor.detach().cpu().numpy()
                y, cur_lambda = boxcox(np_tensor)
                cur_scale = (np.max(y)-np.min(y))/255
                offset = np.min(y)
                quant = np.round((y-offset)/cur_scale)
                dequant = quant*cur_scale + offset
                dequant = inv_boxcox(dequant,cur_lambda)
                error= np.sqrt(((dequant-np_tensor)**2).sum())
                total_error = total_error + error
                scale[idx].copy_(torch.from_numpy(np.array(cur_scale)).to(device))
                beta[idx].copy_(torch.from_numpy(np.array(offset)).to(device))
                lambdas[idx].copy_(torch.from_numpy(np.array(cur_lambda)).to(device))
            
            # self.scale.data.copy_(self.x_max/(2**self.bits - 1))
            # # search for optimal nu. TODO: search for nu seperately 
            # cand_nu_list = torch.arange(1, 100, 0.1)
            # cand_nu = torch.cat([nu*torch.ones((1,g_v,g_h,1,1,1,1),dtype=torch.float,device=device) for nu in cand_nu_list],dim=0)
            # best_error = math.inf
            # for nu in cand_nu:
            #     code= self.encode(tensor,nu,self.scale)
            #     quant = torch.round(code)
            #     dequant = self.decode(quant,nu,self.scale) 
            #     error = torch.sqrt(((dequant-tensor)**2).sum())   
            #     if error < best_error:
            #         best_nu = nu
            #         best_error=error
            # self.theta.data = torch.log(torch.exp(best_nu.unsqueeze(0))-1)
            self.init = True


    def box_cox(self,x,lambdas):
        return torch.where(lambdas==0,torch.log(x),(x**lambdas -1)/lambdas)

    def inverse_box_cox(self,x,lambdas):
        return torch.where(lambdas==0,torch.exp(x),(lambdas*x+1)**(1/lambdas))





    # def encode(self, x,nu,scale):
        
    #     code = ((self.x_max)*torch.log(1+nu*(x/self.x_max))/torch.log(1+nu))/scale
    #     return code

    # def decode(self, x, nu,scale):
    #     dequant = (torch.exp((x*scale/self.x_max)*torch.log(1+nu))-1)*(self.x_max/nu)
    #     return dequant


    def forward(self, x):
        n_fr, n_win, inp_size, out_size = x.shape
        n_v, n_h, g_v, g_h = self.n_v, self.n_h, self.g_v, self.g_h
        x = x.view(n_fr,n_v,n_h,inp_size,out_size)
        x = x.view(n_fr,g_v,n_v//g_v,g_h,n_h//g_h,inp_size,out_size)
        x = x.permute(0,1,3,2,4,5,6)

        with torch.no_grad():
            self.x_max = torch.amax(x.detach().abs(), dim=(0, 3, 4, 5, 6), keepdim=True)
            x_max = self.x_max
        g = 1.0 / math.sqrt(inp_size*out_size)
        scale = grad_scale(self.scale, g)
        beta = grad_scale(self.beta,g)
        x_cox_box = self.box_cox(x,self.lambdas)
        code = (x_cox_box-beta)/scale 
        #Log
        # theta = grad_scale(self.theta, g)
        # nu = F.softplus(theta,beta=1,threshold=100)
        # code = self.encode(x,nu,scale)
        # # Random quantization
        mask = torch.zeros_like(code)
        mask.bernoulli_(self.p_quant if self.training else 0)
        noise = (code.round() - code).masked_fill(mask.bool(), 0)
        quant = code + noise.detach()
        #quant = (code.round() - code).detach() + code
        dequant = self.inverse_box_cox(quant*scale + beta,self.lambdas)
        #dequant = self.decode(quant,nu,scale)

        dequant = dequant.permute(0,1,3,2,4,5,6)
        dequant = dequant.view(-1,n_v,n_h,inp_size,out_size)
        dequant = dequant.view(-1,n_v*n_h,inp_size,out_size)
        # Entropy Grouping , I got same entropy grouping as parameters for cdf consistency
        code = code.permute(1,2,0,3,4,5,6).flatten(start_dim=0,end_dim=1)
        quant = quant.permute(1,2,0,3,4,5,6).flatten(start_dim=0,end_dim=1)
        #nu = nu.permute(1,2,0,3,4,5,6).flatten(start_dim=0,end_dim=1).flatten(1)
        #x_max = x_max.permute(1,2,0,3,4,5,6).flatten(start_dim=0,end_dim=1).flatten(1)
        #rate_ml = (1/(x.mean((0,3,4,5,6),keepdim=False)+1e-20)).view(-1,1)

        return code, quant, dequant



if __name__ == "__main__":
    # Define args as a class
    class Args:
        def __init__(self):
            self.quant = True
            self.h = 16
            self.w = 16
            self.ksize = (4, 4)
            self.quant_model_bit = 8  
            self.quant_bias_bit = 8   
            self.per_channel_w = False
            self.per_channel_b = False
            self.no_tiles = 1         
            self.n_frames = 2         # Number of frames (adjust as needed)
    args = Args()
    customL = custom_linear(group_size=(2,2),w_nchan=16,input_size=2,output_size=2,complex=True,rank=1,args = args)
    rand_w_real = torch.randint(low=0, high=10, size=(1, 16, 4, 4)).float()
    rand_w_imag = torch.randint(low=0, high=10, size=(1, 16, 4, 4)).float()
    rand_b_real = torch.randint(low=0, high=10, size=(2, 16, 1, 4)).float()
    rand_b_imag = torch.randint(low=0, high=10, size=(2, 16, 1, 4)).float()
    rand_w = torch.view_as_complex(torch.stack([rand_w_real,rand_w_imag],dim=-1))
    rand_b = torch.view_as_complex(torch.stack([rand_b_real,rand_b_imag],dim=-1))
    w_mag = torch.abs(rand_w)  # Magnitude
    w_angle = torch.angle(rand_w)  # Phase
    b_mag = torch.abs(rand_b)
    b_angle = torch.angle(rand_b)
    # Computation of scales are tested and appropriately works
    customL.weight = nn.Parameter(torch.view_as_real(rand_w))
    customL.bias = nn.Parameter(torch.view_as_real(rand_b))
    customL.weight_angle_quantizer.init_data(w_angle)
    customL.bias_angle_quantizer.init_data(b_angle)
    customL.weight_mag_quantizer.init_data(w_mag)
    customL.bias_mag_quantizer.init_data(b_mag)
    # Scaled, Qauntized, Dequantized, We define function recall all weight quantizer before model call to simulate quantization
    # and compute code burden
    code_w_ang, quant_w_ang, dequant_w_angle = customL.weight_angle_quantizer(w_angle)
    customL.dequant_w_angle = dequant_w_angle
    code_w_mag, quant_w_mag, dequant_w_mag = customL.weight_mag_quantizer(w_mag)
    customL.dequant_w_mag = dequant_w_mag
    code_b_mag, quant_b_mag, dequant_b_mag = customL.bias_mag_quantizer(b_mag)
    customL.dequant_b_mag = dequant_b_mag
    code_b_mag, quant_b_mag, dequant_b_angle = customL.bias_angle_quantizer(b_angle)
    customL.dequant_b_angle = dequant_b_angle

    # Check multiplication
    rand_inp_real = torch.randint(low=0, high=10, size=(1, 16, 4, 4)).float()    
    rand_inp_imag = torch.randint(low=0, high=10, size=(1, 16, 4, 4)).float()  
    rand_inp = torch.stack([rand_inp_real,rand_inp_imag],dim=-1) 
    rand_inp = torch.view_as_complex(rand_inp)   
    out = customL(rand_inp)
    

