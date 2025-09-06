import math
import torch
from torch.nn.modules.module import Module
from typing import Optional, List, Tuple, Union
from torch.nn.modules.utils import _single, _pair, _triple, _reverse_repeat_tuple, _ntuple
from torch.nn.parameter import Parameter
from torch import Tensor
from torch.nn import init

class ConvDecoder(Module):
    def __init__(
        self,
        block_size: Tuple[int, ...],
        conv_dim: int,
        rank: int,
        init_type: str,
        decode_norm: str,
        decode_matrix:str,
        std: float,
        no_shift: bool,
        device=None,
        dtype=None,
        num_hidden:int = 0,
        hidden_sizes:int = 9,
        use_bn:bool = False,
        nonlinearity:str = 'none',
        **kwargs
    ) -> None:
        super(ConvDecoder, self).__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        channels = math.prod(block_size)
        conv_dim = channels if conv_dim == -1 else conv_dim
        self.block_size = block_size
        self.decode_matrix = decode_matrix
        self.channels = channels
        self.conv_dim = conv_dim
        self.norm = decode_norm
        self.div = 1 if decode_norm == 'none' else None
        self.num_hidden, self.use_bn =  num_hidden, use_bn
        if num_hidden>0:
            self.hidden_sizes = _ntuple(num_hidden)(hidden_sizes)
        self.no_shift = no_shift
        if num_hidden == 0:
            if decode_matrix == 'sq':
                self.scale = Parameter(torch.empty((conv_dim, channels), **factory_kwargs).to(rank)) 
            elif decode_matrix == 'dft':
                self.scale = Parameter(torch.empty((1,channels), **factory_kwargs))
            elif decode_matrix == 'dft_fixed':
                self.scale = Parameter(torch.empty((1,channels), **factory_kwargs),requires_grad=True)
            self.shift = Parameter(torch.empty((1,conv_dim), **factory_kwargs)) if not no_shift else 0.0
            if decode_matrix != 'sq':
                self.dft = Parameter(get_dft_matrix(conv_dim, channels), requires_grad=False)
            self.reset_parameters(init_type, std)
        else:
            act_dict = {'none':torch.nn.Identity(), 'sigmoid':Sigmoid(), 'tanh':torch.nn.Tanh(),
                             'relu':torch.nn.ReLU(), 'sine':Sine(30.0)}
            self.act = act_dict[nonlinearity]
            layers = []
            inp_dim = self.conv_dim
            for l in range(num_hidden):
                out_dim = self.hidden_sizes[l]
                out_dim = self.channels if out_dim == -1 else out_dim
                layers.append(torch.nn.Linear(inp_dim,out_dim,bias=not self.no_shift))
                if use_bn:
                    layers.append(BNGlobal())
                layers.append(self.act)
                inp_dim = out_dim
            layers.append(torch.nn.Linear(inp_dim,self.channels,bias=not self.no_shift))
            self.layers = torch.nn.Sequential(*layers)
            self.reset_parameters('random')
                    

    def reset_parameters(self, init_type, std=1.0) -> None:
        if self.num_hidden == 0:
            if init_type == 'random':
                init.normal_(self.scale, std=std)
                # init.normal_(self.shift)
            elif init_type == 'constant':
                init.constant_(self.scale, std)
            elif init_type == 'value':
                self.scale.data = std
            else:
                raise Exception(f'unknown init_type {init_type}')
            if not self.no_shift:
                init.zeros_(self.shift)
        else:
            assert init_type == 'random'
            for i,layer in enumerate(self.layers.children()):
                if isinstance(layer,torch.nn.Linear):
                    w_std = (1/layer.in_features) if i==0 else (math.sqrt(6/layer.in_features)/30)
                    if i == len(list(self.layers.children()))-1:
                        w_std = std/layer.in_features
                    # torch.nn.init.constant_(layer.weight, w_std)
                    torch.nn.init.uniform_(layer.weight, -w_std, w_std)
                    if layer.bias is not None:
                        # torch.nn.init.constant_(layer.bias, w_std)
                        torch.nn.init.uniform_(layer.bias, -w_std, w_std)

    def forward(self, input: Tensor) -> Tensor:
        # assert input.dim() == 4 and input.size(2)*input.size(3)==self.channels
        # w_in = input.reshape(input.size(0),input.size(1)*input.size(2)*input.size(3)) #assume oixhw
        if self.num_hidden == 0:
            # print(self.div, input.max(),input.min(), )
            if self.decode_matrix == 'sq':
                w_out = torch.matmul(input/self.div+self.shift,self.scale)
            else:
                w_out = torch.matmul(input/self.div+self.shift,self.dft)*self.scale
        else:
            w_out = self.layers(input/self.div)

        return w_out

def get_dft_matrix(conv_dim, channels):
    dft = torch.zeros(conv_dim,channels)
    for i in range(conv_dim):
        for j in range(channels):
            dft[i,j] = math.cos(torch.pi/channels*(i+0.5)*j)/math.sqrt(channels)
            dft[i,j] = dft[i,j]*(math.sqrt(2) if j>0 else 1)
    return dft

class Sigmoid(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.act = torch.nn.Sigmoid()

    def forward(self, x):
        return 2*self.act(x)-1

class Sine(torch.nn.Module):
    """Sine activation with scaling.
    Args:
        w0 (float): Omega_0 parameter.
    """
    def __init__(self, w0=1.):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)
    

class BNGlobal(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return (x - torch.mean(x,dim=0))/(torch.std(x,dim=0)+1e-8)