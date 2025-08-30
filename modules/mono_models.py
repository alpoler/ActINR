#!/usr/bin/env python

import pdb
import math

import numpy as np

import torch
from torch import nn
    
class ReLULayer(nn.Module):
    '''
        Drop in replacement for SineLayer but with ReLU non linearity
    '''
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30, scale=10.0):
        '''
            is_first, and omega_0 are not used.
        '''
        super().__init__()
        self.in_features = in_features
        self.omega_0 = omega_0
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
    def forward(self, input):
        return nn.functional.relu(self.linear(input))
    
class ReLUNormalizedLayer(nn.Module):
    '''
        Drop in replacement for SineLayer but with ReLU non linearity
    '''
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30, scale=10.0):
        '''
            is_first, and omega_0 are not used.
        '''
        super().__init__()
        self.in_features = in_features
        self.omega_0 = omega_0
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.bn = nn.BatchNorm1d(out_features)
        
    def forward(self, input):
        output = nn.functional.leaky_relu(self.linear(input))
        return self.bn(output[0, ...])[None, ...]
    
class GaussLayer(nn.Module):
    '''
        Drop in replacement for SineLayer but with Gaussian non linearity
    '''
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30, scale=10.0):
        '''
            is_first, and omega_0 are not used.
        '''
        super().__init__()
        self.in_features = in_features
        self.omega_0 = omega_0
        self.scale = scale
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
    def forward(self, input):
        return torch.exp(-(self.scale*self.linear(input))**2)
    
class MexicanHatLayer(nn.Module):
    '''
        Drop in replacement for SineLayer but with Mexican hat non linearity
    '''
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30, scale=10.0):
        '''
            is_first, and omega_0 are not used.
        '''
        super().__init__()
        self.in_features = in_features
        self.omega_0 = omega_0
        self.scale = scale
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
    def forward(self, input):
        x2 = self.linear(input)**2
        return (1 - self.scale*x2) * torch.exp(-self.scale*x2)
    
class DifferenceOfGaussianLayer(nn.Module):
    '''
        Drop in replacement for SineLayer but with Mexican hat non linearity
    '''
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30, scale=10.0):
        '''
            is_first, and omega_0 are not used.
        '''
        super().__init__()
        self.in_features = in_features
        self.omega_0 = omega_0
        self.scale = scale
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.k = omega_0
        
    def forward(self, input):
        x2 = self.linear(input)**2
        return torch.exp(-self.scale*self.k*x2) - torch.exp(-self.scale*x2)
    
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
                 is_first=False, omega_0=30, scale=10.0, init_weights=True):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        if True:
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

class RealGaborLayer(nn.Module):
    '''
        Implicit representation with Gabor nonlinearity
        
        Inputs;
            in_features: Input features
            out_features; Output features
            bias: if True, enable bias for the linear operation
            is_first: Legacy SIREN parameter
            omega_0: Legacy SIREN parameter
            omega: Frequency of Gabor sinusoid term
            scale: Scaling of Gabor Gaussian term
    '''
    
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30, omega=10.0, scale=10.0):
        super().__init__()
        self.omega_0 = omega_0
        self.scale_0 = scale
        self.is_first = is_first
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            const = 2*np.sqrt(6 / self.in_features) / max(1e-3, self.omega_0)
            self.linear.weight.uniform_(-const, const)
        
    def forward(self, input):
        lin_output = self.linear(input)  
        omega = self.omega_0 * lin_output
        arg = self.scale_0*self.scale_0*lin_output**2
        
        return torch.cos(omega)*torch.exp(-arg)

class ComplexGabor3DLayer(nn.Module):
    '''
        Implicit representation with complex Gabor nonlinearity
        
        Inputs;
            in_features: Input features
            out_features; Output features
            bias: if True, enable bias for the linear operation
            is_first: Legacy SIREN parameter
            omega_0: Legacy SIREN parameter
            omega: Frequency of Gabor sinusoid term
            scale: Scaling of Gabor Gaussian term
    '''
    
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30, omega=10.0, scale=60.0):
        super().__init__()
        self.omega_0 = omega_0
        self.scale_0 = scale
        self.is_first = is_first
        
        self.in_features = in_features
        
        if self.is_first:
            dtype = torch.float
        else:
            dtype = torch.cfloat
        self.linear = nn.Linear(in_features,
                                out_features,
                                bias=bias,
                                dtype=dtype)
        
        self.scale_x = nn.Linear(in_features, out_features,
                                 bias=bias, dtype=dtype)
        self.scale_y = nn.Linear(in_features, out_features,
                                 bias=bias, dtype=dtype)
        self.scale_z = nn.Linear(in_features, out_features,
                                 bias=bias, dtype=dtype)
        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            const = np.sqrt(6 / self.in_features) / max(1e-3, self.omega_0)
            self.linear.weight.uniform_(-const, const)
            
        # Should first layer be real-only weights?
        
    def forward(self, input):
        lin = self.linear(input)
        omega = self.omega_0 * lin
        
        scale_x = self.scale_x(input)
        scale_y = self.scale_y(input)
        scale_z = self.scale_z(input)
        
        freq_term = torch.exp(1j*omega)
        
        arg = scale_x.abs().square() +\
              scale_y.abs().square() +\
              scale_z.abs().square()
        gauss_term = torch.exp(-self.scale_0*arg/3)
                
        return freq_term*gauss_term

class ComplexGabor2DLayer(nn.Module):
    '''
        Implicit representation with complex Gabor nonlinearity
        
        Inputs;
            in_features: Input features
            out_features; Output features
            bias: if True, enable bias for the linear operation
            is_first: Legacy SIREN parameter
            omega_0: Legacy SIREN parameter
            omega: Frequency of Gabor sinusoid term
            scale: Scaling of Gabor Gaussian term
    '''
    
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30, omega=10.0, scale=60.0):
        super().__init__()
        self.omega_0 = omega_0
        self.scale_0 = scale
        self.is_first = is_first
                
        self.in_features = in_features
        
        if self.is_first:
            dtype = torch.float
        else:
            dtype = torch.cfloat
        self.linear = nn.Linear(in_features,
                                out_features,
                                bias=bias,
                                dtype=dtype)
        
        #self.scale_x = nn.Linear(in_features, out_features,
        #                         bias=bias, dtype=dtype)
        self.scale_y = nn.Linear(in_features, out_features,
                                 bias=bias, dtype=dtype)
        
        #self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            const = 2*np.sqrt(6 / self.in_features) / max(1e-3, self.omega_0)
            self.linear.weight.uniform_(-const, const)
            
    def forward(self, input):
        lin = self.linear(input)
        
        scale_x = lin #self.scale_x(input)
        scale_y = self.scale_y(input)
        
        freq_term = torch.exp(1j*self.omega_0*lin)
        
        arg = scale_x.abs().square() + scale_y.abs().square()
        gauss_term = torch.exp(-self.scale_0*self.scale_0*arg)
                
        return freq_term*gauss_term
    
class ComplexGabor1DLayer(nn.Module):
    '''
        Implicit representation with complex Gabor nonlinearity
        
        Inputs;
            in_features: Input features
            out_features; Output features
            bias: if True, enable bias for the linear operation
            is_first: Legacy SIREN parameter
            omega_0: Legacy SIREN parameter
            omega: Frequency of Gabor sinusoid term
            scale: Scaling of Gabor Gaussian term
    '''
    
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30, omega=10.0, scale=60.0):
        super().__init__()
        self.omega_0 = omega_0
        self.scale_0 = scale
        self.is_first = is_first
        
        use_siglinear = False
        
        self.in_features = in_features
        
        if self.is_first:
            dtype = torch.float
        else:
            dtype = torch.cfloat
            self.omega_0 = 1
       
        self.linear = nn.Linear(in_features,
                                out_features,
                                bias=bias,
                                dtype=dtype)
        #self.init_weights()
        
    @torch.no_grad()
    def init_weights(self):
        
            if self.is_first:
                const = np.sqrt(1/self.in_features)
                #self.linear.bias.uniform_(-0.5, 0.5)
            else:
                const = 2*np.sqrt(6 / self.in_features) / max(1e-3, self.omega_0)
            self.linear.weight.uniform_(-const, const)
            
    def forward(self, input):
        lin = self.linear(input)
        omega = self.omega_0 * lin
                
        arg = 1j*omega - self.scale_0*self.scale_0*lin.abs().square()
                
        return torch.exp(arg)
    
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
            fn_samples = sidelength
            self.num_frequencies = 4
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
    
class INR(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, 
                 out_features, nonlinearity='sine', outermost_linear=True,
                 first_omega_0=30, hidden_omega_0=30., scale=10.0,
                 pos_encode=False, sidelength=512, fn_samples=None,
                 use_nyquist=True):
        super().__init__()
        self.pos_encode = pos_encode
        
        self.complex = False
        if nonlinearity == 'sine':
            self.nonlin = SineLayer
        elif nonlinearity == 'gauss':
            self.nonlin = GaussLayer
        elif nonlinearity == 'gabor':
            self.nonlin = ComplexGabor1DLayer
            hidden_features = int(hidden_features/np.sqrt(2))
            self.complex = True
        elif nonlinearity == 'dog':
            self.nonlin = DifferenceOfGaussianLayer
        elif nonlinearity == 'gabor2d':
            self.nonlin = ComplexGabor2DLayer
            self.complex = True
            hidden_features = int(hidden_features/2)
        elif nonlinearity == 'realgabor':
            self.nonlin = RealGaborLayer
        elif nonlinearity == 'mexicanhat':
            self.nonlin = MexicanHatLayer
        elif nonlinearity == 'relunorm':
            self.nonlin = ReLUNormalizedLayer
        elif nonlinearity == 'relu':
            self.nonlin = ReLULayer
        else:
            raise ValueError('Nonlinearity not known')
            
        if pos_encode:
            self.positional_encoding = PosEncoding(in_features=in_features,
                                                   sidelength=sidelength,
                                                   fn_samples=fn_samples,
                                                   use_nyquist=use_nyquist)
            in_features = self.positional_encoding.out_dim
            
        self.net = []
        self.net.append(self.nonlin(in_features, hidden_features, 
                                  is_first=True, omega_0=first_omega_0,
                                  scale=scale))

        for i in range(hidden_layers):
            self.net.append(self.nonlin(hidden_features, hidden_features, 
                                      is_first=False, omega_0=hidden_omega_0,
                                      scale=scale))

        if outermost_linear:
            if self.complex:
                dtype = torch.cfloat
            else:
                dtype = torch.float
            final_linear = nn.Linear(hidden_features,
                                     out_features,
                                     dtype=dtype)
            
            if nonlinearity in ['sine', 'gabor', 'realgabor', 'gabor2d']:
                with torch.no_grad():
                    const = np.sqrt(6/hidden_features)/max(hidden_omega_0, 1e-12)
                    final_linear.weight.uniform_(-const, const)
                        
            self.net.append(final_linear)
        else:
            self.net.append(self.nonlin(hidden_features, out_features, 
                                      is_first=False, omega_0=hidden_omega_0,
                                      scale=scale))
        
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords):
        if self.pos_encode:
            coords = self.positional_encoding(coords)
            
        output = self.net(coords)
        if self.complex:
            self.imag_output = output.imag
            output = output.real
            
        return output
    
class ResINR(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, 
                 out_features, nonlinearity='sine', outermost_linear=True,
                 first_omega_0=30, hidden_omega_0=30., scale=10.0,
                 pos_encode=False, sidelength=512, fn_samples=None,
                 use_nyquist=True):
        super().__init__()
        self.pos_encode = pos_encode
        self.hidden_layers = hidden_layers
        
        self.complex = False
        if nonlinearity == 'sine':
            self.nonlin = SineLayer
        elif nonlinearity == 'gauss':
            self.nonlin = GaussLayer
        elif nonlinearity == 'gabor':
            self.nonlin = ComplexGabor1DLayer
            hidden_features = int(hidden_features/np.sqrt(2))
            self.complex = True
        elif nonlinearity == 'dog':
            self.nonlin = DifferenceOfGaussianLayer
        elif nonlinearity == 'gabor2d':
            self.nonlin = ComplexGabor2DLayer
            self.complex = True
            hidden_features = int(hidden_features/2)
        elif nonlinearity == 'realgabor':
            self.nonlin = RealGaborLayer
        elif nonlinearity == 'mexicanhat':
            self.nonlin = MexicanHatLayer
        elif nonlinearity == 'relunorm':
            self.nonlin = ReLUNormalizedLayer
        elif nonlinearity == 'relu':
            self.nonlin = ReLULayer
        else:
            raise ValueError('Nonlinearity not known')
            
        if pos_encode:
            self.positional_encoding = PosEncoding(in_features=in_features,
                                                   sidelength=sidelength,
                                                   fn_samples=fn_samples,
                                                   use_nyquist=use_nyquist)
            in_features = self.positional_encoding.out_dim
            
        self.net = []
        self.net.append(self.nonlin(in_features, hidden_features, 
                                  is_first=True, omega_0=first_omega_0,
                                  scale=scale))

        if self.complex:
            dtype = torch.cfloat
        else:
            dtype = torch.float
            
        for i in range(hidden_layers):
            self.net.append(self.nonlin(hidden_features, hidden_features, 
                                      is_first=False, omega_0=hidden_omega_0,
                                      scale=scale))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features*hidden_layers,
                                     out_features,
                                     dtype=dtype)
            
            if nonlinearity in ['sine', 'gabor', 'realgabor', 'gabor2d']:
                with torch.no_grad():
                    const = np.sqrt(6/hidden_features)/max(hidden_omega_0, 1e-12)
                    final_linear.weight.uniform_(-const, const)
                        
            self.net.append(final_linear)
        else:
            self.net.append(self.nonlin(hidden_features, out_features, 
                                      is_first=False, omega_0=hidden_omega_0,
                                      scale=scale))
        
        self.net = nn.ModuleList(self.net)
    
    def forward(self, coords):
        if self.pos_encode:
            coords = self.positional_encoding(coords)
            
        inp = coords
        final_input = []
        
        for idx in range(self.hidden_layers):
            output = self.net[idx](inp)
            inp = output
            const = (idx + 1)/(self.hidden_layers + 1)
            final_input.append(output*const)
            
        final_input = torch.cat(final_input, -1)
        output = self.net[-1](final_input)
            
        if self.complex:
            self.imag_output = output.imag
            output = output.real
            
        return output
    
        
class GaborSAPE(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, 
                 out_features, first_omega_0=30, hidden_omega_0=30.,
                 scale=10.0):
        
        super().__init__()
        
        self.nonlin = ComplexGabor1DLayer
        self.complex = True
        
        self.net = []
        self.net.append(ReLULayer(in_features, hidden_features))

        for i in range(hidden_layers-1):
            self.net.append(ReLULayer(hidden_features,
                                                hidden_features))
        
        self.net.append(self.nonlin(hidden_features, hidden_features, 
                                  is_first=True, omega_0=first_omega_0,
                                  scale=scale))
        if self.complex:
            dtype = torch.cfloat
        else:
            dtype = torch.float
        final_linear = nn.Linear(hidden_features,
                                    out_features,
                                    dtype=dtype)
        
        with torch.no_grad():
            const = np.sqrt(6/hidden_features)/max(hidden_omega_0, 1e-12)
            final_linear.weight.uniform_(-const, const)
                
        self.net.append(final_linear)
        
        self.net = nn.Sequential(*self.net)
        
    def forward(self, coords):
            
        output = self.net(coords)
        if self.complex:
            output = output.real
            self.imag_output = output.imag
            
        return output
        
class Wire(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, 
                 out_features, wavelet='gabor', scale=10.0, omega=10.0):
        super().__init__()
        
        # Only Gabor filter implemented right now
        if wavelet == 'gabor':
            self.nonlin = ComplexGabor2DLayer
            
        self.net = []
        
        self.net.append(self.nonlin(in_features,
                                    hidden_features, 
                                    scale=scale,
                                    omega=omega))

        for i in range(hidden_layers):
            self.net.append(self.nonlin(hidden_features,
                                        hidden_features, 
                                        scale=scale,
                                        omega=omega))

    
        final_linear = nn.Linear(hidden_features, out_features)            
        self.net.append(final_linear)
        
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords):
        output = self.net(coords)
         
        return output
    
# GaborLayer and GaborNet from https://github.com/addy1997/mfn-pytorch/blob/main/model/MultiplicativeFilterNetworks.py
class GaborLayer(nn.Module):
    def __init__(self, in_dim, out_dim, padding, alpha, beta=1.0, bias=False):
        super(GaborLayer, self).__init__()

        self.mu = nn.Parameter(torch.rand((out_dim, in_dim)) * 2 - 1)
        self.gamma = nn.Parameter(torch.distributions.gamma.Gamma(alpha, beta).sample((out_dim, )))
        self.linear = torch.nn.Linear(in_dim, out_dim)
        #self.padding = padding

        self.linear.weight.data *= 128. * torch.sqrt(self.gamma.unsqueeze(-1))
        self.linear.bias.data.uniform_(-np.pi, np.pi)

        # Bias parameters start in zeros
        #self.bias = nn.Parameter(torch.zeros(self.responses)) if bias else None

    def forward(self, input):
        norm = (input ** 2).sum(dim=1).unsqueeze(-1) + (self.mu ** 2).sum(dim=1).unsqueeze(0) - 2 * input @ self.mu.T
        return torch.exp(- self.gamma.unsqueeze(0) / 2. * norm) * torch.sin(self.linear(input))


class GaborNet(nn.Module):
    def __init__(self, in_dim=2, hidden_dim=256, out_dim=1, k=4):
        super(GaborNet, self).__init__()

        self.k = k
        self.gabon_filters = nn.ModuleList([GaborLayer(in_dim, hidden_dim, 0, alpha=6.0 / k) for _ in range(k)])
        self.linear = nn.ModuleList(
            [torch.nn.Linear(hidden_dim, hidden_dim) for _ in range(k - 1)] + [torch.nn.Linear(hidden_dim, out_dim)])

        for lin in self.linear[:k - 1]:
            lin.weight.data.uniform_(-np.sqrt(1.0 / hidden_dim), np.sqrt(1.0 / hidden_dim))

    def forward(self, x):

        # Recursion - Equation 3
        zi = self.gabon_filters[0](x[0, ...])  # Eq 3.a
        for i in range(self.k - 1):
            zi = self.linear[i](zi) * self.gabon_filters[i + 1](x[0, ...])
            # Eq 3.b

        return self.linear[self.k - 1](zi)[None, ...]  # Eq 3.c

@torch.no_grad()
def get_layer_outputs(model, coords, imsize,
                      nfilters_vis=16,
                      get_imag=False):
    '''
        get activation images after each layer
        
        Inputs:
            model: INR model
            coords: 2D coordinates
            imsize: Size of the image
            nfilters_vis: Number of filters to visualize
            get_imag: If True, get imaginary component of the outputs
            
        Outputs:
            atoms_montages: A list of 2d grid of outputs
    '''
    H, W = imsize

    if model.pos_encode:
        coords = model.positional_encoding(coords)
        
    atom_montages = []
    
    for idx in range(len(model.net)-1):
        layer_output = model.net[idx](coords)
        layer_images = layer_output.reshape(1, H, W, -1)[0]
        
        if nfilters_vis is not 'all':
            layer_images = layer_images[..., :nfilters_vis]
        
        if get_imag:
            atoms = layer_images.detach().cpu().numpy().imag
        else:
            atoms = layer_images.detach().cpu().numpy().real
            
        #atoms = normalize(atoms, True)
        
        atoms_min = atoms.min(0, keepdims=True).min(1, keepdims=True)
        atoms_max = atoms.max(0, keepdims=True).max(1, keepdims=True)
        
        signs = (abs(atoms_min) > abs(atoms_max))
        atoms = (1 - 2*signs)*atoms
        
        # Arrange them by variance
        atoms_std = atoms.std((0,1))
        std_indices = np.argsort(atoms_std)
        
        atoms = atoms[..., std_indices]
        
        atoms_min = atoms.min(0, keepdims=True).min(1, keepdims=True)
        atoms_max = atoms.max(0, keepdims=True).max(1, keepdims=True)
        
        atoms = (atoms - atoms_min)/np.maximum(1e-14, atoms_max - atoms_min)
        
        atoms[:, [0, -1], :] = 1
        atoms[[0, -1], :, :] = 1
        
        atoms_montage = build_montage(np.transpose(atoms, [2, 0, 1]))
        
        atom_montages.append(atoms_montage)
        coords = layer_output
        
    return atom_montages
