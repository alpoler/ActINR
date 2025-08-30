from pyexpat.errors import messages
import torch
import math
from torch.autograd import Function
import constriction
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from compressai.ans import BufferedRansEncoder, RansDecoder
from functools import partial

def get_np_size(x):
    return x.size * x.itemsize

def cdf_log_exp(y, rate, x_max, nu,scale):
    z = (x_max / nu) * (torch.exp(y*torch.log(1+nu)*scale/x_max) - 1.0)
    return (1 - torch.exp(-rate * z)).clamp_(min=0)  


def icdf_log_exp(y,rate,x_max,nu,scale):
    z= 1-(nu*torch.log(1-y)/(rate*x_max))
    return (x_max/torch.log(1+nu)/scale)*torch.log(z)


def np_cdf_log_exp(y, rate, x_max, nu,scale):
    z = (x_max / nu) * (math.exp(y*math.log(1+nu)*scale/x_max) - 1.0)
    result = (1 - math.exp(-rate * z))
    return max(0,result)

def np_icdf_log_exp(y,rate,x_max,nu,scale):
    z= 1-(nu*math.log(1-y)/(rate*x_max))
    return (x_max/math.log(1+nu)/scale)*math.log(z)

class LogExponential:
    def __init__(self,rate, x_max, nu,scale):
        self.rate = rate
        self.x_max = x_max
        self.nu = nu
        self.scale=scale
    
    def cdf(self,y):
        z = (self.x_max / self.nu) * (torch.exp(y*torch.log(1+self.nu)*self.scale/self.x_max) - 1.0)
        return (1 - torch.exp(-self.rate * z)).clamp_(min=0)  




class DiffEntropyModel():
    def __init__(self, distribution="gaussian"):
        self.distribution = distribution

    def cal_bitrate(self, code, quant, training,soft_round=False):
        return self.cal_global_bitrate(code, quant, training,soft_round)


    def cal_global_bitrate(self, code_list, quant_list, training,soft_round=False):
        if training:
            code =code_list.flatten(1)
            if self.distribution == "gaussian":
                means = torch.mean(code,dim=1,keepdim=True)
                stds = torch.std(code,dim=1,keepdim=True)
            else:
                means = torch.median(code,dim=1,keepdim=True)[0]
                stds = torch.mean(torch.abs(code-means),dim=1,keepdim=True)
            if not soft_round:
                noise = torch.empty_like(code).uniform_(-0.5, 0.5)
                code = code+noise
            total_bits = torch.sum(self.get_bits(code, means, stds))
            total_real_bits = 0
        else:
            bit_list = [] 
            real_bit_list = []
            means= []
            stds= []
            idxxx = 0
            for code,quant in zip(torch.unbind(code_list, dim=0), torch.unbind(quant_list, dim=0)):

                if self.distribution == "gaussian":
                    mean = torch.mean(code)
                    std = torch.std(code)
                    means.append(mean)
                    stds.append(std)
                    code = quant
                    real_bits = compress_matrix_flatten_gaussian_global(code, mean, std)
                else:
                    mean = torch.median(code)
                    std = torch.mean(torch.abs(code-mean))
                    means.append(mean)
                    stds.append(std)
                    code = quant

                    all_same = torch.all(code == code.flatten()[0])
                    real_bits = compress_matrix_flatten_categorical(code) if (not all_same) else 0
                
                idxxx+=1
                real_bit_list.append(real_bits)
           
            
            means = torch.stack(means) 
            stds = torch.stack(stds)
            total_bits = torch.sum(self.get_bits(code_list.flatten(1),means.view(-1,1),stds.view(-1,1)))
            total_real_bits = sum(real_bit_list)

        return {"bitrate": total_bits, "mean":means, "std":stds, "real_bitrate":total_real_bits}

    def get_bits(self, x, mu, sigma):
        sigma = sigma.clamp(1e-5, 1e10)
        if self.distribution == "gaussian":
            gaussian = torch.distributions.normal.Normal(mu, sigma)
        else:
            gaussian = torch.distributions.laplace.Laplace(mu, sigma)
        probs = gaussian.cdf(x + 0.5) - gaussian.cdf(x - 0.5)
        bits = -1.0 * torch.log(probs + 1e-5) / math.log(2.0)
        bits = LowerBound.apply(bits, 0)
        return bits


def compress_matrix_flatten_gaussian_global(matrix, mean, std):
    '''
    :param matrix: tensor
    :return compressed, symtable
    '''
    mean = mean.item()
    std = std.clamp(1e-5, 1e10).item()
    min_value, max_value = matrix.min().int().item(), matrix.max().int().item()
    if min_value == max_value:
        max_value = min_value + 1
    message = np.array(matrix.int().flatten().tolist(), dtype=np.int32)
    entropy_model = constriction.stream.model.QuantizedGaussian(min_value, max_value, mean, std)
    encoder = constriction.stream.stack.AnsCoder()
    encoder.encode_reverse(message, entropy_model)
    compressed = encoder.get_compressed()
    total_bits = get_np_size(compressed) * 8
    return total_bits




def compress_matrix_flatten_exponential_global(matrix, rate_ml,x_max,nu,scale):
    '''
    :param matrix: tensor
    :return compressed, symtable
    '''
    
    rate = rate_ml.item()
    x_max = x_max.item()
    nu = nu.item()
    scale = scale.item()
    rate = np.array([rate], dtype=np.float32)
    x_max = np.array([x_max], dtype=np.float32)
    scale = np.array([scale], dtype=np.float32)
    nu = np.array([nu], dtype=np.float32)
    min_value, max_value = matrix.min().int().item(), matrix.max().int().item()
    if min_value == max_value:
        max_value = min_value + 1

    message = np.array(matrix.int().flatten().tolist(), dtype=np.int32)
    fixed_cdf = partial(np_cdf_log_exp,rate=rate,x_max=x_max,nu=nu,scale=scale)
    fixed_icdf = partial(np_icdf_log_exp,rate=rate,x_max=x_max,nu=nu,scale=scale) 
    entropy_model = constriction.stream.model.CustomModel(fixed_cdf, fixed_icdf,min_value,max_value)
    encoder = constriction.stream.stack.AnsCoder()
    encoder.encode_reverse(message, entropy_model)
    compressed = encoder.get_compressed()
    total_bits = get_np_size(compressed) * 8
    return total_bits



def compress_matrix_flatten_categorical(matrix):
    '''
    :param matrix: np.array
    :return compressed, symtable
    '''
    matrix = np.array(matrix.detach().cpu()) #matrix.flatten()
    unique, unique_indices, unique_inverse, unique_counts = np.unique(matrix, return_index=True, return_inverse=True, return_counts=True, axis=None)
    min_value = np.min(unique)
    max_value = np.max(unique)
    unique = unique.astype(judege_type(min_value, max_value))
    message = unique_inverse.astype(np.int32)
    probabilities = unique_counts.astype(np.float64) / np.sum(unique_counts).astype(np.float64)
    entropy_model = constriction.stream.model.Categorical(probabilities)
    encoder = constriction.stream.stack.AnsCoder()
    encoder.encode_reverse(message, entropy_model)
    compressed = encoder.get_compressed()
    total_bits = get_np_size(compressed) * 8
    return total_bits

def judege_type(min, max):
    if min>=0:
        if max<=256:
            return np.uint8
        elif max<=65535:
            return np.uint16
        else:
            return np.uint32
    else:
        if max<128 and min>=-128:
            return np.int8
        elif max<32768 and min>=-32768:
            return np.int16
        else:
            return np.int32


class LowerBound(Function):
    @staticmethod
    def forward(ctx, inputs, bound):
        b = torch.ones_like(inputs) * bound
        ctx.save_for_backward(inputs, b)
        return torch.max(inputs, b)

    @staticmethod
    def backward(ctx, grad_output):
        inputs, b = ctx.saved_tensors
        pass_through_1 = inputs >= b
        pass_through_2 = grad_output < 0

        pass_through = pass_through_1 | pass_through_2
        return pass_through.type(grad_output.dtype) * grad_output, None


