import torch
from torch import nn
from torch import Tensor
from torch.nn.parameter import Parameter, UninitializedParameter
from torch.nn import init
import math
import torch.nn.functional as F

class SigLinear(nn.Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = A(x + b)`
    
    Derived from the original pytorch linear module:
    https://pytorch.org/docs/stable/_modules/torch/nn/modules/linear.html#Linear

    This module supports :ref:`TensorFloat32<tf32_on_ampere>`.

    On certain ROCm devices, when using float16 inputs this module will use :ref:`different precision<fp16_on_mi200>` for backward.

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(*, H_{in})` where :math:`*` means any number of
          dimensions including none and :math:`H_{in} = \text{in\_features}`.
        - Output: :math:`(*, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in\_features})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\text{in\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in\_features}}`

    Examples::

        >>> m = siglinear.SigLinear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(in_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, fan_out = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in*self.in_features) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)
            
    def init_bias(self) -> None:
        if self.bias is not None:
            init.uniform_(self.bias, -1, 1)

    def forward(self, input: Tensor) -> Tensor:
        # Only implemented for batched matrices
        B, R, C = input.shape
        x_plus_bias = input + torch.tile(self.bias[None, None, ...], [B, R, 1])
        
        y_times_weight = torch.bmm(x_plus_bias,
                                   torch.tile(self.weight.t(),
                                              [B, 1, 1]))
        return y_times_weight

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )
        
if __name__ == '__main__':
    in_feat = 2
    out_feat = 100
    
    inmat = torch.rand(4, 256, 2, dtype=torch.cfloat)
    
    linear = SigLinear(in_feat, out_feat, dtype=torch.cfloat)
    outmat = linear(inmat)