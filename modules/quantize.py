import torch
from copy import deepcopy

class qfn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, k):
        n = float(2**(k-1) - 1)
        out = torch.floor(torch.abs(input) * n) / n
        out = out*torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None
    

def compute_best_quant_axis(x, thres=0.05):
    """
    Compute the best quantization axis for a tensor. 
    Similar to the one used in HNeRV quantization: https://github.com/haochen-rye/HNeRV/blob/main/hnerv_utils.py#L26
    """
    best_axis = None
    best_axis_dim = 0
    for axis in range(x.ndim):
        dim = x.shape[axis]
        if x.numel() / dim >= x.numel() * thres:
            continue
        if dim > best_axis_dim:
            best_axis = axis
            best_axis_dim = dim
    return best_axis    
    

def _ste(x):
    """
    Straight-through estimator.
    """
    return (x.round() - x).detach() + x

def _quantize_ste(x, n, axis=None):
    """
    Per-channel & symmetric quantization with STE.
    """
    quant_range = 2. ** n - 1.
    x_max = abs(x).max(dim=axis, keepdim=True)[0] if axis is not None else abs(x).max()
    x_scale = 2 * x_max / quant_range + 1e-6
    x_q = _ste(x / x_scale).clamp(-2**(n - 1), 2**(n - 1) - 1)
    return x_q, x_scale

def _quant_tensor(x, quant_level):
    """
    Quantize a tensor.
    """
    axis = compute_best_quant_axis(x)
    with torch.no_grad():
        x_q, x_scale = _quantize_ste(x, quant_level, axis)
        x_q = x_q.to(torch.int32)
        x_qr = x_q.to(x.dtype) * x_scale

    return x_q, x_qr


def quant_model_(model, args):
    model_list = [deepcopy(model)]
    if args.quant_model_bit == -1:
        return model_list, None
    else:
        cur_model = deepcopy(model)
        quant_ckt, cur_ckt = [cur_model.state_dict() for _ in range(2)]
        encoder_k_list = []
        for k,v in cur_ckt.items():
            if 'encoder' in k:
                encoder_k_list.append(k)
            else:
                quant_v, new_v = _quant_tensor(v, args.quant_model_bit)
                quant_v = quant_v[new_v!=0]
                quant_ckt[k] = quant_v
                cur_ckt[k] = new_v

        for encoder_k in encoder_k_list:
            del quant_ckt[encoder_k]
        cur_model.load_state_dict(cur_ckt)
        model_list.append(cur_model)
        
        return model_list, quant_ckt
    




def quant_model(model, args):
    model_list = [deepcopy(model)]
    if args.quant_model_bit == -1:
        return model_list, None
    else:
        cur_model = deepcopy(model)
        quant_ckt, cur_ckt = [cur_model.state_dict() for _ in range(2)]
        encoder_k_list = []
        for k,v in cur_ckt.items():
            if 'encoder' in k:
                encoder_k_list.append(k)
            else:
                quant_v, new_v = quant_tensor(v, args.quant_model_bit)
                quant_v["quant"] = quant_v['quant'][v!=0]
                quant_ckt[k] = quant_v

                cur_ckt[k] = new_v
        for encoder_k in encoder_k_list:
            del quant_ckt[encoder_k]
        cur_model.load_state_dict(cur_ckt)
        model_list.append(cur_model)
        
        return model_list, quant_ckt
    
def quant_tensor(t, bits=8):
    tmin_scale_list = []
    # quantize over the whole tensor, or along each dimenstion
    t_min, t_max = t.min(), t.max()
    scale = (t_max - t_min) / (2**bits-1)
    tmin_scale_list.append([t_min, scale])
    for axis in range(t.dim()):
        t_min, t_max = t.min(axis, keepdim=True)[0], t.max(axis, keepdim=True)[0]
        if t_min.nelement() / t.nelement() < 0.02:
            scale = (t_max - t_min) / (2**bits-1)
            # tmin_scale_list.append([t_min, scale]) 
            tmin_scale_list.append([t_min.to(torch.float16), scale.to(torch.float16)]) 
    # import pdb; pdb.set_trace; from IPython import embed; embed() 
     
    quant_t_list, new_t_list, err_t_list = [], [], []
    for t_min, scale in tmin_scale_list:
        t_min, scale = t_min.expand_as(t), scale.expand_as(t)
        quant_t = ((t - t_min) / (scale)).round().clamp(0, 2**bits-1)
        new_t = t_min + scale * quant_t
        err_t = (t - new_t).abs().mean()
        quant_t_list.append(quant_t)
        new_t_list.append(new_t)
        err_t_list.append(err_t)   

    # choose the best quantization 
    best_err_t = min(err_t_list)
    best_quant_idx = err_t_list.index(best_err_t)
    best_new_t = new_t_list[best_quant_idx]
    best_quant_t = quant_t_list[best_quant_idx].to(torch.uint8)
    best_tmin = tmin_scale_list[best_quant_idx][0]
    best_scale = tmin_scale_list[best_quant_idx][1]
    quant_t = {'quant': best_quant_t, 'min': best_tmin, 'scale': best_scale}

    return quant_t, best_new_t           