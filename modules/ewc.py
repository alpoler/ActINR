
import torch
from copy import deepcopy
from customlinear import custom_linear

import gc
def ewc(model,old_dataset,coords,rank,learn_indices,config,nchunks,fold):
    precision_matrices = {}
    with torch.no_grad():
        for mod_name, cur_module in model.module.named_modules():
            if type(cur_module) in [custom_linear]:
                
                if cur_module.complex:
                    weight = torch.zeros_like(cur_module.weight.data[...,0])
                else:
                    weight = torch.zeros_like(cur_module.weight.data)
                precision_matrices[mod_name] = weight
    model.eval()
    for sample in old_dataset:
        model.zero_grad()
        t_coords = sample["t"].cuda(rank).permute(1,0,2)
        imten = sample["img"].cuda(rank)
        model_idx = sample["model_idx"].cuda(rank)
        t_coords = (t_coords,model_idx)
        im_out = model(coords,learn_indices,t_coords)
        im_out = im_out.permute(0, 3, 2, 1).reshape(1,config.out_features,config.ksize[0],config.ksize[1],-1)
        im_out = im_out.reshape(-1,config.out_features*config.ksize[0]*config.ksize[1],nchunks)
        im_estim = fold(im_out).reshape(-1, config.out_features, config.w, config.h)
        loss= ((im_estim-imten).pow(2)).mean()
        loss.backward()
        for n, cur_module in model.module.named_modules():
            if type(cur_module) in [custom_linear]:
                
                weight = cur_module.weight
                if cur_module.complex:
                    w_angle = torch.angle(torch.view_as_complex(weight))
                    cos_th = torch.cos(w_angle)
                    sin_th = torch.sin(w_angle)
                    dz_dx_dz_dy = torch.stack([cos_th,sin_th],dim=-1)
                    w_grad = ((weight.grad.data)*(dz_dx_dz_dy)).sum(dim=-1)
                else:                    
                    w_grad = weight.grad.data
                precision_matrices[n].data += w_grad.abs() / len(old_dataset)
    precision_matrices = {n: torch.log10(p) for n, p in precision_matrices.items()}
    global_min = min(p.min().item() for p in precision_matrices.values())
    global_max = max(p.max().item() for p in precision_matrices.values())
    normalized_precision_matrices = {
    n: (p - global_min) / (global_max - global_min)
    for n, p in precision_matrices.items()
}
    
    return normalized_precision_matrices



def hessian_trace(model,old_dataset,coords,rank,learn_indices,config,nchunks,fold,num_samples=10):
    """Compute the trace of the Hessian using Hutchinson’s estimator."""
    num_batches = 0
    windowwise_traces = {}
    sample = next(iter(old_dataset))

    t_coords = sample["t"].cuda(rank).permute(1,0,2)
    imten = sample["img"].cuda(rank)
    model_idx = sample["model_idx"].cuda(rank)
    t_coords = (t_coords,model_idx)
    windowwise_traces = {}

    model.zero_grad()
    
    # --- Recompute the forward pass each iteration so you get a fresh graph ---
    im_out = model(coords, learn_indices, t_coords)
    # Reshape/rearrange as needed (update these lines to match your data):
    im_out = im_out.permute(0, 3, 2, 1).reshape(1, config.out_features, config.ksize[0], config.ksize[1], -1)
    im_out = im_out.reshape(-1, config.out_features * config.ksize[0] * config.ksize[1], nchunks)
    im_estim = fold(im_out).reshape(-1, config.out_features, config.w, config.h)
    
    loss = ((im_estim - imten).pow(2)).mean()
    loss.backward(create_graph=True)
    # Compute first derivatives with create_graph=True and retain_graph=True
    params =[]
    grads = []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        params.append(param)
        grads.append(0. if param.grad is None else param.grad + 0.)
   
    gradients = torch.autograd.grad(
        loss, model.parameters(), create_graph=True,
        allow_unused=True, retain_graph=True)
        




    for iter_idx in range(num_samples):









        # For each parameter compute the Hessian-vector product (HVP)
        for (name, param), grad in zip(model.named_parameters(), gradients):
            if grad is None:
                print(f"Skipping {name} (No gradient)")
                continue  # Skip if parameter is not involved in loss
            
            if grad.ndim >= 2:
                no_windows = grad.shape[1]  # Assuming window dimension is dim=1
                
                # Initialize trace accumulator for this parameter if needed
                if name not in windowwise_traces:
                    windowwise_traces[name] = torch.zeros(no_windows, device=param.device)
                
                # Generate a Rademacher random vector with same shape as grad:
                random_vec = torch.randint_like(grad, high=2, dtype=torch.float32) * 2 - 1
                
                # Compute Hessian-vector product.
                # Use retain_graph=True so that the saved intermediates from the forward pass are still available.
                hvp = torch.autograd.grad(
                    grad, param, grad_outputs=random_vec,
                    retain_graph=False, only_inputs=True)[0]
                
                if hvp is not None:
                    # Sum over all dimensions except the window dimension.
                    # Adjust the view if necessary so that dim0 corresponds to windows.
                    trace_est = (random_vec * hvp).view(no_windows, -1).sum(dim=1)
                    windowwise_traces[name] += trace_est
        
        # Explicitly free tensors holding the graph so that it’s cleared before the next iteration.
        del loss, gradients, hvp, random_vec, im_out, im_estim
        gc.collect()
        torch.cuda.empty_cache()

    # Average the accumulated Hessian trace estimates over iterations
    for name in windowwise_traces:
        windowwise_traces[name] /= num_samples
        # Replace any negative values with 0
        windowwise_traces[name] = torch.where(
            windowwise_traces[name] <= 0,
            torch.tensor(0., device=windowwise_traces[name].device),
            windowwise_traces[name])
        
    return windowwise_traces

