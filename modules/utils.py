#!/usr/bin/env python

'''
    Miscellaneous utilities that are extremely helpful but cannot be clubbed
    into other modules.
'''
import torch
import yaml
import argparse

# Scientific computing
import numpy as np
import scipy as sp
import scipy.linalg as lin

# Plotting
import cv2
import matplotlib.pyplot as plt

from modules import volutils
from modules import folding_utils as unfoldNd

def load_config(configfile):
    '''
        Wrapper to load configuration file
    '''
    with open(configfile, 'r') as cfg_file:
        config_dict = yaml.safe_load(cfg_file)
        
    # Convert dictionary to namespace
    config = argparse.Namespace(**config_dict)
    
    return config

def save_video(imstack, savename):
    '''
        Save a video sequence
    '''
    imstack = (255*np.clip(imstack, 0, 1)).astype(np.uint8)
    nimg, H, W, _ = imstack.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    video = cv2.VideoWriter(savename, fourcc, 10, (W, H))

    for idx in range(nimg):
        video.write(imstack[idx, ...])

def nextpow2(x):
    '''
        Return smallest number larger than x and a power of 2.
    '''
    logx = np.ceil(np.log2(x))
    return pow(2, logx)

def normalize(x, fullnormalize=False):
    '''
        Normalize input to lie between 0, 1.

        Inputs:
            x: Input signal
            fullnormalize: If True, normalize such that minimum is 0 and
                maximum is 1. Else, normalize such that maximum is 1 alone.

        Outputs:
            xnormalized: Normalized x.
    '''

    if x.sum() == 0:
        return x
    
    xmax = x.max()

    if fullnormalize:
        xmin = x.min()
    else:
        xmin = 0

    xnormalized = (x - xmin)/(xmax - xmin)

    return xnormalized

def asnr(x, xhat, compute_psnr=False):
    '''
        Compute affine SNR, which accounts for any scaling and shift between two
        signals

        Inputs:
            x: Ground truth signal(ndarray)
            xhat: Approximation of x

        Outputs:
            asnr_val: 20log10(||x||/||x - (a.xhat + b)||)
                where a, b are scalars that miminize MSE between x and xhat
    '''
    mxy = (x*xhat).mean()
    mxx = (xhat*xhat).mean()
    mx = xhat.mean()
    my = x.mean()
    

    a = (mxy - mx*my)/(mxx - mx*mx)
    b = my - a*mx

    if compute_psnr:
        return psnr(x, a*xhat + b)
    else:
        return rsnr(x, a*xhat + b)

def rsnr(x, xhat):
    '''
        Compute reconstruction SNR for a given signal and its reconstruction.

        Inputs:
            x: Ground truth signal (ndarray)
            xhat: Approximation of x

        Outputs:
            rsnr_val: RSNR = 20log10(||x||/||x-xhat||)
    '''
    xn = lin.norm(x.reshape(-1))
    en = lin.norm((x-xhat).reshape(-1)) + 1e-12
    rsnr_val = 20*np.log10(xn/en)

    return rsnr_val

def psnr(x, xhat):
    ''' Compute Peak Signal to Noise Ratio in dB

        Inputs:
            x: Ground truth signal
            xhat: Reconstructed signal

        Outputs:
            snrval: PSNR in dB
    '''
    err = x - xhat
    denom = torch.mean(torch.pow(err, 2)) + 1e-12

    snrval = 10*torch.log10(torch.max(x)/denom)

    return snrval

def embed(im, embedsize):
    '''
        Embed a small image centrally into a larger window.

        Inputs:
            im: Image to embed
            embedsize: 2-tuple of window size

        Outputs:
            imembed: Embedded image
    '''

    Hi, Wi = im.shape
    He, We = embedsize

    dH = (He - Hi)//2
    dW = (We - Wi)//2

    imembed = np.zeros((He, We), dtype=im.dtype)
    imembed[dH:Hi+dH, dW:Wi+dW] = im

    return imembed

def measure(x, noise_snr=40, tau=100):
    ''' Realistic sensor measurement with readout and photon noise

        Inputs:
            noise_snr: Readout noise in electron count
            tau: Integration time. Poisson noise is created for x*tau.
                (Default is 100)

        Outputs:
            x_meas: x with added noise
    '''
    x_meas = np.copy(x)

    noise = np.random.randn(x_meas.size).reshape(x_meas.shape)*noise_snr

    # First add photon noise, provided it is not infinity
    if tau != float('Inf'):
        x_meas = x_meas*tau

        x_meas[x > 0] = np.random.poisson(x_meas[x > 0])
        x_meas[x <= 0] = -np.random.poisson(-x_meas[x <= 0])

        x_meas = (x_meas + noise)/tau

    else:
        x_meas = x_meas + noise

    return x_meas
        
def build_montage(images):
    '''
        Build a montage out of images
    '''
    nimg, H, W = images.shape
    
    nrows = int(np.ceil(np.sqrt(nimg)))
    ncols = int(np.ceil(nimg/nrows))
    
    montage_im = np.zeros((H*nrows, W*ncols), dtype=np.float32)
    
    cnt = 0
    for r in range(nrows):
        for c in range(ncols):
            h1 = r*H
            h2 = (r+1)*H
            w1 = c*W
            w2 = (c+1)*W

            if cnt == nimg:
                break

            montage_im[h1:h2, w1:w2] = images[cnt, ...]
            cnt += 1
    
    return montage_im

def ims2rgb(im1, im2):
    '''
        Concatenate images into RGB
        
        Inputs:
            im1, im2: Two images to compare
    '''
    H, W = im1.shape
    
    imrgb = np.zeros((H, W, 3))
    imrgb[..., 0] = im1
    imrgb[..., 2] = im2

    return imrgb

def textfunc(im, txt):
    return cv2.putText(im, txt, (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (1, 1, 1),
                        2,
                        cv2.LINE_AA)

def boxify(im, topleft, boxsize, color=[1, 1, 1], width=2):
    '''
        Generate a box around a region.
    '''
    h, w = topleft
    dh, dw = boxsize
    
    im[h:h+dh+1, w:w+width, :] = color
    im[h:h+width, w:w+dh+width, :] = color
    im[h:h+dh+1, w+dw:w+dw+width, :] = color
    im[h+dh:h+dh+width, w:w+dh+width, :] = color

    return im

def moduloclip(cube, mulsize):
    '''
        Clip a cube to have multiples of mulsize
        
        Inputs:
            cube: (H, W, T) sized cube
            mulsize: (h, w) tuple having smallest stride size
            
        Outputs:
            cube_clipped: Clipped cube with size equal to multiples of h, w
    '''
    if len(mulsize) == 2:
        H, W = cube.shape[:2]
        
        H1 = mulsize[0]*(H // mulsize[0])
        W1 = mulsize[1]*(W // mulsize[1])
        
        cube_clipped = cube[:H1, :W1]
    else:
        H, W, T = cube.shape
        H1 = mulsize[0]*(H // mulsize[0])
        W1 = mulsize[1]*(W // mulsize[1])
        T1 = mulsize[2]*(T // mulsize[2])
        
        cube_clipped = cube[:H1, :W1, :T1]
    
    return cube_clipped

# Create folders and unfolders
def fold_clip(im, config):
    H, W = im.shape[:2]
    unfold = torch.nn.Unfold(kernel_size=config.ksize, stride=config.stride)
    fold = torch.nn.Fold(output_size=(H, W), kernel_size=config.ksize,
                         stride=config.stride)
    
    template = fold(unfold(torch.ones(1, 1, H, W))).squeeze()
    
    [rows, cols] = np.where(template.numpy() > 0)
    Hmax, Wmax = rows.max(), cols.max()
    
    return im[:Hmax+1, :Wmax+1, :]

def get_scheduler(scheduler_type, optimizer, args):
    '''
        Get a scheduler 
        
        Inputs:
            scheduler_type: 'none', 'step', 'exponential', 'cosine'
            optimizer: One of torch.optim optimizers
            args: Namspace containing arguments relevant to each optimizer
            
        Outputs:
            scheduler: A torch learning rate scheduler
    '''
    if scheduler_type == 'none':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=args.epochs)
    elif scheduler_type == 'step':
        # Compute gamma 
        gamma = pow(10, -1/(args.epochs/args.step_size))
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=args.step_size,
                                                    gamma=gamma)
    elif scheduler_type == 'exponential':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,
                                                        gamma=args.gamma)
        
    return scheduler

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_coords(imsize, ksize, coordstype, unfold):
    '''
        Generate coordinates for MINER training
        
        Inputs:
            imsize: (H, W) image size
            ksize: Kernel size
            coordstype: 'global' or 'local'
            unfold: Unfold operator
    '''
    ndim = len(imsize)
    if ndim == 2:
        H, W = imsize    
        
        #nchunks = int(H*W/(ksize**ndim))
        nchunks = unfold(torch.ones(1, 1, H, W)).shape[-1]
        
        # Create inputs
        if coordstype == 'global':
            X, Y = torch.meshgrid(torch.linspace(-1, 1, W),
                                  torch.linspace(-1, 1, H))
            coords = torch.cat((X[None, None, ...], Y[None, None, ...]), 0)
            coords_chunked = unfold(coords).permute(2, 1, 0)
        elif coordstype == 'local':
            Ysub, Xsub = torch.meshgrid(
                torch.linspace(-1, 1, ksize[0]),
                torch.linspace(-1, 1, ksize[1]))
            coords_sub = torch.cat((Xsub[None, None, ...],
                                    Ysub[None, None, ...]), 0)
            coords_chunked_sub = unfold(coords_sub).permute(2, 1, 0)
            coords_chunked = torch.repeat_interleave(coords_chunked_sub,
                                                     nchunks, 0)
        else:
            raise AttributeError('Coordinate type not understood')
    else:
        H, W, T = imsize    
        nchunks = int(H*W*T/(ksize**ndim))
        # Create inputs
        if coordstype == 'global':
            X, Y, Z = torch.meshgrid(torch.linspace(-1, 1, W),
                                     torch.linspace(-1, 1, H),
                                     torch.linspace(-1, 1, T))
            coords = torch.cat((X[None, None, ...],
                                Y[None, None, ...],
                                Z[None, None, ...]), 0)
            coords_chunked = unfold(coords).permute(2, 1, 0)
        elif coordstype == 'local':
            Xsub, Ysub, Zsub = torch.meshgrid(torch.linspace(-1, 1, ksize),
                                              torch.linspace(-1, 1, ksize),
                                              torch.linspace(-1, 1, ksize))
            coords_sub = torch.cat((Xsub[None, None, ...],
                                    Ysub[None, None, ...],
                                    Zsub[None, None, ...]), 0)
            coords_chunked_sub = unfold(coords_sub).permute(2, 1, 0)
            coords_chunked = torch.repeat_interleave(coords_chunked_sub,
                                                     nchunks, 0)
        else:
            raise AttributeError('Coordinate type not understood')
    
    return coords_chunked

def drawblocks_single(learn_indices, imsize, ksize):
    '''
        Draw blocks for a single image at a single scale
        
        Inputs:
            learn_indices: List of active indices
            imsize: Size of the image
            ksize: Kernel size
            
        Outputs:
            im_labels: Label image
    '''
    im_labels = torch.zeros((3, 1, imsize[0], imsize[1]))
    
    # Create folders and unfolders
    unfold = torch.nn.Unfold(kernel_size=ksize, stride=ksize)
    fold = torch.nn.Fold(output_size=imsize, kernel_size=ksize,
                        stride=ksize)
    
    im_labels_chunked = unfold(im_labels).reshape(3, ksize[0], ksize[1], -1)
    
    im_labels_chunked[2, 0, :, learn_indices] = 1
    im_labels_chunked[2, -1, :, learn_indices] = 1
    im_labels_chunked[2, :, 0, learn_indices] = 1
    im_labels_chunked[2, :, -1, learn_indices] = 1
    
    im_labels_chunked[1, 0, :, learn_indices] = 1
    im_labels_chunked[1, -1, :, learn_indices] = 1
    im_labels_chunked[1, :, 0, learn_indices] = 1
    im_labels_chunked[1, :, -1, learn_indices] = 1
    
    im_labels_chunked = im_labels_chunked.reshape(3, ksize[0]*ksize[1], -1)
    im_labels = fold(im_labels_chunked).permute(2, 3, 0, 1).squeeze()
        
    return im_labels

def drawcubes_single(learn_indices, cubesize, ksize, savename):
    '''
    Draw blocks for a single cube at a single scale
    
    Inputs:
        learn_indices: Indices for drawing the blocks
        cubesize: Size of the full cube
        ksize: Size of folding kernel
        savename: Name of the file for saving the plot
        
    Outputs:
        None
    '''
    unfold = unfoldNd.UnfoldNd(kernel_size=ksize, stride=ksize)
    
    # Get coordinates 
    coords_chunked = get_coords(cubesize,
                                ksize,
                                'global',
                                unfold)
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_aspect('auto')
    
    size = 2*np.array(ksize)/np.array(cubesize)
    size[...] = size[0]
    for idx in learn_indices:
        pos, _ = coords_chunked[idx, :, :].min(0)
        volutils.plotCubeAt(pos=pos[[0, 1, 2]], size=size, color='tab:blue',
                            edgecolor='tab:blue', alpha=0.05, ax=ax)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    
    ax.set_xticks([-1, 1])
    ax.set_yticks([-1, 1])
    ax.set_zticks([-1, 1])
    
    plt.savefig(savename)
    plt.close('all')

def drawblocks(learn_indices_list, imsize, ksize):
    '''
        Draw lines around blocks to showcase when they got terminated
        
        Inputs:
            learn_indices_list: List of tensor arrarys of indices
            imsize: Size of image            
            ksize: Size of kernel
            
        Outputs:
            im_labels: Labeled image
    '''
    im_labels = torch.zeros((1, 1, imsize[0], imsize[1]))
    nscales = len(learn_indices_list)
    H, W = imsize
    
    for idx in range(len(learn_indices_list)):
        learn_indices = learn_indices_list[idx]
        
        fold_tsize = (ksize*pow(2, nscales-idx-1),
                      ksize*pow(2, nscales-idx-1))
        
        # Create folders and unfolders
        unfold = torch.nn.Unfold(kernel_size=fold_tsize, stride=fold_tsize)
        fold = torch.nn.Fold(output_size=(H, W), kernel_size=fold_tsize,
                            stride=fold_tsize)
        
        im_labels_chunked = unfold(im_labels).reshape(1, fold_tsize[0],
                                                      fold_tsize[1], -1)
        im_labels_chunked[:, 0:2, :, learn_indices] = idx + 1
        im_labels_chunked[:, -3:-1, :, learn_indices] = idx + 1
        im_labels_chunked[:, :, 0:2, learn_indices] = idx + 1
        im_labels_chunked[:, :, -3:-1, learn_indices] = idx + 1
        
        im_labels_chunked = im_labels_chunked.reshape(1,
                                                      fold_tsize[0]*fold_tsize[1],
                                                      -1)
        im_labels = fold(im_labels_chunked)       
        
    return im_labels.squeeze().detach().cpu()
    
    
def sperical_interpolation(p1, p2, alpha=0.5):
    #p1 - N,2,f
    #p2 - N,2,f
    
    p1_norm = torch.sqrt((p1 * p1).sum(-1))
    p2_norm = torch.sqrt((p2 * p2).sum(-1))   
    
    p1 = p1 / torch.unsqueeze(p1_norm, -1)
    p2 = p2 / torch.unsqueeze(p2_norm, -1)
    
    cosine_angle = torch.arccos((p1*p2).sum(-1)) # N,2
    
    sin_omega = torch.sin(cosine_angle).unsqueeze(2) # N,2,1
    sin_alpha_omega = torch.sin(alpha*cosine_angle).unsqueeze(2) # N,2,1
    sin_alpha_omega_ = torch.sin((1-alpha)*cosine_angle).unsqueeze(2) # N,2,1
    
    p = ( sin_alpha_omega_*p1 + sin_alpha_omega*p2 ) / sin_omega
    
    return p
    
    
    
