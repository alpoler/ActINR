#%%
#!/usr/bin/env python

import os
import sys
import importlib
import time
import argparse
import random 

import numpy as np
from scipy import io
from torch.distributed import init_process_group, destroy_process_group
import cv2
import torch

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
plt.gray()

sys.path.append('modules')

import torch.multiprocessing as mp

import utils
import miner
import kilonerf_time


utils = importlib.reload(utils)
miner = importlib.reload(miner)
kilonerf = importlib.reload(kilonerf_time)


def ddp_setup(rank,world_size):
    os.environ["MASTER_ADDR"]="localhost"
    os.environ["MASTER_PORT"]="12356"
    init_process_group(backend="nccl",rank=rank,world_size=world_size)

def num_params(model):
    n=0
    nb=0
    for key in model.keys():
        n = n + model[key].numel()
        if 'bias' in key:
            #print(key)
            nb = nb + model[key].numel()
    return n, nb 

#if __name__ == '__main__':
def main(rank,config,world_size,args):

    configname = config 
    target_mse = 0
    

   
    save_img = True
    save_models = True
    
    
    # Read configuration
    config = utils.load_config(configname)
    # config.start = args.start
    #config.nfeat = args.nfeat
    # config.partition_size = args.partition
    # config.n_frames = args.nframes
    # config.exp_name = "bosph_gop{}".format(config.end)
    if not config.inference:
        #torch.cuda.set_device(rank)
        ddp_setup(rank,world_size)
    

    

        
    optim_func = getattr(kilonerf, "multibias")
    

    if config.inference:
        config.epochs = 1

    print(config)

    save_path = os.path.join(config.save_path, config.dataset_name, config.model_type+'_'+str(config.nfeat).zfill(2))
    os.makedirs(os.path.join(save_path),  exist_ok=True)
 

    # Load images
    if config.pretrained_path != "None":
        pretrained_models=[]
        pretrained_models.append(torch.load(config.pretrained_path,map_location=torch.device("cuda")))
    else:
        pretrained_models=None
    tic = time.time()
    if not config.inference:
        
        best_img, info, models = optim_func(rank,target_mse, config,world_size=world_size,pretrained_models=pretrained_models)
        
    if config.inference:
        config.epochs=1
        best_img, info, models = optim_func(rank,target_mse, config,pretrained_models=pretrained_models)
        

    if rank==0:
        total_time = time.time() - tic
        
        psnr_array_test = info['psnr_array_test']
        psnr_array_train = info['psnr_array_train']
        nparams_array = info['nparams_array']

        info['total_time'] = total_time
        print(psnr_array_test)
        print(psnr_array_train)
        print(nparams_array)
        
        os.makedirs('%s/%s'%(save_path, config.exp_name), exist_ok=True)
        if not config.inference:
            savename = '%s_%s_%s_ksize%dx%d_nfeat%d_%s'%(
                config.nonlin, config.model_type, config.optim_type,
                config.ksize[0], config.ksize[1], config.nfeat, config.dataset_name
            )
        else:
            savename = '%s_%s_%s_ksize%dx%d_nfeat%d_nframes%d_inf'%(
                config.nonlin, config.model_type, config.optim_type,
                config.ksize[0], config.ksize[1], config.nfeat, config.n_frames
            )
        
        io.savemat('%s/%s/%s.mat'%(save_path, config.exp_name, savename), info)
        
        
        os.makedirs('videos/%s'%config.exp_name, exist_ok=True)
        utils.save_video(best_img, 'videos/%s/%s.avi'%(
                config.exp_name, savename))
                
        if save_img:
            for i in range(best_img.shape[0]):
                if not config.inference:
                    image_name = os.path.join(save_path, config.exp_name, str(i).zfill(5)+'_.png')
                else:
                    image_name = os.path.join(save_path, config.exp_name, str(i).zfill(5)+'_inf.png')
            
                cv2.imwrite(image_name, np.uint8(255*best_img[i, ...].clip(0,1)))
        
        pth_kb_array = []
        if save_models:
            state_dict_ = models.state_dict()
            model_i_path = os.path.join(save_path, config.exp_name,config.dataset_name+ '.pth')
            torch.save(state_dict_, model_i_path)
            pth_kb_array.append(os.path.getsize(model_i_path))
            if i==0:
                print(num_params(state_dict_))

        
        if not config.inference:            
            pth_kb_array = np.array(pth_kb_array)
            
        #if pth_kb_array.ndim>1:
        #    pth_kb_array = pth_kb_array.sum(0)
        

        print(pth_kb_array)
    
    if not config.inference:
        destroy_process_group()


parser = argparse.ArgumentParser()
parser.add_argument('--model_type', type=str,
                    default=None, help='choose kilonerf or miner')
parser.add_argument('--optim_type', type=str,
                    default='multibias', help='choose multibias or single_inr')         
parser.add_argument('--config', type=str,
                    default='configs/wire_20x32_anil.yaml', help='choose path to the config file from configs folder')         
parser.add_argument('--frame_list', type=str,
                    default=None, help='choose kilonerf or miner')
parser.add_argument('--save_path', type=str,
                    default='/home/akayabasi/bias_interpolation/results', help='choose save path')
parser.add_argument('--nframes', type=int,
                    default=50, help='in range of 3 to 16')
parser.add_argument('--partition', type=int,
                    default=50, help='in range of 3 to 16')
parser.add_argument('--start', type=int,
                    default=0, help='in range of 3 to 16')
parser.add_argument('--end', type=int,
                    default=5, help='in range of 3 to 16')
parser.add_argument('--nfeat', type=int,
                    default=None, help='in range of 8 to 32')
parser.add_argument('--save_model', type=int,
                    default=1, help='0 or 1')
parser.add_argument('--share_weights', type=int,
                    default=1, help='0 or 1')
parser.add_argument('--inference', type=int,
                    default=0, help='0 or 1')
parser.add_argument('--split_img', type=int,
                    default=0, help='0 or 1')          
                              
#save_model=True, inference=False
args = parser.parse_args()


if __name__ == '__main__':
    inference = False
    if not inference:
        config = "configs/wire_20x32_anil.yaml"
        world_size = torch.cuda.device_count()
        print(world_size)
        mp.spawn(main,args=(config,world_size,args),nprocs=world_size)
    else:
        main(rank=0,config=args.config,world_size=2,args=args)
