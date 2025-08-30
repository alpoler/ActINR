#!/usr/bin/env python

import importlib

import utils
import siren_hyperinr
import losses
import volutils
import wire_time
import wire_time_qat
import wire_time_rt
import wire_time_qat_finetune
import wire_time_final
import wire_time_nirvana
import wire
import gaussian
from torch.nn.parallel import DistributedDataParallel as DDP


utils = importlib.reload(utils)
siren_hyperinr = importlib.reload(siren_hyperinr)
losses = importlib.reload(losses)
volutils = importlib.reload(volutils)
wire = importlib.reload(wire)
wire_time = importlib.reload(wire_time)
wire_time_qat = importlib.reload(wire_time_qat)
wire_time_qat_finetune = importlib.reload(wire_time_qat_finetune)
wire_time_rt = importlib.reload(wire_time_rt)
wire_time_final = importlib.reload(wire_time_final)
wire_time_nirvana = importlib.reload(wire_time_nirvana)
wire_lr = importlib.reload(wire)
gaussian = importlib.reload(gaussian)

def get_model(config, nchunks,rank):
    if  config.nonlin == 'sine':
        model = siren_hyperinr.AdaptiveMultiSiren(
                in_features=config.in_features,
                out_features=config.out_features, 
                n_channels=nchunks,
                hidden_features=config.nfeat, 
                hidden_layers=config.nlayers,
                outermost_linear=True,
                share_weights=config.share_weights,
                first_omega_0=config.omega_0,
                hidden_omega_0=config.omega_0,
                n_img = config.n_frames,
                pos_encode=False
            ).cuda()
    elif config.nonlin == 'wire':
        hidden_omega_0 = config.scale*2.0
        #hidden_omega_0 = 1.0
        
        model = wire.AdaptiveMultiWIRE(
            in_features=config.in_features,
            out_features=config.out_features, 
            n_channels=nchunks,
            hidden_features=config.nfeat, 
            hidden_layers=config.nlayers,
            outermost_linear=True,
            share_weights=config.share_weights,
            first_omega_0=config.omega_0,
            hidden_omega_0=config.omega_0,
            scale=config.scale,
            n_img=config.n_frames,
            pos_encode=config.pos_encode,
            const=1.0
        ).cuda()
    elif config.nonlin == 'wire_time':
        hidden_omega_0 = config.scale*2.0
        #hidden_omega_0 = 1.0
        
        model = wire_time.AdaptiveMultiWIRE(
            in_features=config.in_features,
            out_features=config.out_features, 
            n_channels=nchunks,
            hidden_features=config.nfeat, 
            hidden_layers=config.nlayers,
            outermost_linear=True,
            mode=config.mode,
            share_weights=config.share_weights,
            first_omega_0=config.omega_0,
            hidden_omega_0=config.omega_0,
            scale=config.scale,
            n_img=config.n_frames,
            pos_encode=config.pos_encode,
            const=1.0,
            rank=rank,
            config=config
        )   
    elif config.nonlin == "wire_qat":
            hidden_omega_0 = config.scale*2.0
            #hidden_omega_0 = 1.0
            model = wire_time_qat.AdaptiveMultiWIRE(
            in_features=config.in_features,
            out_features=config.out_features, 
            n_channels=nchunks,
            hidden_features=config.nfeat, 
            hidden_layers=config.nlayers,
            outermost_linear=True,
            mode=config.mode,
            share_weights=config.share_weights,
            first_omega_0=config.omega_0,
            hidden_omega_0=config.omega_0,
            scale=config.scale,
            n_img=config.n_frames,
            pos_encode=config.pos_encode,
            const=1.0,
            rank=rank,
            config=config
        )

    elif config.nonlin == "wire_qat_finetune":
            hidden_omega_0 = config.scale*2.0
            hidden_omega_0 = 1.0
            model = wire_time_qat_finetune.AdaptiveMultiWIRE(
            in_features=config.in_features,
            out_features=config.out_features, 
            n_channels=nchunks,
            hidden_features=config.nfeat, 
            hidden_layers=config.nlayers,
            outermost_linear=True,
            mode=config.mode,
            share_weights=config.share_weights,
            first_omega_0=config.omega_0,
            hidden_omega_0=config.omega_0,
            scale=config.scale,
            n_img=config.n_frames,
            pos_encode=config.pos_encode,
            const=1.0,
            rank=rank,
            config=config
        )
    elif config.nonlin == "wire_time_rt":
            hidden_omega_0 = config.scale*2.0
            #hidden_omega_0 = config
            model = wire_time_rt.AdaptiveMultiWIRE(
            in_features=config.in_features,
            out_features=config.out_features, 
            n_channels=nchunks,
            hidden_features=config.nfeat, 
            hidden_layers=config.nlayers,
            outermost_linear=True,
            mode=config.mode,
            share_weights=config.share_weights,
            first_omega_0=config.omega_0,
            hidden_omega_0=config.omega_0,
            scale=config.scale,
            n_img=config.n_frames,
            pos_encode=config.pos_encode,
            const=1.0,
            rank=rank,
            config=config
        )
    elif config.nonlin == "wire_time_final":
            #hidden_omega_0 = config.scale*2.0
            #hidden_omega_0 = config
            model = wire_time_final.AdaptiveMultiWIRE(
            in_features=config.in_features,
            out_features=config.out_features, 
            n_channels=nchunks,
            hidden_features=config.nfeat, 
            hidden_layers=config.nlayers,
            outermost_linear=True,
            mode=config.mode,
            share_weights=config.share_weights,
            first_omega_0=config.omega_0,
            hidden_omega_0=config.omega_0,
            scale=config.scale,
            n_img=config.n_frames,
            pos_encode=config.pos_encode,
            const=1.0,
            rank=rank,
            config=config
        ) 
    elif config.nonlin == "wire_time_nirvana":
        hidden_omega_0 = config.scale*2.0
        #hidden_omega_0 = config
        model = wire_time_nirvana.AdaptiveMultiWIRE(
        in_features=config.in_features,
        out_features=config.out_features, 
        n_channels=nchunks,
        hidden_features=config.nfeat, 
        hidden_layers=config.nlayers,
        outermost_linear=True,
        mode=config.mode,
        share_weights=config.share_weights,
        first_omega_0=config.omega_0,
        hidden_omega_0=config.omega_0,
        scale=config.scale,
        n_img=config.n_frames,
        pos_encode=config.pos_encode,
        const=1.0,
        rank=rank,
        config=config
    )                
                
    elif config.nonlin == 'wire_lr':
        hidden_omega_0 = config.scale*2.0
        hidden_omega_0 = 1.0
        
        model = wire_lr.AdaptiveMultiWIRELR(
            in_features=config.in_features,
            out_features=config.out_features, 
            n_channels=nchunks,
            hidden_features=config.nfeat, 
            hidden_layers=config.nlayers,
            outermost_linear=True,
            share_weights=config.share_weights,
            first_omega_0=config.omega_0,
            hidden_omega_0=hidden_omega_0,
            scale=config.scale,
            const=1.0,
            rank=config.rank
        ).cuda()

    elif config.nonlin == 'gauss':
        model = gaussian.AdaptiveMultiGauss(
            in_features=config.in_features,
            out_features=config.out_features, 
            n_channels=nchunks,
            hidden_features=config.nfeat, 
            hidden_layers=config.nlayers,
            outermost_linear=True,
            share_weights=config.share_weights,
            first_omega_0=config.omega_0,
            hidden_omega_0=config.omega_0,
            scale=config.scale,
            const=1.0
        ).cuda()
    model = model.cuda(rank)
    if config.nonlin=="wire_time_rt":
        for k in model.prob_models.keys():
            if "layer1" in k:
                model.prob_models[k]["linear"].to(rank)
                model.prob_models[k]["orthogonal"].to(rank)      
            else:
                try:
                    model.prob_models[k]["linear"][0].to(rank)
                    model.prob_models[k]["linear"][1].to(rank)
                    model.prob_models[k]["orthogonal"][0].to(rank)    
                    model.prob_models[k]["orthogonal"][1].to(rank) 
                except:
                    model.prob_models[k][0].to(rank)
                    model.prob_models[k][1].to(rank) 
            
    if not config.inference:
        model = DDP(model,device_ids=[rank],find_unused_parameters=False)

    return model