#%%
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.nn.functional import interpolate
from torch.utils.data import Sampler
from typing import Iterator, List, Optional, Union
from torch.utils.data import DistributedSampler
from operator import itemgetter
from math import ceil
from matplotlib import pyplot as plt

class VideoDataset(Dataset):

    def __init__(self, path, freq, no_data, train, part,resize=(480,960),unfold=None,start=0,end=100,config=None):
        image_list = open(path,"r")
        image_list = [i.strip("\n") for i in image_list]
        image_list.sort()
        self.start = start 
        self.end = end
        self.image_list = image_list[start:end]
        self.unfold = unfold
        #img= cv2.imread(image_list[0])
        H,W=1080,1920
        h_max,w_max= fold_clip(config=config,H=H,W=W)
        self.h_max= h_max
        self.w_max = w_max
        # Find out number of chunks
        weighing = torch.ones(1,1,h_max+1,w_max+1)
        nchunks = unfold(weighing).shape[-1]    
        # nchunks=1
        self.nchunks = nchunks  
        
        self.construct_t(no_data,freq,train,nchunks,part)
        if resize != -1:
            self.resize = (resize[0],resize[1])
        else:
            self.resize=-1



    
    def construct_t(self, no_data, freq, train, no_chunks, part):

        full_idx = list(range(0, no_data))
        if part!=1:
            step = part - 1
        else:
            step = 1
        chunks = []
        chunk_indices = []
        times = []
        no_tiles = 0
        last_element = ((no_data-1)//freq)*freq
        first_elements = []
        for idx, i in enumerate(range(0, len(full_idx) - 1 if part!=1 else last_element+1, step)):
            chunk = full_idx[i:i+part]
            first_elements.append(chunk[0])
            chunks.extend(chunk)
            chunk_indices.extend([idx] * len(chunk))
            #time = [x / (len(chunk) - 1) for x in range(len(chunk))] # local coordinates
            time = [x / (last_element) for x in chunk] # global coordinates
            times.extend(time)
            no_tiles+=1
        last_element = (chunks[-1]//freq)*freq
        self.first_elements = first_elements
    
        if train or part==1:
            series = [el for el in chunks if (el%freq)==0]
            model_idx = [idx for (el,idx) in zip(chunks,chunk_indices) if (el%freq)==0]
            times = [t for (el,t) in zip(chunks,times) if (el%freq)==0]
        else:
            series = [el for el in chunks if (el%freq)!=0 and el<(max(chunk)//freq)*freq]
            model_idx = [idx for (el,idx) in zip(chunks,chunk_indices) if (el%freq)!=0 and el<(max(chunk)//freq)*freq]
            times = [t for (el,t) in zip(chunks,times) if (el%freq)!=0 and el<(max(chunk)//freq)*freq]
        self.image_list = [self.image_list[j] for j in series]
        self.times= torch.repeat_interleave(torch.tensor(times)[None,:,None],repeats=no_chunks,dim=0)
        self.model_idx = torch.tensor(model_idx)
        self.no_tiles = no_tiles
        self.grouping = model_idx
      


        #no_data = self.end - self.start - 1 
        # data_idx = list(range(0,no_data,freq))
        # last_no = data_idx[-1]
        # if not train:
        #     full_idx = list(range(0,last_no+1,1))
        #     data_idx = [inst_idx for inst_idx in full_idx if inst_idx not in data_idx]
        # self.image_list = [self.image_list[j] for j in data_idx]
        # self.data_idx = torch.tensor(data_idx,dtype=torch.float)
        # y_ = ceil(last_no/ part)
        # x_ = y_ // 2        
        # centers = torch.tensor([x_ + y_*center_no for center_no in range(part)],dtype=torch.float)
        # centers = centers.unsqueeze(-1)
        # cluster_to_idx_dist = (torch.abs(centers - self.data_idx.unsqueeze(0)) - (torch.arange(part)*1e-3).unsqueeze(-1))
        # self.model_idx = torch.argmin(cluster_to_idx_dist,dim=0)
        # if train:
        #     self.grouping = torch.cat([torch.argwhere(self.model_idx==i) for i in range(part)],dim=1)
        # t = (self.data_idx )/(last_no+1)
        # t = t[None,:,None]
        # self.t = torch.repeat_interleave(t,repeats=no_chunks,dim=0)
        # self.y_ = y_
        # self.freq = freq
        # self.part = part


    def edge_init(self,model):
        edge_feat = 30
        for block_no,i in enumerate(range(len(self.first_elements))):
            im = cv2.imread(self.image_list[self.first_elements[i]//2])
            #im = im[60:1020,:,:]
            unfold_im = self.unfold(torch.tensor(im).permute(2,0,1)[None,...].float())
            unfold_im = unfold_im.view(3,self.h_max+1,self.w_max+1,-1)
            edge = cv2.Canny(im,150,200)
            unfold_edge = self.unfold(torch.tensor(edge[None,None,...]).float())
            unfold_edge = unfold_edge.to(torch.uint8)
            unfold_edge = unfold_edge.view(self.h_max+1,self.w_max+1,-1)

            for j in range(unfold_edge.shape[-1]):
                cv2.imwrite("edge_map.png",np.array(unfold_edge[...,j]))
                cv2.imwrite("image_patch.png",np.array(unfold_im[...,j].permute(1,2,0)))
                [edge_y_idx,edge_x_idx] = torch.where(unfold_edge[...,j]==255)
                if (len(edge_y_idx) >= edge_feat):
                    no_feat = edge_feat
                    total = len(edge_y_idx)
                elif (len(edge_y_idx) < edge_feat) and (len(edge_y_idx)>0):
                    no_feat= len(edge_y_idx)
                    total = len(edge_y_idx)
                else:
                    continue
                idx_select = np.random.randint(0, total, no_feat)
                bias_x = -(edge_x_idx[idx_select] - (self.h_max+1)/2)/((self.h_max+1)/2)
                bias_y = -(edge_y_idx[idx_select] - (self.w_max+1)/2)/((self.w_max+1)/2)
                bias_x = np.array(bias_x)
                bias_y = np.array(bias_y)
                unfold_img = np.array(unfold_im[...,j].permute(1,2,0))
                gradx = cv2.Sobel(unfold_img[...,1],cv2.CV_32F,1,0,ksize=3)
                grady = cv2.Sobel(unfold_img[...,1],cv2.CV_32F,0,1,ksize=3)
                grad_angle = np.arctan2(gradx, grady)*np.array(unfold_edge[...,j])
                grad_angle = grad_angle[edge_y_idx, edge_x_idx]
                #print(grad_angle)
                #print(idx_select)
                theta = grad_angle[idx_select]*0.0 if len(idx_select) >1 else grad_angle
                scale_x = np.ones(no_feat)
                scale_y = np.ones(no_feat)
                bias_x = (np.cos(theta)*bias_x + np.sin(theta)*bias_y)/scale_x
                bias_y = (-np.sin(theta)*bias_x + np.cos(theta)*bias_y)/scale_y

                tct = torch.cos(torch.tensor(theta))
                tst = torch.sin(torch.tensor(theta))

                with torch.no_grad():
                    model.module.net[0].linear.bias[block_no,j,0,:no_feat] = torch.tensor(bias_x)
                    model.module.net[0].orth_scale.bias[block_no,j,0,:no_feat] = torch.tensor(bias_y)

                    #model.module.net[0].linear.bias[block_no,j,0,:no_feat].requires_grad = False
                    #model.module.net[0].orth_scale.bias[block_no,j,0,:no_feat].requires_grad = False


                    model.module.net[0].linear.weight[block_no, j, 0, :no_feat] = tct*scale_x
                    model.module.net[0].linear.weight[block_no, j, 1, :no_feat] = -tst*scale_y

                    model.module.net[0].orth_scale.weight[block_no, j, 0, :no_feat] = tst*scale_x
                    model.module.net[0].orth_scale.weight[block_no, j, 1, :no_feat] = tct*scale_y


    def img_load(self,idx):
        im = cv2.imread(self.image_list[idx])
        im = im.astype(np.float32)/255
        im = im[:self.h_max+1,:self.w_max+1,:]
        # im = np.load(self.image_list[idx])[None,:,None]
        # im = im / im.max()
        #grad_map = torch.tensor(self.extract_gradient_map(im))
        im = torch.tensor(im).permute(2,0,1)
        return im #,grad_map
    
    def extract_gradient_map(self,im):
        gradx = cv2.Sobel(im[...,1],cv2.CV_32F,1,0,ksize=3)
        grady = cv2.Sobel(im[...,1],cv2.CV_32F,0,1,ksize=3)
        grad_map = np.sqrt(gradx**2 + grady**2)
        return (1+0.0*grad_map)
    
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
        #data_idx = self.data_idx[idx]
        tensor_image = self.img_load(idx) #, grad_map
        t = self.times[:,idx,:]
        model_idx = self.model_idx[idx]
        if self.resize != -1:
            tensor_image = interpolate(tensor_image.unsqueeze(0),size=self.resize,mode='area').squeeze(0)
        sample = {'img': tensor_image, 't': t, 'model_idx': model_idx} # , "grad_map": grad_map
        return sample
    

class BalancedSampler(Sampler):
    def __init__(self,dataset):
        self.y = dataset.y_
        self.freq = dataset.freq
        self.part = dataset.part
        self.len = len(dataset)

    def __iter__(self):
        indices_t = torch.stack([torch.randperm(self.y//2)+(self.y//2)*j for j in range(self.part)]).transpose(1,0).flatten()
        return iter(indices_t)
    
    def __len__(self):
        return self.len

class DatasetFromSampler(Dataset):
    """Dataset to create indexes from `Sampler`.

    Args:
        sampler: PyTorch sampler
    """

    def __init__(self, sampler: Sampler):
        """Initialisation for DatasetFromSampler."""
        self.sampler = sampler
        self.sampler_list = None

    def __getitem__(self, index: int):
        """Gets element of the dataset.

        Args:
            index: index of the element in the dataset

        Returns:
            Single element by index
        """
        if self.sampler_list is None:
            self.sampler_list = list(self.sampler)
        return self.sampler_list[index]

    def __len__(self) -> int:
        """
        Returns:
            int: length of the dataset
        """
        return len(self.sampler)

class DistributedSamplerWrapper(DistributedSampler):
    """
    Wrapper over `Sampler` for distributed training.
    Allows you to use any sampler in distributed mode.

    It is especially useful in conjunction with
    `torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSamplerWrapper instance as a DataLoader
    sampler, and load a subset of subsampled data of the original dataset
    that is exclusive to it.

    .. note::
        Sampler is assumed to be of constant size.
    """

    def __init__(
        self,
        sampler,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
    ):
        """

        Args:
            sampler: Sampler used for subsampling
            num_replicas (int, optional): Number of processes participating in
                distributed training
            rank (int, optional): Rank of the current process
                within ``num_replicas``
            shuffle (bool, optional): If true (default),
                sampler will shuffle the indices
        """
        super(DistributedSamplerWrapper, self).__init__(
            DatasetFromSampler(sampler),
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
        )
        self.sampler = sampler

    def __iter__(self) -> Iterator[int]:
        """Iterate over sampler.

        Returns:
            python iterator
        """
        self.dataset = DatasetFromSampler(self.sampler)
        indexes_of_indexes = super().__iter__()
        subsampler_indexes = self.dataset
        return iter(itemgetter(*indexes_of_indexes)(subsampler_indexes))
    
def fold_clip(config,H,W):
    
    unfold = torch.nn.Unfold(kernel_size=config.ksize, stride=config.stride)
    fold = torch.nn.Fold(output_size=(H, W), kernel_size=config.ksize,
                         stride=config.stride)
    
    template = fold(unfold(torch.ones(1, 1, H, W))).squeeze()
    
    [rows, cols] = np.where(template.numpy() > 0)
    Hmax, Wmax = rows.max(), cols.max()
    
    return Hmax,Wmax


# %%
