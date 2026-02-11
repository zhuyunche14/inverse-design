"""
Created on Sun Feb  1 10:21:15 2026

@author: Yunche Zhu
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import scipy.io as sio
import os
import numpy as np
import random 

class MetaSurfaceDataset(Dataset):
    def __init__(self, data_folder, target_size=64, augment=False):
        """
        Args:
            data_folder (str): data file path
            target_size (int): size of image  default64
            augment (bool): enhance  (train on，verification off)
        """
        self.files = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith('.mat')]
        self.target_size = target_size
        self.augment = augment
        
        print(f"data lodaing successful: in {data_folder}")
        print(f"   - size of samples: {len(self.files)}")
        print(f"   - target size: {target_size}x{target_size}")
        print(f"   - enhance mode: {'on (flip randomly)' if augment else 'off'}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        mat_path = self.files[idx]
        try:
        
            data = sio.loadmat(mat_path)
        except Exception as e:
            print(f"fault : {mat_path} | fault: {e}")
     
            return self.__getitem__((idx + 1) % len(self.files))


        mask_raw = data['mask']
        mask_tensor = torch.from_numpy(mask_raw).float().unsqueeze(0) 

        mask_64 = F.interpolate(mask_tensor.unsqueeze(0), 
                                size=(self.target_size, self.target_size), 
                                mode='bilinear', 
                                align_corners=False)
        mask_64 = mask_64.squeeze(0) 
        

        if self.augment:
            if random.random() > 0.5:
                mask_64 = torch.flip(mask_64, dims=[2])

            if random.random() > 0.5:
                mask_64 = torch.flip(mask_64, dims=[1])

        if 'T_val' in data:
            spec = data['T_val'].flatten()
        elif 'spectrum' in data:
            spec = data['spectrum'].flatten()
        else:

            spec = np.zeros(1000) 
            
        spectrum = torch.from_numpy(spec).float()

        return mask_64, spectrum