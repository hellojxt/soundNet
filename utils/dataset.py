from torchvision import datasets, transforms
import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np
from utils import helper
import random

def rotate(x,axis = [0,2]):
    k = random.randint(0,3)
    for i in x:
        if x[i].dim() < 2:
            continue
        x[i] = torch.rot90(x[i],k,axis)
    return x

class HourGlassDataset(data.Dataset):
    def __init__(self, npz_file):
        data = np.load(npz_file)
        self.vox = data['vox']
        self.target = data['target']
        self.freq = data['freq']
        
    def __len__(self):
        return len(self.vox)

    def __getitem__(self, idx):

        vox = self.vox[idx]

        target = self.target[idx].reshape(16,16,16, -1)

        freq = self.freq[idx]
        freq = freq.reshape(8,-1).mean(-1)

        item = {
            'vox':      torch.as_tensor(vox, dtype=torch.float32),
                        # x_size * y_size * z_size
            'target':   torch.as_tensor(target, dtype=torch.float32)/255,
                         # x_size * y_size * z_size * 3 * res
            'freq':     torch.as_tensor(freq, dtype=torch.float32)
        }

        item = rotate(item)

        return item