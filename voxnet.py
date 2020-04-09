import torch.nn as nn
from model.net import *
import torch.utils.data as data
import numpy as np
from utils import helper
import random
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
import copy
import os
class voxnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = frequency_encoder()
        self.end_line = nn.Linear(256,40)

    def forward(self,x):
        x = self.body(x)
        x = x.view(x.size(0),-1)
        return self.end_line(x)

class genvoxnetDataset(data.Dataset):
    def __init__(self, npz_file):
        data = np.load(npz_file)
        self.vox = data['vox']
        self.target = data['target']
        self.freq = data['freq']
        self.filename = data['filename']

    def trans(self,filename):
        self.label = np.zeros(len(self.vox))
        self.type = []
        for i in range(len(self.vox)):
            name = self.filename[i].split('\\')[-1][:-9]
            if not name in self.type:
                self.type.append(name)
        for i in range(len(self.vox)):
            name = self.filename[i].split('\\')[-1][:-9]
            self.label[i] = self.type.index(name)

        np.savez_compressed(filename,
            vox=self.vox,
            target=self.target,
            freq=self.freq,
            filename=self.filename,
            label=self.label
        )
        

def gen_full_dataset():
    genvoxnetDataset('dataset\\test\\all.npz').trans('dataset\\test\\all_full.npz')
    genvoxnetDataset('dataset\\train\\all.npz').trans('dataset\\train\\all_full.npz')


def rotate(x,axis = [0,2]):
    k = random.randint(0,3)
    for i in x:
        if x[i].dim() < 2:
            continue
        x[i] = torch.rot90(x[i],k,axis)
    return x

class voxnetDataset(data.Dataset):
    def __init__(self, npz_file):
        data = np.load(npz_file)
        self.vox = data['vox']
        self.label = data['label']
    def __len__(self):
        return len(self.vox)

    def __getitem__(self, idx):

        vox = self.vox[idx]
        label = self.label[idx]

        item = {
            'vox':      torch.as_tensor(vox, dtype=torch.float32),
            'label':    torch.as_tensor(label, dtype=torch.long),
        }

        item = rotate(item)

        return item

def train():
    loader = {
        'train':DataLoader(voxnetDataset('dataset\\train\\all_full.npz'), batch_size=32,shuffle=True,drop_last=True,num_workers=0),
        'test':DataLoader(voxnetDataset('dataset\\test\\all_full.npz'), batch_size=32,shuffle=True,drop_last=True,num_workers=0),
    }
    model = voxnet().cuda()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    root_dir = 'result\\voxnet'
    best_loss = None
    for epoch in range(100):
        for phase in ['train', 'test']:
            losses = []
            correct = 0.
            all = 0.
            for i, data in tqdm(enumerate(loader[phase])):
                optimizer.zero_grad()
                inputs = data['vox'].cuda()
                labels = data['label'].cuda()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = nn.CrossEntropyLoss()(outputs, labels)
                    losses.append(loss.item())
                    _, predicted = torch.max(outputs.detach(), 1)
                    all += len(inputs)
                    correct += (predicted == labels).cpu().sum()
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
            loss = np.array(losses).mean()
            print(phase + ':',loss)
            print('acc:{}'.format(correct / all))
            if phase == 'test':
                if best_loss is None or best_loss > loss:
                    best_loss = loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                if epoch % 20 == 0:
                    torch.save(best_model_wts, os.path.join(root_dir, 'voxnet_best.weights'))
            else:
                scheduler.step()

if __name__ == "__main__":
    train()