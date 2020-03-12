from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import numpy as np
import os
import sys
import copy
import random
from tqdm import tqdm
from utils.dataset import HourGlassDataset
from model.net import EnvelopeNet,FrequencyNet

def main(OutputDir, NetName, MaxEpoch, loader):
    loss_fun = nn.MSELoss()
    root_dir = os.path.join('result',OutputDir + str(MaxEpoch))
    writer = SummaryWriter(root_dir)
    if NetName == 'envelope':
        model = EnvelopeNet().cuda()
    else:
        model = FrequencyNet().cuda()
    optimizer = optim.Adam(model.parameters(),0.001)
    #optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)
    
    best_loss = None
    torch.backends.cudnn.benchmark = True
    for epoch in tqdm(range(MaxEpoch)):
        for phase in ['train', 'test']:
            losses = []
            for i, data in enumerate(loader[phase]):
                optimizer.zero_grad()
                if NetName == 'envelope':
                    inputs = data['vox'].cuda()
                    targets = data['target'].cuda()
                    index = (targets != 0).any(-1)
                    targets = targets[index]
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs, index)
                        loss = loss_fun(outputs, targets)
                        losses.append(loss.item())
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                else:
                    inputs = data['vox'].cuda()
                    targets = data['freq'].cuda()
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        loss = loss_fun(outputs, targets)
                        losses.append(loss.item())
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

            if phase == 'train':
                scheduler.step()
        
            loss = np.array(losses).mean()
            writer.add_scalar('loss/'+phase, loss, epoch)
            if phase == 'test':
                if best_loss is None or best_loss > loss:
                    best_loss = loss
                    best_model_wts = copy.deepcopy(model.state_dict())
    writer.close()
    torch.save(best_model_wts, os.path.join(root_dir, NetName+'_best.weights'))

if __name__ == "__main__":
    filename = 'all.npz'
    BATCH_SIZE = 16
    trainset = HourGlassDataset('dataset/train/{}'.format(filename))
    testset = HourGlassDataset('dataset/test/{}'.format(filename))
    loader = {
        'train':DataLoader(trainset, batch_size=BATCH_SIZE,shuffle=True,drop_last=True),
        'test':DataLoader(testset, batch_size=BATCH_SIZE,shuffle=True,drop_last=True),
    }
    main('envelope','envelope',100,loader)
    #main('frequency','frequency',150,loader)
    
    

