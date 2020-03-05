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

def main(OutputDir, NetName):
    BATCH_SIZE = 16
    filename = 'all_clean.npz'
    trainset = HourGlassDataset('dataset/train/{}'.format(filename))
    testset = HourGlassDataset('dataset/test/{}'.format(filename))
    loader = {
        'train':DataLoader(trainset, batch_size=BATCH_SIZE,shuffle=True, num_workers=0,drop_last=True),
        'test':DataLoader(testset, batch_size=BATCH_SIZE,shuffle=True, num_workers=0,drop_last=True),
    }
    loss_fun = nn.MSELoss()
    root_dir = os.path.join('result',OutputDir)
    writer = SummaryWriter(root_dir)
    if NetName == 'envelope':
        model = EnvelopeNet().cuda()
    else:
        model = FrequencyNet().cuda()
    optimizer = optim.Adam(model.parameters(),0.001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)
    best_loss = None

    for epoch in tqdm(range(20)):
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
    main('test_frequency','frequency')
    main('test_envelope','envelope')
    

