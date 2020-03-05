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
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import shutil

def plot_vox(fig, vox, x, y, idx):
    vox[16,:,16] = 1
    ax = fig.add_subplot(x,y,idx,projection='3d',xticks=[], yticks=[],zticks=[])
    ax.voxels(vox,edgecolors='black')
    return ax

def plot_vector(fig, data, x, y, idx):
    ax = fig.add_subplot(x,y,idx,xticks=[], yticks=[])
    ax.bar(np.arange(len(data)),data,width = 0.25)
    return ax

def plot_lists(data_list, title_list):
    length = len(data_list)
    img_width = 6
    fig = plt.figure(figsize=(img_width*length, img_width))
    for idx in np.arange(length):
        data = data_list[idx]
        title = title_list[idx]
        dim = len(data.shape)
        if dim == 3:
            ax = plot_vox(fig, data,1,length,idx+1)
        else:
            ax = plot_vector(fig, data,1,length,idx+1)
        ax.set_title(title_list[idx])
    return fig

def main(net_name):
    filename = 'all_clean.npz'
    testset = HourGlassDataset('dataset/test/{}'.format(filename))
    test_loader = DataLoader(testset, batch_size=1,shuffle=True, num_workers=0,drop_last=True)
    if net_name == 'envelope':
        model = EnvelopeNet().cuda()
    else:
        model = FrequencyNet().cuda()
    model.load_state_dict(torch.load(os.path.join('result','test_'+net_name, net_name + '_best.weights')))
    model.eval()
    with torch.set_grad_enabled(False):
        for i, data in enumerate(test_loader):
            if net_name == 'envelope':
                inputs = data['vox'].cuda()
                targets = data['target'].cuda()
                index = (targets != 0).any(-1)
                targets = targets[index]
                outputs = model(inputs, index)
                idx = random.randrange(len(targets))

                inputs = inputs[0].cpu().numpy()
                targets = targets[idx].cpu().numpy()[:50]
                outputs = outputs[idx].cpu().numpy()[:50]
                
                    
            else:
                inputs = data['vox'].cuda()
                targets = data['freq'].cuda()
                outputs = model(inputs)

                inputs = inputs[0].cpu().numpy()
                targets = targets[0].cpu().numpy()
                outputs = outputs[0].cpu().numpy()

            data_list = [inputs, outputs, targets]
            title_list =  ['input', 'output', 'target']
            writer.add_figure(net_name, plot_lists(data_list,title_list),i)
            if i > 10:
                break

log_dir = os.path.join('result','test')
shutil.rmtree(log_dir)
writer = SummaryWriter(log_dir)

def check_all():
    main(writer, 'envelope')
    main(writer, 'frequency')

def check_vox():
    filename = 'all_clean.npz'
    testset = HourGlassDataset('dataset/test/{}'.format(filename))
    test_loader = DataLoader(testset, batch_size=1,shuffle=True, num_workers=0,drop_last=True)
    for i, data in enumerate(test_loader):
        inputs = data['vox'][0]
        data_list = [inputs]
        title_list =  ['vox']
        writer.add_figure('vox check', plot_lists(data_list,title_list),i)
        if i > 20:
            break
        
if __name__ == "__main__":
    check_vox()
    writer.close()
    