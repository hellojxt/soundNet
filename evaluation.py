import torch
from torch.utils.data import DataLoader
import numpy as np
import os
import sys
import copy
import random
from tqdm import tqdm
from utils.dataset import HourGlassDataset
import model.net as net
import matplotlib.pyplot as plt
import metric
import pandas as pd

class train_sample():
    def __init__(self,filename = 'dataset/train/all_full.npz'):
        self.node_per_item = 20*3
        self.item_per_class = 20
        data = np.load(filename)
        label = data['label'].astype(np.int)
        target_ = data['target']
        vox_ = data['vox']

        target = np.zeros((40, self.item_per_class, self.node_per_item, 64))
        vox = np.zeros((40,self.item_per_class,32,32,32))
        target_full = np.zeros((40, self.item_per_class, 16,16,16,3,64))
        class_item_num = np.zeros(40,dtype=np.int)

        index_list = []
        for i,l in enumerate(label):
            if class_item_num[l] < self.item_per_class:
                index_list.append(i)
                t = target_[i].reshape(-1,64)
                t = t[t.any(1)]
                sample = np.random.permutation(len(t))[:self.node_per_item]
                t = t[sample]/255
                vox[l,class_item_num[l]] = vox_[i]
                target_full[l,class_item_num[l]] = target_[i]/255
                target[l,class_item_num[l]] = t
                class_item_num[l] += 1

        self.vox = vox
        self.target = target
        self.target_ = target_full

class simple_random():
    def __init__(self):
        self.name = 'simple_random'
        print('{} init complete'.format(self.name))
    def __call__(self,vox):
        vox_ = torch.nn.MaxPool3d(2)(vox) 
        index_ = (vox_!= 0)
        length = len(vox_[index_])
        return torch.cuda.FloatTensor(length,3,64).uniform_()

class baseline():
    def __init__(self,target):
        self.name = 'baseline'
        self.data = torch.tensor(target.reshape(-1,64)).cuda()
        print('{} init complete'.format(self.name))

    def __call__(self,vox):
        vox_ = torch.nn.MaxPool3d(2)(vox) 
        index = torch.where(vox_!= 0)[0].repeat(3)
        sample = torch.randint_like(index,0,self.data.size(0),device='cuda:0')
        return self.data[sample].view(-1,3,64)

class voxnet():
    def __init__(self,target,vox,weights_path):
        self.name = 'voxnet'
        self.model = net.frequency_conv(32).cuda()
        self.model.load_state_dict(torch.load(weights_path,map_location='cuda:0'))
        self.model.eval()

        vox = torch.tensor(vox.reshape(-1,32,32,32), dtype=torch.float32).cuda()
        self.data = torch.tensor(target.reshape(len(vox), -1, 64), dtype=torch.float32).cuda()
        self.features = self.model.body(vox).view(-1,256)
        self.weights = torch.ones(self.data.shape[1], dtype=torch.float).cuda()
        print('{} init complete'.format(self.name))

    def __call__(self,vox):
        vox_ = torch.nn.MaxPool3d(2)(vox) 
        index = torch.where(vox_!= 0)[0].unsqueeze(1).repeat(1,3).view(-1)

        outputs = self.model.body(vox).view(-1,256)
        outputs = outputs.unsqueeze(1).repeat(1,len(self.features),1)
        _,nearst_index = ((self.features - outputs)**2).sum(-1).min(-1)

        index = nearst_index[index]
        sample = torch.randint_like(index,0,self.data.size(1),device='cuda:0')

        target = self.data[index,sample].view(-1,3,64)
        return target

class single():
    def __init__(self, weights_path_envelope):
        self.name = 'single network'
        self.model = net.envelope_conv().cuda()
        self.model.load_state_dict(torch.load(weights_path_envelope,map_location='cuda:0'))
        self.model.eval()
        print('{} init complete'.format(self.name))

    def __call__(self,vox):
        vox_ = torch.nn.MaxPool3d(2)(vox) 
        index = torch.where(vox_!= 0)[0]
        envelope = self.model(vox,(vox_ != 0))
        return envelope

class single_match():
    def __init__(self,target,vox , weights_path_envelope):
        self.res = 32
        self.name = 'single_match'
        self.model = net.envelope_conv().cuda()
        self.model.load_state_dict(torch.load(weights_path_envelope,map_location='cuda:0'))
        self.model.eval()

        vox = torch.tensor(vox.reshape(-1,32,32,32), dtype=torch.float32).cuda()

        print(target.shape)
        target = torch.tensor(target.reshape(-1,16,16,16,3,64), dtype=torch.float32).cuda()
        
        self.features = torch.zeros(0,128,device='cuda:0')
        self.data = torch.zeros(0,3,64,device='cuda:0')

        print('{} init'.format(self.name))
        step = 40
        step_length = len(vox) // step
        for i in tqdm(range(step)):
            vox_ = vox[i*step_length:(i+1)*step_length]
            target_ = target[i*step_length:(i+1)*step_length]
            self.init_feature(vox_, target_)
        
    def init_feature(self, vox, target):
        vox_ = torch.nn.MaxPool3d(2)(vox)
        index = (vox_ != 0)
        x = self.model.body(vox)
        x = x.permute(0,2,3,4,1)[index]
        x = x.view(-1,128)
        self.features = torch.cat((self.features, x))
        self.data = torch.cat((self.data,target[index]))

    
    def __call__(self,vox):
        vox_ = torch.nn.MaxPool3d(2)(vox)
        index = (vox_ != 0)
        x = self.model.body(vox)
        x = x.permute(0,2,3,4,1)[index]
        x = x.view(-1,128)
        target = torch.zeros(0,3,64,device='cuda:0')

        for feature in x:
            _,nearst_index = ((self.features - feature)**2).sum(-1).min(-1)
            data = self.data[nearst_index.item()].unsqueeze(0)
            target = torch.cat((target,data))

        return target
        
class ours():
    def __init__(self, weights_path_envelope, weights_path_frequency):
        self.res = 32
        self.name = 'double network'
        self.model_envelope = net.envelope_conv().cuda()
        self.model_envelope.load_state_dict(torch.load(weights_path_envelope,map_location='cuda:0'))
        self.model_envelope.eval()
        self.model_frequency = net.frequency_conv(self.res).cuda()
        self.model_frequency.load_state_dict(torch.load(weights_path_frequency,map_location='cuda:0'))
        self.model_frequency.eval()
        print('{} init complete'.format(self.name))
    def __call__(self,vox):

        vox_ = torch.nn.MaxPool3d(2)(vox) 
        index = torch.where(vox_!= 0)[0]
        freq = self.model_frequency(vox)
        envelope = self.model_envelope(vox,(vox_ != 0))

        k = 64 // self.res
        freq = freq.unsqueeze(-1).repeat(1,1,k)
        freq = (freq*k + 0.5).int()
        shuffle = torch.argsort(torch.randint_like(freq,0,self.res,device='cuda:0')) + 1
        shuffle[shuffle > freq] = 0
        shuffle[shuffle != 0] = 1
        shuffle = shuffle.view(-1,64).unsqueeze(1).repeat(1,3,1)

        return envelope*shuffle[index]

import time

if __name__ == "__main__":
    BatchSize = 16
    torch.set_grad_enabled(False)
    train_data = train_sample()

    model_list = [
        #simple_random(),
        #baseline(train_data.target),
        #voxnet(train_data.target, train_data.vox, 'result/frequency_conv_res_32/frequency_conv_best.weights'),
        single('result/envelope_conv_single/envelope_conv_best.weights'),
        #single_match(train_data.target_, train_data.vox, 'result/envelope_conv/envelope_conv_best.weights'),
        ours('result/envelope_conv/envelope_conv_best.weights','result/frequency_conv_res_32/frequency_conv_best.weights'),
    ]

    outputs_list = [None for model in model_list]
    targets = None
    loader = DataLoader(HourGlassDataset('dataset/test/all.npz',32), batch_size=BatchSize, num_workers=0)

    idx = 0

    for data in tqdm(loader):
        vox = data['vox'].cuda()
        target = data['target'].cuda()
        index = (target != 0).any(-1)
        target = target[index].cpu().numpy().reshape(-1,64)
        
        if targets is None:
            targets = target
        else:
            targets = np.vstack([targets,target])
        
        for i in range(len(model_list)):
            output = model_list[i](vox).cpu().numpy().reshape(-1,64)
            if outputs_list[i] is None:
                outputs_list[i] = output
            else:
                outputs_list[i] = np.vstack([outputs_list[i],output])

        if idx > 10:
            break
        else:
            idx += 1

    metric_list = [
        metric.l2(),
        metric.meanf(),
        metric.kmeans(train_data.target),
        metric.score()
    ]

    table = {}
    table['name'] = [model.name for model in model_list]
    for metric_class in tqdm(metric_list):
        table[metric_class.name] = [metric_class(targets,outputs) for outputs in outputs_list]
    
    print(pd.DataFrame(table)) 

    
    n = len(model_list)
    fig = plt.figure(figsize=(14,10))
    for k in range(1):
        plt.clf()
        #lst = np.random.choice(len(targets), 4)
        lst = [124281,25765,170682,211270,69883]
        for j,idx in enumerate(lst):
            ax = fig.add_subplot(len(lst),n+1,j*(n+1) + 1)
            ax.tick_params(axis='y',which='both',left=False, right=False, labelleft=False)
            ax.set_ylim([0,1])
            plt.xlabel('ground truth')
            ax.spines['top'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(False)
            data = targets[idx]
            ax.bar(np.arange(len(data)),data)
            for i in range(n):
                ax = fig.add_subplot(len(lst),n+1,j*(n+1) + 2 + i)
                ax.tick_params(axis='y',which='both',left=False, right=False, labelleft=False)
                ax.set_ylim([0,1])
                plt.xlabel(model_list[i].name)
                plt.ylabel('')
                ax.spines['top'].set_visible(False)
                ax.spines['left'].set_visible(False)
                ax.spines['right'].set_visible(False)
                data = outputs_list[i][idx]
                ax.bar(np.arange(len(data)),data)
        plt.savefig('out.png'.format(k))
        

    
            
