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
import model.net as net
import argparse

def main(NetName, cuda, lr):
    device = torch.device("cuda:{}".format(cuda))
    if 'envelope' in NetName:
        net_type = 'envelope'
        root_dir = os.path.join('result',NetName)
        model = getattr(net,NetName)().to(device)
    if 'frequency' in NetName:
        net_type = 'frequency'
        root_dir = os.path.join('result',NetName + '_res_{}'.format(FREQ_RES))
        res= int(np.log(FREQ_RES)/np.log(2) - 2)
        model = getattr(net,NetName)(res).to(device)

    writer = SummaryWriter(root_dir)
    loss_fun = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    best_loss = None
    torch.backends.cudnn.benchmark = True
    for epoch in range(MAX_EPOCH):
        print(root_dir + '\tepoch:{}'.format(epoch))
        for phase in ['train', 'test']:
            losses = []
            for i, data in tqdm(enumerate(loader[phase])):
                optimizer.zero_grad()
                if net_type == 'envelope':
                    inputs = data['vox'].to(device)
                    targets = data['target'].to(device)
                    index = (targets != 0).any(-1)
                    targets = targets[index]
                    targets = targets.view(targets.size(0),3,-1)
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs, index)
                        loss = loss_fun(outputs, targets)
                        losses.append(loss.item())
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                else:
                    inputs = data['vox'].to(device)
                    targets = data['freq'].to(device)
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        loss = loss_fun(outputs, targets)
                        losses.append(loss.item())
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

            loss = np.array(losses).mean()
            writer.add_scalar(root_dir+'/'+phase, loss, epoch)
            if phase == 'test':
                if best_loss is None or best_loss > loss:
                    best_loss = loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                if epoch % 20 == 0:
                    torch.save(best_model_wts, os.path.join(root_dir, NetName+'_best.weights'))
            else:
                scheduler.step()
            print('loss:{}'.format(loss))
    writer.close()
    print(NetName+'best_loss:{}'.format(best_loss))  
    f = open(os.path.join(root_dir,'best_loss.txt'),'w')
    f.write('best_loss:{}'.format(best_loss))
    f.close()
      

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train envelope net or frequency net.')
    parser.add_argument('--net', type=str, help='class name of the network')
    parser.add_argument('--res', type=int, default=8, help='resolution for frequency')
    parser.add_argument('--cuda', type=int, default=0, help='GPU index')
    parser.add_argument('--epoch', type=int, default=100, help='max epoch')
    parser.add_argument('--bsize', type=int, default=16, help='batch size')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    
    args = parser.parse_args()

    MAX_EPOCH = args.epoch
    FREQ_RES = args.res
    BATCH_SIZE = args.bsize

    trainset = HourGlassDataset('dataset/train/all_smooth.npz',FREQ_RES)
    testset = HourGlassDataset('dataset/test/all_smooth.npz',FREQ_RES)

    loader = {
        'train':DataLoader(trainset, batch_size=BATCH_SIZE,shuffle=True,drop_last=True,num_workers=8),
        'test':DataLoader(testset, batch_size=BATCH_SIZE,shuffle=True,drop_last=True,num_workers=8),
    }
    main(args.net, args.cuda, args.lr)
    
    

