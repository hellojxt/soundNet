import torch
from torch.utils.data import DataLoader
import numpy as np
import os
import sys
import copy
import random
from tqdm import tqdm
from utils.dataset import HourGlassDataset
from model.netconv import EnvelopeNet,FrequencyNet
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import ml_metrics 

def plot_vox(fig, vox, x, y, idx):
    vox[16,:,16] = 1
    ax = fig.add_subplot(x,y,idx,projection='3d',xticks=[], yticks=[],zticks=[])
    ax.voxels(vox,edgecolors='black')
    return ax

def plot_vector(fig, data, x, y, idx):
    ax = fig.add_subplot(x,y,idx)
    ax.set_ylim(0,1)
    ax.bar(np.arange(len(data)),data,width = 0.25)
    return ax

def plot_lists(data_list, title_list = None):
    length = len(data_list)
    img_width = 6
    fig = plt.figure(figsize=(img_width*length, img_width))
    for idx in np.arange(length):
        data = data_list[idx]
        dim = len(data.shape)
        if dim == 3:
            ax = plot_vox(fig, data,1,length,idx+1)
        else:
            ax = plot_vector(fig, data,1,length,idx+1)
        if title_list != None:
            ax.set_title(title_list[idx])
        
    return fig

def main():
    filename = 'all.npz'
    trainset = HourGlassDataset('dataset/train/{}'.format(filename))
    target = trainset.target.reshape(-1,64)
    target = target[target.any(1)]
    sample = np.random.permutation(len(target))[:100000]
    target = target[sample]/255
    # print(target.shape)
    # pca = PCA(20)
    # pca.fit(target)
    # print(sum(pca.explained_variance_ratio_))
    # target_low_dim = pca.transform(target)
    kmeans = KMeans(n_clusters=4, random_state=0).fit(target)
    plot_lists(kmeans.cluster_centers_)
    plt.savefig('clustering.png')

    filename = 'all.npz'
    testset = HourGlassDataset('dataset/test/{}'.format(filename))
    test_loader = DataLoader(testset, batch_size=1,shuffle=True, num_workers=0,drop_last=True)
    model_envelope = EnvelopeNet().cuda()
    #model_frequency = FrequencyNet().cuda()
    model_envelope.load_state_dict(torch.load(os.path.join('result','envelope40', 'envelope_best.weights')))
    model_envelope.eval()
    #model_frequency.load_state_dict(torch.load(os.path.join('result','test_frequency', 'frequency_best.weights')))
    #model_frequency.eval()

    label1_all = []
    label2_all = []
    label3_all = []
    with torch.set_grad_enabled(False):
        for i, data in tqdm(enumerate(test_loader)):
            inputs = data['vox'].cuda()
            targets = data['target'].cuda()
            freqs = data['freq'].cuda()
            index = (targets != 0).any(-1)
            targets = targets[index]
            targets = targets.view(targets.size(0),3,-1)
            outputs_envelope = model_envelope(inputs, index)
            #outputs_frequency = model_frequency(inputs)

            envelopes = targets.cpu().numpy().reshape(-1,64)
            outputs_envelope = outputs_envelope.cpu().numpy().reshape(-1,64)
            
            if envelopes.shape[0] == 0:
                continue
            label1 = kmeans.predict(envelopes)
            label2 = kmeans.predict(outputs_envelope)
            label3 = kmeans.predict(np.random.rand(*envelopes.shape))
            label1_all = np.concatenate([label1_all,label1])
            label2_all = np.concatenate([label2_all,label2])
            label3_all = np.concatenate([label3_all,label3])
            


    all_precose = []
    num_list = []
    for i in range(kmeans.n_clusters):
        correct_num = sum((label1_all == label2_all) & (label1_all == i))
        all_num = sum(label1_all == i)
        precise = correct_num / all_num
        num_list.append(all_num)
        print('net:')
        print(precise)
        correct_num = sum((label1_all == label3_all) & (label1_all == i))
        all_num = sum(label1_all == i)
        precise = correct_num / all_num
        print('random:')
        print(precise)
    print(np.array(all_precose).mean())
    num_list = np.array(num_list)
    print(num_list.max()/num_list.sum())

    
if __name__ == "__main__":
    main()