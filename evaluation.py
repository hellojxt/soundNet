import torch
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

def diff_band_point2set(p,i,lst,band_width):
    length = len(lst)
    dist_min = 1
    for d in range(-band_width, band_width+1):
        j = i+d
        if j >= 0 and j < length:
            dist_min = min( 
                            (d/band_width)**2 + (p - lst[j])**2,
                            dist_min
                        )
    return dist_min

def diff_band_set2set(list1, list2, band_width):
    dist_all = 0
    for i,p in enumerate(list1):
        dist_all += p*diff_band_point2set(p,i,list2,band_width)
    for i,p in enumerate(list2):
        dist_all += p*diff_band_point2set(p,i,list1,band_width)

    return dist_all / (sum(list1) + sum(list1))

def merge_random_frequency(envelope, freq):
    k = len(envelope) // len(freq)
    new_envelope = np.zeros_like(envelope)

    for i,rate in enumerate(freq):
        mode_num = int(rate*k+0.5)
        idxs = np.random.permutation(k)[:mode_num]
        for idx in idxs:
            j = idx + i*k
            new_envelope[j] = envelope[j]

    return new_envelope

def plot_lists(data_list, title_list):
    length = len(data_list)
    img_width = 5
    fig = plt.figure(figsize=(img_width*length, img_width))
    for idx in np.arange(length):
        data = data_list[idx]
        title = title_list[idx]
        ax = fig.add_subplot(1,length,idx+1,xticks=[], yticks=[])
        ax.bar(np.arange(len(data)),data)
        ax.set_title(title)
    return fig


def evaluation_random_sample(outputs_envelope, outputs_frequency):
    data_list  = [
        outputs_envelope,
        outputs_frequency,
        merge_random_frequency(outputs_envelope, outputs_frequency),
    ]
    
    title_list = [
        'envelope',
        'freqs',
        'merge'
    ]
    fig = plot_lists(data_list, title_list)
    plt.show()

def main():
    filename = 'all_clean.npz'
    testset = HourGlassDataset('dataset/test/{}'.format(filename))
    test_loader = DataLoader(testset, batch_size=1,shuffle=True, num_workers=0,drop_last=True)
    model_envelope = EnvelopeNet().cuda()
    model_frequency = FrequencyNet().cuda()
    model_envelope.load_state_dict(torch.load(os.path.join('result','test_envelope', 'envelope_best.weights')))
    model_envelope.eval()
    model_frequency.load_state_dict(torch.load(os.path.join('result','test_frequency', 'frequency_best.weights')))
    model_frequency.eval()

    scores_without_sample = []
    scores_sample = []
    scores_random = []

    with torch.set_grad_enabled(False):
        for i, data in enumerate(test_loader):
            inputs = data['vox'].cuda()
            targets = data['target'].cuda()
            freqs = data['freq'].cuda()
            index = (targets != 0).any(-1)
            targets = targets[index]

            outputs_envelope = model_envelope(inputs, index)
            outputs_frequency = model_frequency(inputs)

            idx = random.randrange(len(targets))
            inputs = inputs[0].cpu().numpy()
            envelopes = targets[idx].cpu().numpy()[:50]
            outputs_envelope = outputs_envelope[idx].cpu().numpy()[:50]
            freqs = freqs[0].cpu().numpy()
            outputs_frequency = outputs_frequency[0].cpu().numpy()

            scores_without_sample.append(diff_band_set2set(envelopes, outputs_envelope,5))
            scores_sample.append(diff_band_set2set(envelopes, merge_random_frequency(outputs_envelope,outputs_frequency),5))
            scores_random.append(diff_band_set2set(envelopes, np.random.rand(*envelopes.shape),5))
            if i > 100:
                break

    print('scores_without_sample:{}'.format(np.array(scores_without_sample).mean()))
    print('scores_sample:{}'.format(np.array(scores_sample).mean()))
    print('scores_random:{}'.format(np.array(scores_random).mean()))
    

if __name__ == "__main__":
    main()






