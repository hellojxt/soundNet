import torch
from hourglass import Baseline
import sys
import os
from utils import helper,player
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import configparser
import torch.utils.data as data
def test_img():
    mat = 0
    matdir = 'material'
    mat_config_name = os.path.join(matdir, 'material-{}.cfg'.format(mat))
    mat_config = configparser.ConfigParser()
    mat_config.read(mat_config_name)    
    m = mat_config['DEFAULT']
    alpha = float(m['alpha'])
    beta = float(m['beta'])


    model = Baseline().cuda()
    model.load_state_dict(torch.load('output/addrate_1/hourglass.pth'))
    model.eval()
    data = np.load('dataset/test/all_clean.npz')
    filenames = data['filename']
    for i,filename in enumerate(filenames):
        if 'bowl' in filename:
            break
    i = i+5
    vox = torch.tensor(data['vox'][i]).float().unsqueeze(0).unsqueeze(0).cuda()
    target = torch.tensor(data['target'][i][:,:,:,0,:]).float().unsqueeze(0).cuda()
    freq = data['freq'][i].reshape(model.res,helper.resolution//model.res).mean(-1)
    index = (target != 0).any(-1)
    x,g = model.predict(vox)
    x = x.permute(0,2,3,4,1)[index]
    x = model.end_line(x).cpu().detach().numpy().reshape(-1,50)
    g = g.cpu().detach().numpy().reshape(10)
    vox = torch.nn.MaxPool3d(2)(vox).cpu().detach().numpy().reshape(16,16,16)
    target = target[index].cpu().detach().numpy().reshape(-1,50)
    index = np.argwhere(index.cpu().detach().numpy()==True)
    idx_lst = []
    for i,rate in enumerate(g):
        lst = np.arange(5) + i*5
        j = random.sample(list(lst),int(rate*5+0.5))
        idx_lst.append(j)
    
    freq_sample = np.zeros(50)
    for j in idx_lst:
        freq_sample[j] = 1

    for idx in [100,100,200,300]:
        colors = np.empty(vox.shape, dtype=object)
        colors[...] = 'grey'
        ax = plt.subplot(1,1,1,projection='3d')
        ax.voxels(vox,facecolors=colors, edgecolors='black')
        ax.set_xticks(np.linspace(0,32,9))
        ax.set_xticklabels(np.linspace(0,32,9).astype(np.int))
        ax.set_yticks(np.linspace(0,32,9))
        ax.set_yticklabels(np.linspace(0,32,9).astype(np.int))
        ax.set_zticks(np.linspace(0,32,9))
        ax.set_zticklabels(np.linspace(0,32,9).astype(np.int))
        plt.savefig('test.png')
        plt.clf()

        plt.plot(x[idx])
        plt.savefig('envelope.png')
        plt.clf()

        plt.plot(freq_sample)
        plt.savefig('freq.png')
        plt.clf()

        plt.plot(x[idx]*freq_sample)
        plt.savefig('end.png')
        plt.clf()

        plt.plot(target[idx])
        plt.savefig('target.png')
        break
        # plt.subplot(614)
        # plt.plot(freq)
        # output = np.zeros_like(x[idx])
        # for j in idx_lst:
        #     output[j] = x[idx][j]
        # output[output < 0.1] = 0
        # plt.subplot(615)
        # plt.plot(output)
        # a,f,c = helper.decompress(output,alpha=alpha, beta=beta)
        # a = a/a.max()
        # plt.subplot(616)
        # plt.plot(a)
        # plt.savefig('{}.png'.format(idx))
        # a,f,c = helper.decompress(target[idx],alpha=alpha, beta=beta)
        # a = a/a.max()
        # player.write_wav(a,f,c,'{}g.wav'.format(idx))

def draw_spectrum_seperate():
    data = np.load('dataset/test/all_clean.npz')
    filenames = data['filename']
    for i,filename in enumerate(filenames):
        if 'bowl' in filename:
            break
    i = i+15
    target = data['target'][i][:,:,:,0,:].reshape(-1,50)
    freq = data['freq'][i]
    amp = target[(target != 0).any(-1)][0]/255.

    plt.figure(figsize=(10,8))
    plt.subplot(321)
    plt.bar(np.arange(len(amp)),amp,width=1)
    plt.subplot(323)
    amp_ = np.zeros_like(amp)
    index = np.where(freq == 1)[0]
    freq = data['freq'][i].reshape(10,5).mean(-1)
    nearist = 0
    for i,a in enumerate(amp):
        if nearist < len(index) - 1:
            if np.abs(i - index[nearist]) > np.abs(i - index[nearist+1]):
                nearist += 1
        if np.abs(i - index[nearist]) > 5:
            continue
        amp_[i] = amp[index[nearist]]
    plt.bar(np.arange(len(amp_)),amp_,width=1)
    plt.subplot(325)
    amp_smooth = np.zeros_like(amp)
    for i in range(len(amp_)):
        r = min(len(amp), i+3)
        l = max(0, i-2)
        amp_smooth[i]  = amp_[l:r].mean()
    plt.bar(np.arange(len(amp_smooth)),amp_smooth,width=1)

    plt.subplot(322)

    idx_lst = []
    for i,rate in enumerate(freq):
        lst = np.arange(5) + i*5
        j = random.sample(list(lst),int(rate*5+0.5))
        idx_lst.append(j)
    freq_sample = np.zeros(50)
    for j in idx_lst:
        freq_sample[j] = 1
    plt.bar(np.arange(len(freq_sample)),freq_sample,width=1)
    plt.subplot(324)
    amp_ = amp_ * freq_sample
    amp_smooth = amp_smooth * freq_sample
    plt.bar(np.arange(len(amp_)),amp_,width=1)
    plt.subplot(326)
    plt.bar(np.arange(len(amp_smooth)),amp_smooth,width=1)

    plt.savefig('test.png')    
    
class HourGlassDataset(data.Dataset):
    def __init__(self, npz_file):
        data = np.load(npz_file)
        self.vox = data['vox']
        self.target = data['target']
        self.freq = data['freq']
        self.filename = data['filename']
        
    def __len__(self):
        return len(self.vox)

    def save_to(self, filename):
        np.savez_compressed(filename, 
            vox = self.vox, 
            target = self.target,
            freq = self.freq,
            filename = self.filename
            )

    def process(self):
        for i in range(len(self.vox)):
            freq = self.freq[i]
            target = self.target[i]
            origin_size = target.shape
            target = target.reshape(-1,50)
            index = np.where(freq == 1)[0]
            for j,t in enumerate(target):
                if (t == 0).all():
                    continue
                t_ = np.zeros_like(t)
                nearist = 0
                for i,a in enumerate(t):
                    if nearist < len(index) - 1:
                        if np.abs(i - index[nearist]) > np.abs(i - index[nearist+1]):
                            nearist += 1
                    if np.abs(i - index[nearist]) > 5:
                        continue
                    t_[i] = t[index[nearist]]

                
test_img()