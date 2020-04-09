import numpy as np
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from tqdm import tqdm
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

def smooth(dir_path,writer, npz_file = 'all.npz'):
    name = os.path.join(dir_path,npz_file)
    data = np.load(name)
    vox = data['vox']
    target = data['target']
    freq = data['freq']
    filename = data['filename']
 
    index = target.any(-1)

    target_valid = target[index]
    for i,lst in tqdm(enumerate(target_valid)):
        args = np.argwhere(lst != 0)
        idx = 0
        lst_smooth = np.zeros_like(lst)
        for j,item in enumerate(lst):
            if item == 0:
                lst_smooth[j] = lst[args[idx][0]]
            else:
                lst_smooth[j] = item
            if idx < len(args) - 1:
                if abs(args[idx+1][0] - j) < abs(args[idx][0] - j):
                    idx += 1
        if i < 1000 and i % 100 == 0:
            writer.add_figure(dir_path, plot_lists([lst/255,lst_smooth/255]),i//100)
        target_valid[i] = lst_smooth
        
    target[index] = target_valid

    np.savez_compressed(os.path.join(dir_path,'all_smooth'), 
            vox = vox, 
            target = target,
            freq = freq,
            filename = filename
            )
    
if __name__ == "__main__":
    writer = SummaryWriter('result/smoothcheck')
    smooth('dataset/test',writer)
    smooth('dataset/train',writer)
    writer.close()