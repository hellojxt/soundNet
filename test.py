import matplotlib.pyplot as plt
import librosa.display
import utils.helper as helper
import utils.player as player
import numpy as np
import pandas as pd
import librosa
from mpl_toolkits.mplot3d import Axes3D

def test():
    fig = plt.figure(figsize=(10,6))

    for idx,i in  enumerate([2,1,3,0]):
        data = np.load('kmeans.npy')[i]
        ax = fig.add_subplot(2,2,idx+1)
        ax.tick_params(axis='y',which='both',left=False, right=False, labelleft=False)
        ax.set_ylim([0,1])
        plt.xlabel('mel-scale')
        plt.ylabel('log-amplitude')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.bar(np.arange(len(data)),data)

    plt.savefig('spec.png')

def cuboid_data(pos, size=(1,1,1)):
    # code taken from
    # https://stackoverflow.com/a/35978146/4124317
    # suppose axis direction: x: to left; y: to inside; z: to upper
    # get the (left, outside, bottom) point
    o = [a - b / 2 for a, b in zip(pos, size)]
    # get the length, width, and height
    l, w, h = size
    x = [[o[0], o[0] + l, o[0] + l, o[0], o[0]],  
         [o[0], o[0] + l, o[0] + l, o[0], o[0]],  
         [o[0], o[0] + l, o[0] + l, o[0], o[0]],  
         [o[0], o[0] + l, o[0] + l, o[0], o[0]]]  
    y = [[o[1], o[1], o[1] + w, o[1] + w, o[1]],  
         [o[1], o[1], o[1] + w, o[1] + w, o[1]],  
         [o[1], o[1], o[1], o[1], o[1]],          
         [o[1] + w, o[1] + w, o[1] + w, o[1] + w, o[1] + w]]   
    z = [[o[2], o[2], o[2], o[2], o[2]],                       
         [o[2] + h, o[2] + h, o[2] + h, o[2] + h, o[2] + h],   
         [o[2], o[2], o[2] + h, o[2] + h, o[2]],               
         [o[2], o[2], o[2] + h, o[2] + h, o[2]]]               
    return np.array(x), np.array(y), np.array(z)

def plotCubeAt(pos=(0,0,0),ax=None):
    # Plotting a cube element at position pos
    if ax !=None:
        X, Y, Z = cuboid_data( pos )
        ax.plot_surface(X, Y, Z, color='grey', rstride=1, cstride=1, alpha=1)

def plotMatrix(ax, matrix):
    # plot a Matrix 
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            for k in range(matrix.shape[2]):
                if matrix[i,j,k] == 1:
                    # to have the 
                    plotCubeAt(pos=(i-0.5,j-0.5,k-0.5), ax=ax)            

if __name__ == "__main__":
    test()