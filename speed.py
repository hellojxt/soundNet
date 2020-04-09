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
from plyfile import PlyData, PlyElement
import math
import os,configparser
import multiprocessing,signal
from scipy.sparse import csr_matrix,coo_matrix
from scipy.sparse.linalg import eigsh
import time
import utils.helper as helper

class mesh():
    def __init__(self, filename, mat= 4):
        self.filename = filename
        plydata = PlyData.read(filename)
        self.vertices = np.array(plydata['vertex'].data.tolist()).reshape(-1,3)
        self.faces = np.array(plydata['face'].data.tolist()).reshape(-1,3)
        self.voxels = np.array(plydata['voxel'].data.tolist()).reshape(-1,4)
        self.set_material(mat)

    def set_material(self, mat, matdir = 'material',printlog = False):
        mat_config_name = os.path.join(matdir, 'material-{}.cfg'.format(mat))
        mat_config = configparser.ConfigParser()
        mat_config.read(mat_config_name)    
        m = mat_config['DEFAULT']
        if printlog:
            print('set material to ' + m['name'])
        self.youngs = float(m['youngs'])
        self.poison = float(m['poison'])
        self.alpha = float(m['alpha'])
        self.beta = float(m['beta'])
        self.density = float(m['density'])

    def compute_modes(self):
        print('wood, extracting matrix...')
        threadnum = 12
        inputs = [[self, i, threadnum] for i in range(threadnum)]
        pool = multiprocessing.Pool(processes=threadnum)
        pool_outputs = pool.map(global_matrix_thread, inputs)
        pool.close()
        pool.join()
        result_m = pool_outputs[0][0]
        result_k = pool_outputs[0][1]
        for result in pool_outputs[1:]:
            result_m = np.concatenate((result_m,result[0]),axis=1)
            result_k = np.concatenate((result_k,result[1]),axis=1)
        size = len(self.vertices)*3
        coo_m = coo_matrix((result_m[0], (result_m[1], result_m[2])), shape=(size, size))
        coo_k = coo_matrix((result_k[0], (result_k[1], result_k[2])), shape=(size, size))
        if (size < 500):
            return False
        modes_num = 100
        max_freq = 10000
        sigma = ((2*math.pi*max_freq)**2 + (2*math.pi*20)**2)/2
        start = time.time()
        print('compute modes...')
        vals, vecs = eigsh(coo_k, k=modes_num, M=coo_m,which='LM',sigma=sigma)
        while max(vals) < (2*math.pi*max_freq)**2 :
            modes_num *= 2
            print('recompute modes num:{}'.format(modes_num))
            vals, vecs = eigsh(coo_k, k=modes_num, M=coo_m,which='LM',sigma=sigma)
        print(len(self.vertices))
        print('cost time:{}'.format(time.time() - start))


        name = self.filename.split('\\')[-1]
        np.save('output/'+name+'.val',vals)
        np.save('output/'+name+'.vec',vecs)
        return time.time() - start


def global_matrix_thread(params):
    mesh = params[0]
    r = params[1]
    threadnum = params[2]
    Ms = []
    rows_m = []
    cols_m = []
    Ks = []
    rows_k = []
    cols_k = []
    size = len(mesh.voxels)
    m = helper.get_M(mesh.density)
    E = helper.get_E(mesh.youngs,mesh.poison)
    for idx in range(size):
        if idx % threadnum == r:
            ids = mesh.voxels[idx]
            rowid = np.vstack([ids*3,ids*3+1,ids*3+2]).T.reshape(-1)
            col = np.vstack([rowid]*12)
            row = col.T
            x,y,z = mesh.vertices[ids].T
            a,b,c,v = helper.get_abcV(x,y,z)
            B = helper.get_B(a,b,c,v)
            K = v*B.T.dot(E).dot(B).astype(np.float)
            M = m*v
            args = (M != 0)
            Ms.append(M[args])
            rows_m.append(row[args])
            cols_m.append(col[args])
            args = (K != 0)
            Ks.append(K[args])
            rows_k.append(row[args])
            cols_k.append(col[args])
    data_m = np.hstack(Ms)
    row_m = np.hstack(rows_m)
    col_m = np.hstack(cols_m)
    data_k = np.hstack(Ks)
    row_k = np.hstack(rows_k)
    col_k = np.hstack(cols_k)
    return [[data_m,row_m,col_m],[data_k,row_k,col_k]]

class ours():
    def __init__(self, weights_path_envelope, weights_path_frequency):
        self.res = 32
        self.name = 'ours'
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
        start = time.time()
        freq = self.model_frequency(vox)
        envelope = self.model_envelope.select(vox,[0,0,0])
        return time.time() - start

    
from mpl_toolkits.mplot3d import Axes3D
if __name__ == "__main__":
    BatchSize = 64
    torch.set_grad_enabled(False)
    model = ours('result/envelope_conv/envelope_conv_best.weights','result/frequency_conv_res_32/frequency_conv_best.weights')
    loader = DataLoader(HourGlassDataset('dataset/test/all.npz',32), batch_size=BatchSize, num_workers=0,shuffle=True)
    # print(mesh('D:/dataset/test/dresser_0205.off.ply').compute_modes())
    
    for data in loader:
        vox = data['vox'].cuda()
        model(vox)
        break


    time1 = 0
    time2 = 0
    nums = 0
    for data in loader:

        nums += len(data['name'])
        # vox = data['vox'].cuda()
        # time2 += model(vox)
        for name in data['name']:
            time1 += mesh(name+'.ply').compute_modes()
        if nums > 100:
            break

    print(nums,time1,time2)





    
