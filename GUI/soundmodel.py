from plyfile import PlyData, PlyElement
import numpy as np
import math
import os,configparser
from utils import helper
from model.net import EnvelopeNet,FrequencyNet
import torch
class mesh():
    def __init__(self, filename, mat= 4):

        self.filename = filename
        plydata = PlyData.read(filename)
        self.vertices = np.array(plydata['vertex'].data.tolist()).reshape(-1,3)
        self.faces = np.array(plydata['face'].data.tolist()).reshape(-1,3)
        self.voxels = np.array(plydata['voxel'].data.tolist()).reshape(-1,4)

        self.normalize()
        self.set_material(mat)
        
        self.load_modes()
        self.compute_vertex_normal()
        self.vals /= 0.25

    def normalize(self):
        self.vertices = self.vertices - self.vertices.mean(0)
        self.vertices = self.vertices / self.vertices.max() / 3

    def compute_vertex_normal(self):
        self.noramls = np.zeros_like(self.vertices)
        for face in self.faces:
            nml = self.face_nml(face)
            for vid in face:
                self.noramls[vid] += nml
        norm = np.linalg.norm(self.noramls, axis=1)
        for i,n in enumerate(self.noramls):
            self.noramls[i] = n/norm[i]

    def change_mat(self, mat):
        origin_k = self.youngs / self.density
        self.set_material(mat,printlog=True)
        now_k = self.youngs / self.density
        self.vals *= now_k / origin_k

    def load_modes(self):
        self.vals = np.load(self.filename + '.val.npy')
        self.vecs = np.load(self.filename + '.vec.npy')

    def face_nml(self, face):
        points = [self.vertices[x] for x in face]
        nml = np.cross(points[1]-points[0], points[2]-points[1])
        return nml/np.sqrt(np.sum(nml**2))

    def face_center(self, faceid):
        face = self.faces[faceid]
        points = np.array([self.vertices[x] for x in face])
        return sum(points)/3

    def click(self, faceid, force):
        force_unit = self.face_nml(self.faces[faceid])
        amp = np.zeros(len(self.vals))
        for vid in self.faces[faceid]:
            amp += force*force_unit.dot(self.vecs[3*vid:3*(vid+1),:])
        valid = (self.vals > (20*2*math.pi)**2)&(
                self.vals < (20000*2*math.pi)**2)&(
                1 - (self.alpha*self.vals + self.beta)**2/(self.vals*4) > 0)
        amp = amp[valid]

        vals = self.vals[valid]
        c = (self.alpha*vals + self.beta)
        omega = np.sqrt(vals)
        omega_d = omega*np.sqrt(1 - c**2/(omega**2*4))
        amp = amp / omega_d
        return np.abs(amp), omega_d/(2*np.pi), c

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


def read_off(filename):
    file = open(filename)
    if 'OFF' != file.readline().strip():
        raise('Not a valid OFF header')
    n_verts, n_faces, n_dontknow = tuple([int(s) for s in file.readline().strip().split(' ')])
    verts = [[float(s) for s in file.readline().strip().split(' ')] for i_vert in range(n_verts)]
    faces = [[int(s) for s in file.readline().strip().split(' ')][1:] for i_face in range(n_faces)]
    return np.array(verts), np.array(faces)

import matplotlib
matplotlib.use('Qt5Agg') 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def output_vox_image():
    vox = np.load('vox.npy')
    coord = np.load('coord.npy')
    coord = tuple(coord)
    colors = np.empty(vox.shape, dtype=object)
    colors[...] = 'green'
    colors[coord] = 'red'
    vox_ = np.zeros_like(vox)
    vox_[coord] = 1
    ax = plt.subplot(1,1,1,projection='3d')
    ax.voxels(vox_,facecolors=colors, edgecolors='k')
    ax.voxels(vox,facecolors=colors, edgecolors='k')
    plt.show()

def merge_random_frequency(freq):

    k = 50 // len(freq)
    new_envelope = np.zeros((3,50))
    for i,rate in enumerate(freq):
        mode_num = int(rate*k+0.5)
        idxs = np.random.permutation(k)[:mode_num]
        for idx in idxs:
            j = idx + i*k
            new_envelope[:,j] = 1
    return new_envelope

class voxMesh():
    def __init__(self, idx, mat= 4, deep = False):

        data = np.load('dataset//test//all_clean.npz')
        self.vox = data['vox'][idx]
        self.target = data['target'][idx]
        self.freq = data['freq'][idx]
        name = data['filename'][idx]
        self.filename = name.split('/')[-1]
        self.phase = name.split('/')[-2]

        type_dir = self.filename[0:-len(self.filename.split('_')[-1])-1]
        mesh_name = os.path.join('D:','modelnet40',type_dir,self.phase,self.filename)

        print(mesh_name)

        self.vertices,self.faces = read_off(mesh_name)
        self.normalize()
        self.set_material(mat)
        self.compute_vertex_normal()
        self.bbMax, self.bbMin = self.vertices.max(0), self.vertices.min(0)
        self.bb = self.bbMax - self.bbMin
        self.spacing = self.bb.max()/(32-6)
        self.leftBottom = (self.bbMin + self.bbMax)*0.5 - self.spacing*0.5 - self.spacing*15
        self.k = 1
        self.deep = deep
        if deep:
            self.model_envelope = EnvelopeNet().cuda()
            model_frequency = FrequencyNet().cuda()
            self.model_envelope.load_state_dict(torch.load(os.path.join('result','test_envelope', 'envelope_best.weights')))
            self.model_envelope.eval()
            model_frequency.load_state_dict(torch.load(os.path.join('result','test_frequency', 'frequency_best.weights')))
            model_frequency.eval()
            torch.set_grad_enabled(False)
            self.vox = torch.as_tensor(self.vox, dtype=torch.float32).unsqueeze(0).cuda()
            self.frequency = model_frequency(self.vox)[0].cpu().numpy()
            self.frequency = merge_random_frequency(self.frequency)

    def normalize(self):
        self.vertices = self.vertices - self.vertices.mean(0)
        self.vertices = self.vertices / self.vertices.max() / 3

    def compute_vertex_normal(self):
        self.noramls = np.zeros_like(self.vertices)
        for face in self.faces:
            nml = self.face_nml(face)
            for vid in face:
                self.noramls[vid] += nml
        norm = np.linalg.norm(self.noramls, axis=1)
        for i,n in enumerate(self.noramls):
            if norm[i] != 0:
                self.noramls[i] = n/norm[i]

    def change_mat(self, mat):
        origin_k = self.youngs / self.density
        self.set_material(mat,printlog=True)
        now_k = self.youngs / self.density
        self.k = now_k / origin_k

    def face_nml(self, face):
        points = [self.vertices[x] for x in face]
        nml = np.cross(points[1]-points[0], points[2]-points[1])
        return nml/np.sqrt(np.sum(nml**2))

    def face_center(self, faceid):
        face = self.faces[faceid]
        points = np.array([self.vertices[x] for x in face])
        return sum(points)/3

    def click(self, faceid, force):
        print(faceid)
        force_unit = self.face_nml(self.faces[faceid])
        p = self.face_center(faceid)
        coord = ((p - self.leftBottom)/self.spacing)
        # output_vox_image(vox,coord.astype(np.int))
        coord = np.array([coord[0],coord[2],coord[1]]) + 0.5
        coord = coord.astype(np.int)//2
        if self.deep:
            envelope = self.model_envelope.select(self.vox, coord)[0].cpu().numpy().reshape(3,-1)
            envelope = envelope * self.frequency
            lst = np.abs((force_unit*force).dot(envelope))
            target = self.target[tuple(coord)].reshape(3,-1)
            lst_ = np.abs((force_unit*force).dot(target)/255)
            plt.clf()
            ax = plt.subplot(121)
            ax.set_title('predict')
            ax.plot(lst)
            ax = plt.subplot(122)
            ax.set_title('groundtruth')
            ax.plot(lst_)
            plt.savefig('test.png')
            
        else:
            target = self.target[tuple(coord)].reshape(3,-1)
            lst = np.abs((force_unit*force).dot(target)/255)
        a,f,c = helper.decompress(lst)
        return self.translate(a,f,c)
    
    def click_test(self, p):
        coord = ((p - self.leftBottom)/self.spacing)
        # output_vox_image(vox,coord.astype(np.int))
        coord = np.array([coord[0],coord[2],coord[1]]) + 0.5
        print(coord)
        np.save('vox.npy', self.vox)
        np.save('coord.npy', coord.astype(np.int))
        output_vox_image()

    def translate(self, a_, f_, c_):
        wd_ = f_*2*np.pi
        w_ = np.sqrt(wd_*wd_ + c_*c_/4)
        val_ = w_*w_
        val = val_ * self.k
        w = np.sqrt(val)
        c = self.alpha*val + self.beta
        wd = np.sqrt(w*w - c*c/4)
        f = wd/(2*np.pi)
        a = a_  
        return a,f,c

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


if __name__ == "__main__":
    mesh = voxMesh(400)
    p1 = mesh.face_center(115)
    p2 = mesh.face_center(117)
    mesh.click_test(p1)
    mesh.click_test(p2)
    mesh.click_test((p1+p2)/2)
    
    