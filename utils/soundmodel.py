from plyfile import PlyData, PlyElement
import numpy as np
import math
import os,configparser

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


if __name__ == "__main__":
    mesh('dataset\\test\\airplane_0627.off.ply')
    