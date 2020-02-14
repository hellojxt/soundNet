import numpy as np
import math,os

def list_all_files(rootdir):
    _files = []
    list = os.listdir(rootdir)
    for i in range(0,len(list)):
        path = os.path.join(rootdir,list[i])
        if os.path.isdir(path):
            _files.extend(list_all_files(path))
        if os.path.isfile(path):
            _files.append(path) 
    return _files

def hz2bark(f):
    z = 6 * np.arcsinh(f/600)
    return z

def bark2hz(z):
    f = 600*np.sinh(z/6)
    return f

resolution = 50
z_min = hz2bark(50)
z_max = hz2bark(7000)
spacing = (z_max - z_min)/resolution

def hz2index(f):
    z = hz2bark(f)
    return int((z-z_min)/spacing)

def index2hz(i):
    bark = z_min + (i+0.5)*spacing
    return bark2hz(bark)

def compress(amp, freq):
    lst = np.zeros(resolution)
    for i,a in enumerate(amp):
        idx = hz2index(freq[i])
        if idx < resolution and idx >= 0:
            lst[idx] += abs(a)
    return lst

def decompress(lst, alpha=2E-6, beta=60.0):
    fs = np.array([index2hz(i) for i in range(resolution)])
    c = 2.*(1 - np.sqrt(1 - alpha*(beta + alpha * (fs*2*math.pi)**2)) )/alpha
    return lst, fs, c

BITRATE=44100

def generate_wav(amp, freq, c):
    frame = np.arange(BITRATE*3)
    data = np.zeros_like(frame).astype(np.float32)
    for i in range(len(amp)):
        damp = np.exp(-0.5*c[i]*frame/BITRATE)
        sin = np.sin(freq[i]*2*math.pi*frame/BITRATE)
        data += amp[i]*damp*sin
    data = data / np.max(data)
    return data



