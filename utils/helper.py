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

resolution = 64
z_min = hz2bark(50)
z_max = hz2bark(10000)
spacing = (z_max - z_min)/resolution

def hz2index(f):
    z = hz2bark(f)
    return int((z-z_min)/spacing)

def index2hz(i):
    bark = z_min + (i+0.5)*spacing
    return bark2hz(bark)
p_0 = 0.01
k_0 = 0.5
def amp2spl(a):
    return k_0*np.log10(a/p_0)

def spl2amp(db):
    if db == 0:
        return 0
    return 10**(db/k_0)*p_0
    
def compress(amp, freq, c=None):
    lst = np.zeros(resolution)
    amp = np.abs(amp)
    if amp.max() == 0:
        return np.nan
    for i,a in enumerate(amp):
        idx = hz2index(freq[i])
        if idx < resolution and idx >= 0:
            lst[idx] += a
    for i,a in enumerate(lst):
        if a > 0:
            lst[i] = amp2spl(a)
    lst = lst - lst.max() + 1
    lst[lst < 0] = 0
    return lst

def freq_compress(freq):
    lst = np.zeros(resolution)
    for f in freq:
        idx = hz2index(f)
        if idx < resolution and idx >= 0:
            lst[idx] = 1
    return lst

def decompress(lst, alpha=2E-6, beta=60.0):
    fs = np.array([index2hz(i) for i in range(resolution)])
    c = 2.*(1 - np.sqrt(1 - alpha*(beta + alpha * (fs*2*math.pi)**2)) )/alpha
    a = np.array([spl2amp(lst[i]) for i in range(resolution)])
    a = np.nan_to_num(a)
    return a, fs, c


BITRATE=44100

def generate_wav(lst):
    amp,freq,c = decompress(lst)
    frame = np.arange(BITRATE*0.5)
    data = np.zeros_like(frame).astype(np.float32)
    for i in range(len(amp)):
        damp = np.exp(-0.5*c[i]*frame/BITRATE)
        sin = np.sin(freq[i]*2*math.pi*frame/BITRATE)
        data += amp[i]*damp*sin
    #data = data / np.max(data)
    return data




def get_B(a,b,c,V):
    return np.array([
        [a[0],0   ,0   ,a[1],0   ,0   ,a[2],0   ,0   ,a[3],0   ,0   ],
        [0   ,b[0],0   ,0   ,b[1],0   ,0   ,b[2],0   ,0   ,b[3],0   ],
        [0   ,0   ,c[0],0   ,0   ,c[1],0   ,0   ,c[2],0   ,0   ,c[3]],
        [b[0],a[0],0   ,b[1],a[1],0   ,b[2],a[2],0   ,b[3],a[3],0   ],
        [0   ,c[0],b[0],0   ,c[1],b[1],0   ,c[2],b[2],0   ,c[3],b[3]],
        [c[0],0   ,a[0],c[1],0   ,a[1],c[2],0   ,a[2],c[3],0   ,a[3]]
    ])/(6*V)

def get_E(E0,v):
    return np.array([
        [1-v ,v   ,v   ,0    ,0    ,0    ],
        [v   ,1-v ,v   ,0    ,0    ,0    ],
        [v   ,v   ,1-v ,0    ,0    ,0    ],
        [0   ,0   ,0   ,0.5-v,0    ,0    ],
        [0   ,0   ,0   ,0    ,0.5-v,0    ],
        [0   ,0   ,0   ,0    ,0    ,0.5-v]
    ])*E0/(1+v)/(1-2*v)

def get_abcV(x,y,z):
    a = np.zeros(4)
    b = np.zeros(4)
    c = np.zeros(4)
    a[0]=y[1]*(z[3] - z[2])-y[2]*(z[3] - z[1])+y[3]*(z[2] - z[1])
    a[1]=-y[0]*(z[3] - z[2])+y[2]*(z[3] - z[0])-y[3]*(z[2] - z[0])
    a[2]=y[0]*(z[3] - z[1])-y[1]*(z[3] - z[0])+y[3]*(z[1] - z[0])
    a[3]=-y[0]*(z[2] - z[1])+y[1]*(z[2] - z[0])-y[2]*(z[1] - z[0])
    b[0]=-x[1]*(z[3] - z[2])+x[2]*(z[3] - z[1])-x[3]*(z[2] - z[1])
    b[1]=x[0]*(z[3] - z[2])-x[2]*(z[3] - z[0])+x[3]*(z[2] - z[0])
    b[2]=-x[0]*(z[3] - z[1])+x[1]*(z[3] - z[0])-x[3]*(z[1] - z[0])
    b[3]=x[0]*(z[2] - z[1])-x[1]*(z[2] - z[0])+x[2]*(z[1] - z[0])
    c[0]=x[1]*(y[3] - y[2])-x[2]*(y[3] - y[1])+x[3]*(y[2] - y[1])
    c[1]=-x[0]*(y[3] - y[2])+x[2]*(y[3] - y[0])-x[3]*(y[2] - y[0])
    c[2]=x[0]*(y[3] - y[1])-x[1]*(y[3] - y[0])+x[3]*(y[1] - y[0])
    c[3]=-x[0]*(y[2] - y[1])+x[1]*(y[2] - y[0])-x[2]*(y[1] - y[0])
    V=((x[1] - x[0])*((y[2] - y[0])*(z[3] - z[0])-(y[3] - y[0])*(z[2] - z[0]))+(y[1] - y[0])*((x[3] - x[0])*(z[2] - z[0])-(x[2] - x[0])*(z[3] - z[0]))+(z[1] - z[0])*((x[2] - x[0])*(y[3] - y[0])-(x[3] - x[0])*(y[2] - y[0])))/6
    return a,b,c,abs(V)
    
def get_Me(d, v):
    return get_M(d)*v

def get_M(d):
    m =  np.array([
                    [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
                    [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
                    [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1],
                    [0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1],
                    [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
                ],dtype=np.float)
    return (m + m.T)*d/20

