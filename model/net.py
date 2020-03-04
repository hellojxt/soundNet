import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import helper

from model.layer import *


class EnvelopeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.inplanes = 16
        self.pre_process = nn.Sequential(*[
            nn.Conv3d(1, self.inplanes, kernel_size=3, stride=2, padding=1,bias=False),
            nn.BatchNorm3d(self.inplanes),
            nn.ReLU(inplace=True)
        ])
        self.layer1 = self._make_layer(32, 2)
        self.jump1 = Residual(self.inplanes, 64)
        self.layer2 = self._make_layer(32, 2, stride=2)
        self.jump2 = Residual(self.inplanes, 64)
        self.layer3 = self._make_layer(64, 2, stride=2)
        self.deconv_layer1 = self._make_deconv_layer(64,1)
        self.deconv_layer2 = self._make_deconv_layer(64,1)
        mid_channels = 128
        self.end_line = nn.Sequential(*(
                        [FC(self.inplanes,mid_channels)] +
                        [FC(mid_channels,mid_channels)]*3 +  
                        [FC(mid_channels, helper.resolution*3)]
                        ))

    def _make_layer(self, planes, n, stride = 1):
        layers = [Residual(self.inplanes, planes, stride=stride)]
        self.inplanes = planes
        for i in range(1, n):
            layers.append(Residual(self.inplanes, planes))
        return nn.Sequential(*layers)

    def _make_deconv_layer(self, planes, n):
        layers = [nn.ConvTranspose3d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=4,stride=2,padding=1
                ),
                nn.BatchNorm3d(planes),
                nn.ReLU(inplace=True)]
        self.inplanes = planes
        for _ in range(1, n):
            layers.append(Residual(self.inplanes, planes))
        return nn.Sequential(*layers)

    def body(self, x):
        x = x.unsqueeze(1)
        x = self.pre_process(x)
        x = self.layer1(x)
        r1 = self.jump1(x)
        x = self.layer2(x)
        r2 = self.jump2(x)
        x = self.layer3(x)
        x = self.deconv_layer1(x) + r2
        x = self.deconv_layer2(x) + r1
        return x

    def forward(self, x, index):
        x = self.body(x)
        x = x.permute(0,2,3,4,1)[index]
        return self.end_line(x)

    def select(self, x, p1, p2, p3):
        x = self.body(x)
        x = x[:,:,p1,p2,p3]
        return self.end_line(x)

    # def forward(self, data, loss_fun):
    #     inputs = data['vox'].cuda()
    #     target = data['target'].cuda()
    #     freq = data['freq'].cuda()
    #     freq = freq.view(freq.size(0),self.res,helper.resolution//self.res).mean(-1)
    #     x = inputs
    #     out = []
    #     index = (target != 0).any(-1)
    #     target = target[index]

    #     x,g = self.predict(x)

    #     x = x.permute(0,2,3,4,1)[index]
    #     x = self.end_line(x)

    #     out.append(inputs)
    #     out.append(index)
    #     out.append(freq)
    #     out.append(g)
    #     out.append(x)
        
    #     loss1 = loss_fun(x,target)
    #     loss2 = loss_fun(g,freq)
    #     loss =  loss1 + loss2
    #     return out,target,loss,loss1


class FrequencyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.inplanes = 8
        self.res = 10
        self.layer1 = nn.Sequential(*[
                            nn.Conv3d(1, self.inplanes, kernel_size=3, stride=2, padding=1,bias=False),
                            nn.BatchNorm3d(self.inplanes),
                            nn.ReLU(inplace=True)
                        ])
        self.layer2 = self._make_layer(8, 2)
        self.layer3 = self._make_layer(16, 2, stride=2)
        self.layer4 = self._make_layer(32, 2, stride=2)
        self.layer5 = self._make_layer(64, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.fcs = nn.Sequential(*(
                            [FC(64,128)] + 
                            [FC(128,128)]*3 + 
                             [FC(128,self.res)])
                        )
    def _make_layer(self, planes, n, stride = 1):
        layers = [Residual(self.inplanes, planes, stride=stride)]
        self.inplanes = planes
        for i in range(1, n):
            layers.append(Residual(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.avgpool(x)
        x = x.view(x.size(0),-1)
        x = self.fcs(x)
        return x

  
# def show_spectrum(inputs, targets, img_name,dim = None):
#     import random 
     
#     index = np.argwhere(inputs[1].cpu().numpy()==True)
#     voxs = nn.MaxPool3d(2)(inputs[0]).cpu().numpy().reshape(-1,16,16,16)
#     plt.figure(figsize=(15,15))
#     width = 4
#     for i,idx in enumerate(random.sample(range(len(targets)),5)):
#         plt.subplot(5,width,i*width + 1)
#         plt.plot(targets[idx].cpu().numpy())
#         plt.subplot(5,width,i*width + 2)
#         plt.plot(inputs[-1][idx].cpu().numpy())
#         colors = np.empty(voxs[0].shape, dtype=object)
#         colors[...] = '#FFD65D05'
#         colors[(*index[idx][1:],)] = 'red'
#         voxs_ = np.zeros_like(voxs[index[idx][0]])
#         voxs_[(*index[idx][1:],)] = 1
#         ax = plt.subplot(5,width,i*width + 4,projection='3d')
#         ax.voxels(voxs[index[idx][0]],facecolors=colors, edgecolors='#0f0f0f10')
#         ax.voxels(voxs_,facecolors=colors, edgecolors='#0f0f0f10')
#     plt.savefig(img_name)
#     plt.clf()
#     plt.close()
#     i = 0
#     plt.figure(figsize=(10,6))
#     for f,g in zip(inputs[2].cpu().numpy(), inputs[3].cpu().numpy()):
#         plt.subplot(5,2,i*2 + 1)
#         plt.plot(f)
#         plt.subplot(5,2,i*2 + 2)
#         plt.plot(g)
#         i += 1
#         if i == 4:
#             break
#     plt.savefig(img_name+'rate.png')
#     plt.clf()
#     plt.close()
