import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import helper
from .layer import *

class envelope_encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.inplanes = 32
        self.pre_process = nn.Sequential(*[
            nn.Conv3d(1, self.inplanes, kernel_size=3, stride=2, padding=1,bias=False),
            nn.BatchNorm3d(self.inplanes),
            nn.ReLU()
        ])
        self.layer1 = self._make_layer(32, 2)
        self.jump1 = Residual(self.inplanes, 128)
        self.layer2 = self._make_layer(64, 2, stride=2)
        self.jump2 = Residual(self.inplanes, 128)
        self.layer3 = self._make_layer(128, 2, stride=2)
        self.deconv_layer1 = self._make_deconv_layer(128,1)
        self.deconv_layer2 = self._make_deconv_layer(128,1)
        self.mid_layer = self._make_layer(128,1)

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

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.pre_process(x)
        x = self.layer1(x)
        r1 = self.jump1(x)
        x = self.layer2(x)
        r2 = self.jump2(x)
        x = self.layer3(x)
        x = self.deconv_layer1(x) + r2
        x = self.deconv_layer2(x) + r1
        x = self.mid_layer(x)
        return x

class envelope_conv(nn.Module):
    def __init__(self):
        super().__init__()
        self.body =envelope_encoder()
        self.end_line = nn.Sequential(
                        Bottleneck(128,64,stride=2), Bottleneck(64,64),
                        Bottleneck(64,32,stride=2), Bottleneck(32,32),
                        Bottleneck(32,16,stride=2), Bottleneck(16,16),
                        nn.ConvTranspose1d(16,3,kernel_size=8,stride=8,bias=False)
                        )
    def forward(self, x, index):
        x = self.body(x)
        x = x.permute(0,2,3,4,1)[index]
        x = x.view(x.size(0),-1,1)
        x =  self.end_line(x)
        return x

    def select(self, x, coord):
        p1,p2,p3 = coord[0],coord[1],coord[2]
        x = self.body(x)
        x = x[:,:,p1,p2,p3]
        x = x.view(x.size(0),-1,1)
        x = self.end_line(x)
        return x


class frequency_encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.inplanes = 32
        self.layer1 = nn.Sequential(*[
                            nn.Conv3d(1, self.inplanes, kernel_size=3, stride=2, padding=1,bias=False),
                            nn.BatchNorm3d(self.inplanes),
                            nn.ReLU(inplace=True)
                        ])
        self.layer2 = self._make_layer(32, 2)
        self.layer3 = self._make_layer(64, 2, stride=2)
        self.layer4 = self._make_layer(128, 2, stride=2)
        self.layer5 = self._make_layer(256, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool3d(1)

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
        return x

class frequency_conv(nn.Module):
    def __init__(self, res):
        super().__init__()
        self.body = frequency_encoder()
        self.end_line = nn.Sequential(
                Bottleneck(256,128,stride=2), Bottleneck(128,128),
                Bottleneck(128,64,stride=2), Bottleneck(64,64),
                Bottleneck(64,32,stride=2), Bottleneck(32,32),
                nn.ConvTranspose1d(32,1,kernel_size=res//8,stride=res//8,bias=False)
                )
    def forward(self,x):
        x = self.body(x)
        x = x.view(x.size(0),-1,1)
        x = self.end_line(x)
        return x.view(x.size(0),-1)


class voxnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = frequency_encoder()
        self.end_line = nn.Linear(256,40)

    def forward(self,x):
        x = self.body(x)
        x = x.view(x.size(0),-1)
        return self.end_line(x)

