import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import helper

from .layer import *

class EnvelopeNet_dilation(nn.Module):
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

        self.end_line = nn.Sequential(
                        nn.ConvTranspose1d(64,24,kernel_size=8), nn.BatchNorm1d(24), Bottleneck(24,24),Bottleneck(24,24),
                        nn.ConvTranspose1d(24,12,kernel_size=2,stride=2), nn.BatchNorm1d(12), Bottleneck(12,12),Bottleneck(12,12),
                        nn.ConvTranspose1d(12,6,kernel_size=2,stride=2), nn.BatchNorm1d(6), Bottleneck(6,6),Bottleneck(6,6),
                        nn.ConvTranspose1d(6,3,kernel_size=2,stride=2), nn.Sigmoid()
                        )

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

class EnvelopeNet_origin(nn.Module):
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
        self.deconv_layer1 = self._make_deconv_layer(64,2)
        self.deconv_layer2 = self._make_deconv_layer(64,2)

        self.end_line = nn.Sequential(
                        nn.ConvTranspose1d(64,24,kernel_size=8), nn.BatchNorm1d(24), Bottleneck(24,24),Bottleneck(24,24),
                        nn.ConvTranspose1d(24,12,kernel_size=2,stride=2), nn.BatchNorm1d(12), Bottleneck(12,12),Bottleneck(12,12),
                        nn.ConvTranspose1d(12,6,kernel_size=2,stride=2), nn.BatchNorm1d(6), Bottleneck(6,6),Bottleneck(6,6),
                        nn.ConvTranspose1d(6,3,kernel_size=2,stride=2), nn.Sigmoid()
                        )

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

  

EnvelopeNet = EnvelopeNet_origin