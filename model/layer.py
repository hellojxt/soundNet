import torch
import torch.nn as nn
import torch.nn.functional as F

class Residual(nn.Module):
    def __init__(self, input_channels, out_channels,stride = 1,use_bn = True):
        '''
        in -> out/2 -> out 
        
        ↓_______________↑

        output = ac(bn(conv(input)))
        '''
        super().__init__()
        mid_channels = out_channels // 2
        self.use_bn = use_bn
        self.out_channels   = out_channels
        self.input_channels = input_channels
        self.mid_channels   = mid_channels
        self.down_channel = nn.Conv3d(input_channels, self.mid_channels, kernel_size = 1)
        self.AcFunc       = nn.ReLU(inplace=True)
        if use_bn:
            self.bn_0 = nn.BatchNorm3d(num_features = self.mid_channels)
            self.bn_1 = nn.BatchNorm3d(num_features = self.mid_channels)
            self.bn_2 = nn.BatchNorm3d(num_features = self.out_channels)

        self.conv = nn.Conv3d(self.mid_channels, self.mid_channels, kernel_size = 3, padding = 1, stride = stride)

        self.up_channel = nn.Conv3d(self.mid_channels, out_channels, kernel_size= 1)

        self.trans = None
        if input_channels != out_channels or stride != 1:
            self.trans = nn.Conv3d(input_channels, out_channels, kernel_size = 1, stride = stride)
    
    def forward(self, inputs):
        x = self.down_channel(inputs)
        if self.use_bn:
            x = self.bn_0(x)
        x = self.AcFunc(x)

        x = self.conv(x)
        if self.use_bn:
            x = self.bn_1(x)
        x = self.AcFunc(x)

        x = self.up_channel(x)

        if self.trans != None:
            x += self.trans(inputs)
        else:
            x += inputs

        if self.use_bn:
            x = self.bn_2(x)
        
        return self.AcFunc(x)



class FC(nn.Module):
    def __init__(self, inplanes, outplanes , active=nn.ReLU()):
        '''
        output = bn(ac(linear(input)))
        '''
        super().__init__()
        self.nn = nn.Linear(inplanes,outplanes)
        self.bn = nn.BatchNorm1d(outplanes)
        self.active = active

    def forward(self, x):
        return self.active(self.bn(self.nn(x)))



class Bottleneck(nn.Module):
    expansion = 2
    def __init__(self, in_planes, out_planes, stride = 1, active=nn.ReLU()):
        super(Bottleneck, self).__init__()
        self.active = active
        planes = out_planes // self.expansion
        self.bn1 = nn.BatchNorm1d(planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.bn3 = nn.BatchNorm1d(out_planes)
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=1, bias=False)
        if stride == 1:
            self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, padding=1, bias=False)
        else:
            self.conv2 = nn.ConvTranspose1d(planes, planes,
                                            kernel_size=3,
                                            stride=stride, bias=False,
                                            padding=1,
                                            output_padding=1)

        self.conv3 = nn.Conv1d(planes, out_planes, kernel_size=1, bias=False)

        self.shortcut = nn.Sequential()

        if in_planes != out_planes:
            if stride == 1:
                self.shortcut = nn.Sequential(
                    nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=1, bias=False),
                    nn.BatchNorm1d(out_planes)
                )
            else:
                self.shortcut = nn.Sequential(
                    nn.ConvTranspose1d(in_planes, 
                               out_planes,
                               kernel_size=1, stride=stride,
                               bias=False, output_padding=1),
                    nn.BatchNorm1d(out_planes)
                )
            

    def forward(self, x):
        out = self.active(self.bn1(self.conv1(x)))
        out = self.active(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = self.active(out)
        return out