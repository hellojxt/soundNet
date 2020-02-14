import torch.nn as nn

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
    def __init__(self, inplanes, outplanes , active=nn.Tanh()):
        '''
        output = bn(ac(linear(input)))
        '''
        super().__init__()
        self.nn = nn.Linear(inplanes,outplanes)
        self.bn = nn.BatchNorm1d(outplanes)
        self.active = active

    def forward(self, x):
        return self.bn(self.active(self.nn(x)))