# model

from torch import nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable

import math
from torch.nn import Parameter

class BottleneckBlock(nn.Module):
    def __init__(self,in_planes,out_planes,dropRate=0.0):
        #out_planes => growh_rate를 입력으로 받게 된다.
        super(BottleneckBlock,self).__init__()
        inter_planes = out_planes * 4 # bottleneck layer의 conv 1x1 filter chennel 수는 4*growh_rate이다.
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace = True)
        self.conv1 = nn.Conv2d(in_planes,inter_planes,kernel_size=1,stride=1,padding=0,bias=False)
        self.bn2 = nn.BatchNorm2d(inter_planes)
        self.conv2 = nn.Conv2d(inter_planes,out_planes,kernel_size=3,stride=1,padding=1,bias=False)
        self.droprate = dropRate
        
    def forward(self,x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate>0:
            out = F.dropout(out,p=self.droprate,inplace=False,training = self.training)
        out = self.conv2(self.relu(self.bn2(out)))
        if self.droprate>0:
            out = F.dropout(out,p=self.droprate,inplace=False,training = self.training)
        return torch.cat([x,out],1) # 입력으로 받은 x와 새로 만든 output을 합쳐서 내보낸다

class DenseBlock(nn.Module):
    def __init__(self,nb_layers,in_planes,growh_rate,block,dropRate=0.0):
        super(DenseBlock,self).__init__()
        self.layer = self._make_layer(block, in_planes, growh_rate, nb_layers, dropRate)
    
    def _make_layer(self,block,in_planes,growh_rate,nb_layers,dropRate):
        layers=[]
        for i in range(nb_layers):
            layers.append(block(in_planes + i*growh_rate ,growh_rate,dropRate))
            
        return nn.Sequential(*layers)
    
    def forward(self,x):
        return self.layer(x)

class TransitionBlock(nn.Module):
    def __init__(self,in_planes,out_planes,dropRate=0.0):
        super(TransitionBlock,self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes,out_planes,kernel_size=1,stride=1,padding=0,bias=False)
        self.droprate = dropRate
        
    def forward(self,x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate>0:
            out = F.dropout(out,p=self.droprate,inplace=False,training=self.training)
        return F.avg_pool2d(out, 2) # 2   4번

class ConvBlock(nn.Module):
    def __init__(self, inp, oup, k, s, p, dw=False, linear=False):
        super(ConvBlock, self).__init__()
        self.linear = linear
        if dw:
            self.conv = nn.Conv2d(inp, oup, k, s, p, groups=inp, bias=False)
        else:
            self.conv = nn.Conv2d(inp, oup, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(oup)
        if not linear:
            self.prelu = nn.PReLU(oup)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.linear:
            return x
        else:
            return self.prelu(x)

Mobilefacenet_bottleneck_setting = [
    # t, c , n ,s
    [2, 64, 5, 2],
    [4, 128, 1, 2],
    [2, 128, 6, 1],
    [4, 128, 1, 2],
    [2, 128, 2, 1]
]

class MobileFacenet(nn.Module):
    def __init__(self, bottleneck_setting=Mobilefacenet_bottleneck_setting):
        super(MobileFacenet, self).__init__()

        self.conv1 = ConvBlock(3, 64, 3, 2, 1)

        self.dw_conv1 = ConvBlock(64, 64, 3, 1, 1, dw=True)

        block = BottleneckBlock
        
        self.denseblock1 = DenseBlock(4,64,4,block,dropRate=0.2)
        # DenseBlock(n=layer개수,in_planes,growh_rate,block,dropRate)
        # growth rate = 4, dropRate=0.2 default

        self.trans1 = TransitionBlock(80, int(math.floor(80*1)),dropRate=0.2)

        self.denseblock2 = DenseBlock(4,80,4,block,dropRate=0.2)
        
        self.trans2 = TransitionBlock(96, int(math.floor(96*1)),dropRate=0.2)
        
        self.denseblock3 = DenseBlock(4,96,4,block,dropRate=0.2)
        
#         self.trans3 = TransitionBlock(112, int(math.floor(112*1)),dropRate=0.2)
        
#         self.denseblock4 = DenseBlock(4,112,4,block,dropRate=0.2)
        
#         self.trans4 = TransitionBlock(128, int(math.floor(128*1)),dropRate=0.2)
        
        self.conv2 = ConvBlock(112, 256, 1, 1, 0)
        # 여기 inp 맞춰줘야함
        
        self.linear7 = ConvBlock(256, 512, (7, 6), 1, 0, dw=True, linear=True)

        self.linear1 = ConvBlock(512, 128, 1, 1, 0, linear=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.dw_conv1(x)
        x = self.denseblock1(x)
        x = self.trans1(x)
        x = self.denseblock2(x)
        x = self.trans2(x)
        x = self.denseblock3(x)
        x = self.trans3(x)
#         x = self.denseblock4(x)
#         x = self.trans4(x)
        x = self.conv2(x)
        x = self.linear7(x)
        x = self.linear1(x)
        x = x.view(x.size(0), -1)

        return x


if __name__ == "__main__":
    input = torch.FloatTensor(2, 3, 112, 96)
    net = MobileFacenet()
    #print(net)
    x = net(input)
    print(x.shape)