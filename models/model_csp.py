# sj_csp
# dw_conv1(x)이후에 x.clone() x_2

from torch import nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import math
from torch.nn import Parameter
#from utils import print_model_info

class Bottleneck(nn.Module):
    def __init__(self, inp, oup, stride, expansion):
        super(Bottleneck, self).__init__()
        self.connect = stride == 1 and inp == oup
        #
        self.conv = nn.Sequential(
            #pw
            nn.Conv2d(inp, inp * expansion, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expansion),
            nn.PReLU(inp * expansion),

            #dw
            nn.Conv2d(inp * expansion, inp * expansion, 3, stride, 1, groups=inp * expansion, bias=False),
            nn.BatchNorm2d(inp * expansion),
            nn.PReLU(inp * expansion),

            #pw-linear
            nn.Conv2d(inp * expansion, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup)
        )

    def forward(self, x):
        if self.connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

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
    [2, 32, 5, 2],
    [4, 64, 1, 2],
    [2, 64, 6, 1],
    [4, 64, 1, 1],
    [2, 64, 2, 1]
]

class MobileFacenet_csp(nn.Module):
    def __init__(self, bottleneck_setting=Mobilefacenet_bottleneck_setting):
        super(MobileFacenet_csp, self).__init__()

        self.conv1 = ConvBlock(3, 32, 3, 2, 1)

        self.dw_conv1 = ConvBlock(32, 32, 3, 1, 1, dw=True)

        self.inplanes = 32
        block = Bottleneck
        self.blocks = self._make_layer(block, bottleneck_setting)

        self.conv2 = ConvBlock(64, 256, 1, 1, 0)
        self.conv3 = ConvBlock(32, 256, 1, 2, 0)

        self.linear7 = ConvBlock(512, 512, (14, 12), 1, 0, dw=True, linear=True)

        self.linear1 = ConvBlock(512, 128, 1, 1, 0, linear=True)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, setting):
        layers = []
        for t, c, n, s in setting:
            for i in range(n):
                if i == 0:
                    layers.append(block(self.inplanes, c, s, t))
                else:
                    layers.append(block(self.inplanes, c, 1, t))
                self.inplanes = c

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.dw_conv1(x)

        x_2 = x.clone()
        
        x = self.blocks(x)
        x = self.conv2(x)

        x_2  = self.conv3(x_2) 
        x_2 = F.max_pool2d(x_2, 2)

        x = torch.cat((x,x_2), dim=1)
        
        x = self.linear7(x)
        x = self.linear1(x)
        x = x.view(x.size(0), -1)

        return x

# if __name__ == "__main__":
#     input = Variable(torch.FloatTensor(2, 3, 112, 96))
#     net = MobileFacenet()
#     # print_model_info(net,(3,112,96))
#     # print(net)
#     x = net(input)
#     print(x.shape)