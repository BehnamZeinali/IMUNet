"""
This is the 1-D  version of MobileNet
Original paper is "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"
Link: https://arxiv.org/abs/1704.04861

The implementation in https://hackmd.io/@machine-learning/rk-MSuYFU has been modified.

"""

import torch.nn as nn
import torch
from collections import OrderedDict
from torch.autograd import Variable


class DSConv(nn.Module):

    def __init__(self, f_3x3, f_1x1, stride=1, padding=0):
        super(DSConv, self).__init__()

        self.feature = nn.Sequential(OrderedDict([
            ('dconv', nn.Conv1d(f_3x3,
                                f_3x3,
                                kernel_size=3,
                                groups=f_3x3,
                                stride=stride,
                                padding=padding,
                                bias=False
                                )),
            ('bn1', nn.BatchNorm1d(f_3x3)),
            ('act1', nn.ReLU()),
            ('pconv', nn.Conv1d(f_3x3,
                                f_1x1,
                                kernel_size=1,
                                bias=False)),
            ('bn2', nn.BatchNorm1d(f_1x1)),
            ('act2', nn.ReLU())
        ]))

    def forward(self, x):
        out = self.feature(x)
        return out


class MobileNet(nn.Module):
    """
        MobileNet-V1 architecture for CIFAR-10.
    """

    def __init__(self, channels, width_multiplier=1.0, num_classes=2):
        super(MobileNet, self).__init__()

        channels = [int(elt * width_multiplier) for elt in channels]

        self.conv = nn.Sequential(OrderedDict([
            ('conv', nn.Conv1d(6, channels[0], kernel_size=3,
                               stride=2, padding=1, bias=False)),
            ('bn', nn.BatchNorm1d(channels[0])),
            ('act', nn.ReLU())
        ]))

        self.features = nn.Sequential(OrderedDict([
            ('dsconv1', DSConv(channels[0], channels[1], 1, 1)),
            ('dsconv2', DSConv(channels[1], channels[2], 2, 1)),
            ('dsconv3', DSConv(channels[2], channels[2], 1, 1)),
            ('dsconv4', DSConv(channels[2], channels[3], 2, 1)),
            ('dsconv5', DSConv(channels[3], channels[3], 1, 1)),
            ('dsconv6', DSConv(channels[3], channels[4], 2, 1)),
            ('dsconv7_a', DSConv(channels[4], channels[4], 1, 1)),
            ('dsconv7_b', DSConv(channels[4], channels[4], 1, 1)),
            ('dsconv7_c', DSConv(channels[4], channels[4], 1, 1)),
            ('dsconv7_d', DSConv(channels[4], channels[4], 1, 1)),
            ('dsconv7_e', DSConv(channels[4], channels[4], 1, 1)),
            ('dsconv8', DSConv(channels[4], channels[5], 2, 1)),
            ('dsconv9', DSConv(channels[5], channels[5], 1, 1))
        ]))

        # self.output = nn.Sequential(OrderedDict([
        #     ('conv_last', nn.Conv1d(1024,512,kernel_size=1,stride=1,bias=False)),
        #     ('bn_last', nn.BatchNorm1d(512) ),
        #     ('flaten', nn.Flatten() ),
        #     ('first_linear' ,  nn.Linear(3584, 1024)),
        #     ('act_1' , nn.ReLU(inplace= True)),
        #     ('drop_1' , nn.Dropout(0.5)),
        #     ('second_linear' ,  nn.Linear(1024, 128)),
        #     ('act_2' , nn.ReLU(inplace= True)),
        #     ('drop_2' , nn.Dropout(0.5)),
        #     ('third_linear' ,  nn.Linear(128, 2))

        # ]))

        self.avgpool = nn.AdaptiveAvgPool1d((1))
        self.linear = nn.Linear(channels[5], num_classes)

    def forward(self, x):
        out = self.conv(x)
        out = self.features(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.linear(out)
        # out = self.output(out)
        return out

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == '__main__':
    net = MobileNet(channels=[32, 64, 128, 256, 512, 1024], width_multiplier=1)
    print(net)
    x_image = Variable(torch.randn(1, 6, 200))
    y = net(x_image)
    print(y)
