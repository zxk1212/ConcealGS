
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from torch import nn
import torch
from torchvision import models
import torchvision
from torch.nn import functional as F

import pdb


def conv3x3(in_, out):
    return nn.Conv2d(in_, out, 3, padding=1)


class ConvRelu(nn.Module):
    def __init__(self, in_: int, out: int):
        super(ConvRelu, self).__init__()
        self.conv = conv3x3(in_, out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x
    

class DecoderBlock(nn.Module):
    """
    Paramaters for Deconvolution were chosen to avoid artifacts, following
    link https://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, middle_channels, out_channels, is_deconv=True):
        super(DecoderBlock, self).__init__()
        self.in_channels = in_channels

        if is_deconv:
            self.block = nn.Sequential(
                ConvRelu(in_channels, middle_channels),
                nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=4, stride=2,
                                   padding=1),
                nn.ReLU(inplace=True)
            )
        else:
            self.block = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear'),
                ConvRelu(in_channels, middle_channels),
                ConvRelu(middle_channels, out_channels),
            )

    def forward(self, x):
        return self.block(x)
    

class VGG16UNet_64_2(nn.Module):
    def __init__(self, in_channel=3, out_channel=3, num_filters=32, pretrained=True, requires_grad=True,args=None):

        super().__init__()
        self.out_channel = out_channel

        self.pool = nn.MaxPool2d(2, 2)

        self.encoder = torchvision.models.vgg16(pretrained=pretrained).features

        self.relu = nn.ReLU(inplace=True)

        self.args = args

        self.wm_resize = 64
    
        self.encoder_conv1_1=nn.Conv2d(in_channel, 64, 3, padding=(1, 1))#
        self.conv1 = nn.Sequential(
            self.encoder_conv1_1,
            self.relu,
            self.encoder[2],
            self.relu
        )

        self.conv2 = nn.Sequential(
            self.encoder[5],
            self.relu,
            self.encoder[7],
            self.relu
        )

        self.conv3 = nn.Sequential(
            self.encoder[10],
            self.relu,
            self.encoder[12],
            self.relu,
            self.encoder[14],
            self.relu
        )

        self.conv4 = nn.Sequential(
            self.encoder[17],
            self.relu,
            self.encoder[19],
            self.relu,
            self.encoder[21],
            self.relu
        )

        self.center = DecoderBlock(512, num_filters * 8 * 2, num_filters * 8)

        self.dec4 = DecoderBlock(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8)
        self.dec3 = DecoderBlock(256 + num_filters * 8, num_filters * 4 * 2, num_filters * 2)
        self.dec2 = DecoderBlock(128 + num_filters * 2, num_filters * 2 * 2, num_filters)
        self.dec1 = ConvRelu(64 + num_filters, num_filters)
        # self.final = nn.Conv2d(num_filters, out_channel, kernel_size=5, stride=4, padding=(1, 1))
        self.final = nn.Conv2d(num_filters, out_channel, kernel_size=1)

        self.c_pool = nn.AdaptiveMaxPool2d((2,2))
        self.classifer_fc = nn.Sequential(
            nn.Linear(512*2*2,(512*2*2)//8),
            self.relu,
            nn.Linear((512*2*2)//8,1)
        )

        self.cov1 = nn.Conv2d(256, 1024, 3, stride=1, padding=(1, 1))
        # self.cov1 = nn.Conv2d(256, 512, 3, stride=2, padding=(1, 1))
        self.pool1 = nn.MaxPool2d(2, 2)

    def forward(self, x, return_c=False, cmask=False):

        conv1 = self.conv1(x) # torch.Size([2, 64, 756, 1008])
        conv2 = self.conv2(self.pool(conv1)) # torch.Size([2, 128, 378, 504])
        conv3 = self.conv3(self.pool(conv2)) # torch.Size([2, 256, 189, 252])
        conv4 = self.conv4(self.pool(conv3)) # torch.Size([2, 512, 94, 126])
        # conv5 = self.conv5(self.pool(conv4))

        # center = self.center(self.pool(conv5))\
        center = self.center(self.pool(conv4)) # torch.Size([2, 256, 94, 126]) center <> upsampling module 
        # pool(conv4) 1,512,3,3
        dec4 = self.dec4(torch.cat([center, conv4], 1)) # torch.Size([2, 256, 188, 252])

        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(torch.cat([dec2, conv1], 1))

        x_out=self.final(dec1)
        return x_out