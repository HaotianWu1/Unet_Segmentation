from torch import nn
import torch
import torchvision
import numpy as np
import torch
import torchvision.transforms
from torch.utils.data import Dataset
import os
import cv2
from torch.utils.data import DataLoader
from PIL import Image
import gif2numpy

def crop_img(origin, target):
    target_size = target.size()[2]
    origin_size = origin.size()[2]
    delta = origin_size - target_size
    delta = delta // 2
    return origin[:,:,delta:origin_size-delta, delta:origin_size-delta]

def double_conv(in_cha, out_cha):
    conv_layer = nn.Sequential(
        nn.Conv2d(in_cha, out_cha, 3, 1, 1),
        nn.BatchNorm2d(out_cha),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_cha, out_cha, 3, 1, 1),
        nn.BatchNorm2d(out_cha),
        nn.ReLU(inplace=True)
    )
    return conv_layer

def up_conv(in_cha, out_cha):
    conv_layer = nn.Sequential(
        nn.ConvTranspose2d(in_cha, out_cha, kernel_size=2, stride=2),
        nn.BatchNorm2d(out_cha),
        nn.ReLU(inplace=True)
    )
    return conv_layer






class Unet(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(Unet, self).__init__()
        self.down_conv_1 = double_conv(input_channels, 64)
        self.down_conv_2 = double_conv(64, 128)
        self.down_conv_3 = double_conv(128, 256)
        self.down_conv_4 = double_conv(256, 512)
        self.down_conv_5 = double_conv(512, 1024)
        self.maxpool2x2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.up_trans_6 = up_conv(1024, 512)
        self.conv_6 = double_conv(1024, 512)

        self.up_trans_7 = up_conv(512, 256)
        self.conv_7 = double_conv(512, 256)

        self.up_trans_8 = up_conv(256, 128)
        self.conv_8 = double_conv(256, 128)

        self.up_trans_9 = up_conv(128, 64)
        self.conv_9 = double_conv(128, 64)


        self.conv_10 = nn.Conv2d(in_channels=64,out_channels=output_channels,kernel_size=1)

    def forward(self, image):
        # encoder part
        x1 = self.down_conv_1(image)

        x2 = self.maxpool2x2(x1)
        x3 = self.down_conv_2(x2)

        x4 = self.maxpool2x2(x3)
        x5 = self.down_conv_3(x4)

        x6 = self.maxpool2x2(x5)
        x7 = self.down_conv_4(x6)

        x8 = self.maxpool2x2(x7)
        x9 = self.down_conv_5(x8)

        # decoder part
        x10 = self.up_trans_6(x9)
        x11 = torch.cat([x10, x7], dim=1)
        x12 = self.conv_6(x11)

        x13 = self.up_trans_7(x12)
        x14 = torch.cat([x13, x5], dim=1)
        x15 = self.conv_7(x14)

        x16 = self.up_trans_8(x15)
        x17 = torch.cat([x16, x3], dim=1)
        x18 = self.conv_8(x17)

        x19 = self.up_trans_9(x18)
        x20 = torch.cat([x19, x1], dim=1)
        x21 = self.conv_9(x20)


        x = self.conv_10(x21)
        return x








if __name__ == "__main__":
    label = np.array(Image.open("data/test_masks/0cdf5b5d0ce1_01_mask.gif").convert("L"), dtype=np.float32)
    label[label == 255.0] = 1.0
    print(label.shape)













