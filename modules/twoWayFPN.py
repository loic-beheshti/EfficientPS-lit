import torch.nn as nn
import torch
from inplace_abn import ABN

from .conv_utils import (
    MaxPool2dStaticSamePadding,
    iABNConv1dBlock,
    iABNSeparableConvBlock,  
)

class DualFPN(nn.Module):
    """
    2 way FPN for EfficientPS, currently fixed for efficientNet b-5
    """
    def __init__(self):
        super().__init__()

        self._conv1_up = iABNConv1dBlock(in_channels=40)
        self._conv2_up = iABNConv1dBlock(in_channels=64)
        self._conv3_up = iABNConv1dBlock(in_channels=176)
        self._conv4_up = iABNConv1dBlock(in_channels=2048)
        self._conv1_down = iABNConv1dBlock(in_channels=40)
        self._conv2_down = iABNConv1dBlock(in_channels=64)
        self._conv3_down = iABNConv1dBlock(in_channels=176)
        self._conv4_down = iABNConv1dBlock(in_channels=2048)

        self._upsample_1 = nn.Upsample(scale_factor=2, mode='nearest')
        self._upsample_2 = nn.Upsample(scale_factor=2, mode='nearest')
        self._upsample_3 = nn.Upsample(scale_factor=2, mode='nearest')
        
        self._downsample_1 = MaxPool2dStaticSamePadding(3, 2)
        self._downsample_2 = MaxPool2dStaticSamePadding(3, 2)
        self._downsample_3 = MaxPool2dStaticSamePadding(3, 2)
        
        self._p32_conv = iABNSeparableConvBlock()
        self._p16_conv = iABNSeparableConvBlock()
        self._p8_conv = iABNSeparableConvBlock()
        self._p4_conv = iABNSeparableConvBlock()

    def forward(self, inputs):
        b1, b2, b3, b4 = inputs
        """
        left to right fpn
        """
        b1_up = self._conv1_up(b1)
        b2_up = self._conv2_up(b2) + self._downsample_1(b1_up)
        b3_up = self._conv3_up(b3) + self._downsample_2(b2_up)
        b4_up = self._conv4_up(b4) + self._downsample_3(b3_up)

        """
        right to left fpn
        """
        b4_down = self._conv4_down(b4)
        b3_down = self._conv3_down(b3) + self._upsample_1(b4_down)
        b2_down = self._conv2_down(b2) + self._upsample_2(b3_down)
        b1_down = self._conv1_down(b1) + self._upsample_3(b2_down)

        """
        p32 to p4 extraction
        """
        p32 = self._p32_conv(b4_up + b4_down)
        p16 = self._p16_conv(b3_up + b3_down)
        p8 = self._p8_conv(b2_up + b2_down)
        p4 = self._p4_conv(b1_up + b1_down)
        return [p32, p16, p8, p4]