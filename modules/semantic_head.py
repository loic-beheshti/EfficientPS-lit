import torch.nn as nn
import torch
from inplace_abn import ABN

from .conv_utils import (
    iABNConv1dBlock,
    iABNSeparableConvBlock
)

class DPC(nn.Module):
    def __init__(self):
        super().__init__()
        
        self._1x6_conv = iABNSeparableConvBlock(dilation=(1,6))
        self._1x1_conv = iABNSeparableConvBlock(dilation=(1,1))
        self._6x21_conv = iABNSeparableConvBlock(dilation=(6,21))
        self._18x15_conv = iABNSeparableConvBlock(dilation=(18,15))
        self._6x3_conv = iABNSeparableConvBlock(dilation=(6,3))
        self._final_conv1d = iABNConv1dBlock(in_channels=1280, out_channels=128)

    def forward(self, inputs):
        x = inputs
        c_1 = self._1x6_conv(x)
        c_2 = self._1x1_conv(c_1)
        c_3 = self._6x21_conv(c_1)
        c_4 = self._18x15_conv(c_1)
        c_5 = self._6x3_conv(c_4)
        out = torch.cat((c_1, c_2, c_3, c_4, c_5), 1)
        out = self._final_conv1d(out)
        return out

class LFSE(nn.Module):
    def __init__(self):
        super().__init__()
        self._first_conv = iABNSeparableConvBlock(out_channels=128)
        self._second_conv = iABNSeparableConvBlock(in_channels=128, out_channels=128)

    def forward(self, inputs):
        x = inputs
        x = self._first_conv(x)
        x = self._second_conv(x)
        return x

class MCModule(nn.Module):
    def __init__(self):
        super().__init__()
        self._first_conv = iABNSeparableConvBlock(in_channels=128, out_channels=128) # not matching with paper to be verified with the author expecting 256
        self._second_conv = iABNSeparableConvBlock(in_channels=128, out_channels=128)
        self._up_sample = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, inputs):
        x = inputs
        x = self._first_conv(x)
        x = self._second_conv(x)
        x = self._up_sample(x)
        return x

class SemanticSegHead(nn.Module):
    def __init__(self):
        super().__init__()
        self._dpc_1 = DPC()
        self._dpc_2 = DPC()
        self._lfse_1 = LFSE()
        self._lfse_2 = LFSE()
        self._mc_1 = MCModule()
        self._mc_2 = MCModule()
        self._mc_3 = MCModule()
        self._up_sample_1 = nn.Upsample(scale_factor=8, mode='bilinear')
        self._up_sample_2 = nn.Upsample(scale_factor=4, mode='bilinear')
        self._up_sample_3 = nn.Upsample(scale_factor=2, mode='bilinear')
        self._last_conv = iABNConv1dBlock(in_channels=512, out_channels=30)
        self._last_up_sample = nn.Upsample(scale_factor=4, mode='bilinear')

    def forward(self, inputs):
        p32, p16, p8, p4 = inputs
        c_1 = self._dpc_1(p32)
        c_2 = self._dpc_2(p16) + self._mc_1(c_1)
        c_3 = self._lfse_1(p8) + self._mc_2(c_2)
        c_4 = self._lfse_2(p4) + self._mc_3(c_3)
        
        c_1 = self._up_sample_1(c_1)
        c_2 = self._up_sample_2(c_2)
        c_3 = self._up_sample_3(c_3)
        
        conc_feats = torch.cat((c_1, c_2, c_3, c_4), 1)
        out = self._last_conv(conc_feats)
        out = self._last_up_sample(out)
        
        return out