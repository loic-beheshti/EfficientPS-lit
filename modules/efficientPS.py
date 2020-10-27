import torch.nn as nn
import torch
from inplace_abn import ABN
from .efficientnet import EfficientNet
from .twoWayFPN import DualFPN
from .semantic_head import SemanticSegHead
from .efficientnet_utils import get_model_params

class EfficientPS(nn.Module):
    def __init__(self, num_classes=30, effNet_name = 'efficientnet-b5',  **kwargs):
        super(EfficientPS, self).__init__()

        self.num_classes = num_classes
        
        blocks_args, global_params = get_model_params(effNet_name, override_params={'num_classes': num_classes}) # tbm (1.6, 2.2, 456, 0.4)

        self.backbone_net = EfficientNet(blocks_args, global_params)

        self.dualfpn = DualFPN()

        self.semseg_net = SemanticSegHead()

    def freeze_bn(self):
        for m in self.modules():
            #if isinstance(m, nn.BatchNorm2d):
            if isinstance(m, ABN):
                m.eval()

    def forward(self, inputs):

        p1, p2, p3, p4, p5, p6, p7, p8, p9 = self.backbone_net(inputs)

        #There is a difference between the levels written in the paper and the level in the graph
        #In this example I chose to trust figure 2 and get (p3, p4, p6, p9) instead of (p2,p3,p5,p9)
        features = (p3, p4, p6, p9) 

        features = self.dualfpn(features)

        seg_out = self.semseg_net(features)
        print("seg_out = ", seg_out.size())

        return seg_out