import torch.nn as nn
import torch
import torch.nn.functional as F
from inplace_abn import ABN
from .efficientnet import EfficientNet
from .twoWayFPN import DualFPN
from .semantic_head import SemanticSegHead
from .efficientnet_utils import get_model_params
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import Cityscapes
import torchvision.transforms as transforms


class EfficientPS(pl.LightningModule):
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
        print("inputs = ", inputs.size())
        p1, p2, p3, p4, p5, p6, p7, p8, p9 = self.backbone_net(inputs)

        #There is a difference between the levels written in the paper and the level in the graph
        #In this example I chose to trust figure 2 and get (p3, p4, p6, p9) instead of (p2,p3,p5,p9)
        features = (p3, p4, p6, p9) 

        features = self.dualfpn(features)

        seg_out = self.semseg_net(features)
        print("seg_out = ", seg_out.size())

        return seg_out
    
    def training_step(self, batch, batch_nb):
        img, mask = batch
        img = img.float()
        mask = mask.long()
        out = self(img)
        loss_val = F.cross_entropy(out, mask) # ignore_index=250
        log_dict = {'train_loss': loss_val}
        return {'loss': loss_val, 'log': log_dict, 'progress_bar': log_dict}

    def validation_step(self, batch, batch_idx):
        img, mask = batch
        img = img.float()
        mask = mask.long()
        out = self(img)
        loss_val = F.cross_entropy(out, mask) # ignore_index=250
        return {'val_loss': loss_val}

    def validation_epoch_end(self, outputs):
        loss_val = torch.stack([x['val_loss'] for x in outputs]).mean()
        log_dict = {'val_loss': loss_val}
        return {'log': log_dict, 'val_loss': log_dict['val_loss'], 'progress_bar': log_dict}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def train_dataloader(self):
        # use default transform
        return DataLoader(Cityscapes('./data/Cityscapes', split='train', mode='fine', target_type='semantic', transform=transforms.ToTensor(), target_transform=transforms.ToTensor()), batch_size=32)

    def val_dataloader(self):
        return DataLoader(Cityscapes('./data/Cityscapes', split='val', mode='fine', target_type='semantic', transform=transforms.ToTensor(), target_transform=transforms.ToTensor()), batch_size=32)