import torch.nn as nn
import torch
import torch.nn.functional as F
from inplace_abn import ABN
from .efficientnet import EfficientNet
from .twoWayFPN import DualFPN
from .semantic_head import SemanticSegHead
from .efficientnet_utils import get_model_params
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional.classification import iou
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import Cityscapes
import torchvision.transforms as transforms
from datasets.cityscapes_transforms import cityscapesTransforms, save_samples_out, instmap_to_segmap
from .semantic_segloss import SemanticSegLoss
from .panopticFusion import PanopticFusion

import torchvision
from .InstanceSeg_head import MyMaskRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torch.jit.annotations import Tuple, List, Dict, Optional



class EfficientPS(pl.LightningModule):
    def __init__(self, num_classes=30, batch_size=2, effNet_name='efficientnet-b5',
            data_dir='./data/Cityscapes',**kwargs):
        super(EfficientPS, self).__init__()

        self.num_classes = num_classes
        self.batch_size = batch_size
        self.data_dir = data_dir
        
        blocks_args, global_params = get_model_params(effNet_name, override_params={'num_classes': num_classes}) # tbm (1.6, 2.2, 456, 0.4)

        self.backbone_net = EfficientNet(blocks_args, global_params)

        self.dualfpn = DualFPN()

        self.semseg_net = SemanticSegHead(num_classes=num_classes)

        self.seg_loss = SemanticSegLoss(ohem = 0.25)

        anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256),),
                                            aspect_ratios=((0.5, 1.0, 2.0),))

        roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0', '1', '2', '3'],
                                                        output_size=14,
                                                        sampling_ratio=2)

        mask_roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0', '1', '2', '3'],
                                                              output_size=14,
                                                              sampling_ratio=2)
        self.mask_rcnn = MyMaskRCNN([],
                          num_classes=19,
                          rpn_anchor_generator=anchor_generator,
                          box_roi_pool=roi_pooler,
                          mask_roi_pool=mask_roi_pooler)

        self.pf = PanopticFusion()        

    def freeze_bn(self):
        for m in self.modules():
            #if isinstance(m, nn.BatchNorm2d):
            if isinstance(m, ABN):
                m.eval()

    def forward(self, inputs, targets=None, no_targets=False):
        p1, p2, p3, p4, p5, p6, p7, p8, p9 = self.backbone_net(inputs)
        #del p1, p2, p5, p7, p8

        #There is a difference between the levels written in the paper and the level in the graph
        #In this example I chose to trust figure 2 and get (p3, p4, p6, p9) instead of (p2,p3,p5,p9)
        features = (p3, p4, p6, p9) 

        features = self.dualfpn(features)

        # have ato handle the case where there is no things elt in the input
        if no_targets == False:
            inst_losses = self.mask_rcnn([inputs[0]], features, targets)
        else: 
            inst_losses = []
        seg_out = self.semseg_net(features)

        return seg_out, inst_losses
    
    # Currently handle batch = 1 only
    def training_step(self, batch, batch_nb):
        img, labels = batch
        mask = labels[0]
        no_targets = False

        if labels[1] == []:
            no_targets=True
            #print("====================== no targets in things ======================")
            target = []
        else:
            target = [{'boxes': labels[1][0], 'masks': labels[2][0], 'labels': labels[3][0]}]
        
        img = img.float()
        mask = mask.long()

        out, inst_losses = self(img, target, no_targets=no_targets)

        loss_seg = self.seg_loss(out, mask)

        log_dict = {'seg_loss': loss_seg}
        if inst_losses != []:
            log_dict.update(inst_losses)

        loss = torch.sum(torch.stack(list(log_dict.values())))
        loss_t = {'loss_sum': loss}
        log_dict.update(loss_t)

        self.log_dict(log_dict, on_step=False, on_epoch=True, prog_bar=True)
        #self.log('loss', loss, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        img, labels = batch
        mask = labels[0]
        img = img.float()
        mask = mask.long()
        out_seg, out_rcnn = self(img)

        self.pf.set_instAndSeg(out_seg[0].cpu(), out_rcnn[0])
        canvas = self.pf.fusion()
        sem_head_pred = torch.argmax(out_seg[0], 0)

        if batch_idx == 0:
            save_samples_out((img[0]*255.).int().cpu().numpy(), sem_head_pred.cpu().numpy(), canvas, out_rcnn[0]["boxes"].cpu().numpy(), name=str(self.current_epoch)+"_out")
        
        val_miou_fusion = iou(mask[0].cpu(), torch.from_numpy(instmap_to_segmap(canvas)))
        val_miou_seg = iou(mask[0], sem_head_pred)

        self.log('val_miou_fusion', val_miou_fusion, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_miou_seg', val_miou_seg, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def train_dataloader(self):
        return DataLoader(Cityscapes(self.data_dir, split='train', mode='fine', target_type=['semantic', 'instance'], transforms=cityscapesTransforms(augmentation=False)), batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(Cityscapes(self.data_dir, split='val', mode='fine', target_type=['semantic', 'instance'], transforms=cityscapesTransforms()), batch_size=self.batch_size)