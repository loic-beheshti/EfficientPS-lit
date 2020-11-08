import torch.nn as nn
import torch
import torchvision
import sys
import numpy as np

class PanopticFusion:

    def __init__(self):
        self._sem_pred = []
        self._sem_thing = []
        self._sem_stuff = []
        self._inst_masks = []
        self._inst_scores = []
        self._inst_boxes = []
        self._inst_labels = []
        self._canvas = []

    def set_instAndSeg(self, semantic_out, instance_out, clip_sem = 0.5):
        self._sem_pred = torch.argmax(semantic_out, dim=0)
        self._sem_stuff, self._sem_thing = torch.split(semantic_out, [11, 8])
        self._inst_masks = instance_out["masks"].cpu()
        self._inst_scores = instance_out["scores"].cpu()
        self._inst_boxes = instance_out["boxes"].cpu()
        self._inst_labels = instance_out["labels"].cpu()
        self._canvas = torch.zeros(semantic_out.size(), dtype=torch.uint8).cpu()
  
    # FL = (σ(MLA) +σ(MLB))(MLA +MLA) with mls = mask from instance head and mlb mask from semantic head
    def fused_mask_logits(self, mask_inst, mask_sem):
        mask_inst = mask_inst[0]
        FL = (torch.sigmoid(mask_inst)+torch.sigmoid(mask_sem))*(mask_inst+mask_inst)
        return FL

    def fusion(self, inputs = None):
        fl_list = []
        for i, box in enumerate(self._inst_boxes):
            box = box.int()

            if self._inst_labels[i] >= 8:
                continue 
                #self._inst_labels[i] = 1 # to debug 
            
            sem_clipped = torch.zeros(self._sem_stuff[0].size())
            sem_clipped[box[1]:box[3], box[0]:box[2]] = self._sem_thing[self._inst_labels[i]][box[1]:box[3], box[0]:box[2]]
            fl = self.fused_mask_logits(self._inst_masks[i], sem_clipped)
            fl_list.append(fl)
        
        # tensor from list of fused things things
        if fl_list != []: 
            fl_t = torch.stack(fl_list)

            # Identify the winner between things and stuffs
            inter_pred = torch.argmax(torch.cat((self._sem_stuff, fl_t), 0), dim=0)

            # keep only things and fill canvas
            self._canvas = self._canvas.numpy()
            inter_pred = inter_pred.numpy()
            inter_pred = np.where(inter_pred > self._sem_stuff.size()[0], inter_pred, 0)
            self._canvas = self._canvas + inter_pred # want to avoid 0 being stuff 
            
            # Fill empty elements with sem_head results
            self._canvas = np.where(self._canvas > 0, self._canvas, self._sem_pred)
        return self._canvas

"""
print(FL.size())
print("mask_inst, mask_sem ====", mask_inst.size(), mask_sem.size())
print(box, self._inst_labels[i])
print("LABELS  ====================", self.inst_labels[i].item())
print("SEM clipped  ====================", self.sem_thing[self.inst_labels[i]].size())
print("boxes  ====================", box[1].item(), box[3].item(), box[0].item(), box[2].item())
print("sem ====== ", self.sem_stuff.size()[0])
print("=====================================", self.sem_stuff.size(), fl_t.size())
"""