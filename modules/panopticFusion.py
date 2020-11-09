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

        self.fl_list = []
        self.label_list = []
        self.label_instances = []
        self.label_tracker = {}

    def set_instAndSeg(self, semantic_out, instance_out, clip_sem = 0.5):
        self._sem_pred = torch.argmax(semantic_out, dim=0)
        self._sem_stuff, self._sem_thing = torch.split(semantic_out, [11, 8])
        self._inst_masks = instance_out["masks"].cpu()
        self._inst_scores = instance_out["scores"].cpu()
        self._inst_boxes = instance_out["boxes"].cpu()
        self._inst_labels = instance_out["labels"].cpu()
        self._canvas = torch.zeros(semantic_out[0].size(), dtype=torch.long).cpu()
  
    # FL = (σ(MLA)+σ(MLB))(MLA+MLA) with mls = mask from instance head and mlb mask from semantic head
    def fused_mask_logits(self, mask_inst, mask_sem):
        mask_inst = mask_inst[0]
        FL = (torch.sigmoid(mask_inst)+torch.sigmoid(mask_sem))*(mask_inst+mask_inst)
        return FL

    # keep only things and fill canvas and convert to instance map format where
    # x < 1000 = semantics and x > 1000 = (label x //1000 and instance number x % 1000)
    def write_to_instmap(self, label, stuff_size = 11):
        inst = 0
        if label > stuff_size:
            label -= stuff_size
            inst = self.label_list[label]*1000 + self.label_instances[label]            
        return inst

    def fusion(self, inputs = None):
        self.fl_list = []
        self.label_list = []
        self.label_instances = []
        self.label_tracker = {}
        for i, box in enumerate(self._inst_boxes):
            box = box.int()
            label = self._inst_labels[i].item()

            if label >= 8:
                continue 
                #self._inst_labels[i] = 1 # to debug
                #label = 1
            
            
            sem_clipped = torch.zeros(self._sem_stuff[0].size())
            sem_clipped[box[1]:box[3], box[0]:box[2]] = self._sem_thing[label][box[1]:box[3], box[0]:box[2]]
            fl = self.fused_mask_logits(self._inst_masks[i], sem_clipped)
            
            if label not in self.label_list:
                self.label_instances.append(0)
                self.label_tracker[label] = 0
            else:
                self.label_tracker[label] += 1
                self.label_instances.append(self.label_tracker[label])
                
            self.label_list.append(label)
            self.fl_list.append(fl)
        
        self._canvas = self._canvas.numpy()

        # tensor from list of fused things things
        if self.fl_list != []: 
            fl_t = torch.stack(self.fl_list)

            # Identify the winner between things and 
            inter_pred = torch.argmax(torch.cat((self._sem_stuff, fl_t), 0), dim=0)

            inter_pred = inter_pred.numpy()
            
            np.vectorize(self.write_to_instmap)(inter_pred)
            self._canvas = self._canvas + inter_pred 
            
            # Fill empty elements with sem_head results
            self._canvas = np.where(self._canvas > 0, self._canvas, self._sem_pred)

        return self._canvas
