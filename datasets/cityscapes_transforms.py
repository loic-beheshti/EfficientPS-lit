import os
import torch
import numpy as np
import scipy.misc as m
import sys

class cityscapesTransforms(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]

        self.class_names = [
            "unlabelled",
            "road",
            "sidewalk",
            "building",
            "wall",
            "fence",
            "pole",
            "traffic_light",
            "traffic_sign",
            "vegetation",
            "terrain",
            "sky",
            "person",
            "rider",
            "car",
            "truck",
            "bus",
            "train",
            "motorcycle",
            "bicycle"
        ]

        self.ignore_index = 250
        self.class_map = dict(zip(self.valid_classes, range(19)))
        self.n_classes = 19
        self.img_size = (1024, 2048)

    def encode_instance_label(self, label):
        if label in self.void_classes: 
            return -1
        else:
            return self.class_map[label] 

    def mask_to_tight_box(self, mask):
        a = mask.nonzero()
        bbox = [
            torch.min(a[:, 1]),
            torch.min(a[:, 0]),
            torch.max(a[:, 1]),
            torch.max(a[:, 0]),
        ]
        bbox = list(map(int, bbox))
        #boxes can be lines if small and can break the gradients
        if(bbox[1] == bbox[3]):
            bbox[3]+=1
        if(bbox[0] == bbox[2]):
            bbox[2]+=1
        return bbox  # xmin, ymin, xmax, ymax

    def processBinayMasks(self, ann):
        boxes = []
        masks = []
        labels = []

        # Sort for consistent order between instances as the polygon annotation
        instIds = torch.sort(torch.unique(ann))[0]
        for instId in instIds:
            if instId < 1000:  # group labels
                continue

            mask = ann == instId
            label = (instId / 1000.).int()
            label = self.encode_instance_label(label.item())
            if label < 0:
                continue
            box = self.mask_to_tight_box(mask)

            boxes.append(torch.FloatTensor(box))
            masks.append(mask)
            labels.append(label)

        #print(boxes[0])
        #print(torch.stack(boxes))
        #print(torch.stack(masks))
        if boxes == []:
            return [], [], []

        return torch.stack(boxes), torch.stack(masks), torch.LongTensor(labels)-11


    def encode_segmap(self, mask):
        # Put all void classes to zero
        for _voidc in self.void_classes:
            mask[mask == _voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        return mask
    
    def forward(self, img, lbls, norm = True):
        img = np.array(img, dtype=np.uint8)
        
        sem_lbl = lbls[0]
        inst_lbl = lbls[1]
        
        sem_lbl = self.encode_segmap(np.array(sem_lbl, dtype=np.uint8))

        inst_lbl = np.array(inst_lbl, dtype=np.int32)
        inst_lbl = torch.from_numpy(inst_lbl).long()
        #sem_lbl = torch.from_numpy(sem_lbl).long()
        boxes, masks, labels  = self.processBinayMasks(inst_lbl)
        #print("lab", labels) 
        #labels = labels - 11
        #np.set_printoptions(threshold=sys.maxsize)
        #print(sem_lbl.shape, boxes)

        """
        import cv2
        im = img[...]
        cv2.imwrite('org_img.jpg', im)
        for box in boxes:
            cv2.rectangle(im,(box[0],box[1]),(box[2],box[3]),(0,255,0),2) # add rectangle to image
        cv2.imwrite('bounding_box.jpg', im)
        """
        
        img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float64)
        if norm:
            img = img.astype(float) / 255.0
        # NHWC -> NCHW
        img = img.transpose(2, 0, 1)
        
        sem_lbl = sem_lbl.astype(int)

        img = torch.from_numpy(img).float()
        sem_lbl = torch.from_numpy(sem_lbl).long()

        #print("sem = ", sem_lbl.unique())

        #print(sem_lbl, boxes, masks, labels)

        #print("img, labels", img.size(), lbl.size())
        return img, [sem_lbl, boxes, masks, labels]