import os
import torch
import numpy as np
import scipy.misc as m

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

    def encode_segmap(self, mask):
        # Put all void classes to zero
        for _voidc in self.void_classes:
            mask[mask == _voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        return mask
    
    def forward(self, img, lbl, norm = True):
        img = np.array(img, dtype=np.uint8)

        lbl = self.encode_segmap(np.array(lbl, dtype=np.uint8))

        img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float64)
        if norm:
            img = img.astype(float) / 255.0
        # NHWC -> NCHW
        img = img.transpose(2, 0, 1)

        classes = np.unique(lbl)
        #print("classes", classes)
        
        lbl = lbl.astype(int)

        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()

        #print("img, labels", img.size(), lbl.size())
        return img, lbl