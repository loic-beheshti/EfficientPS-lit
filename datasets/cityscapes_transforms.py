import os
import torch
import numpy as np
import scipy.misc as m
import sys
import cv2
import random
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

"""
Parts of this code from https://github.com/meetshah1995/pytorch-semseg/blob/master/ptsemseg/loader/cityscapes_loader.py
"""
colors = [  # [  0,   0,   0],
            [128, 64, 128],
            [244, 35, 232],
            [70, 70, 70],
            [102, 102, 156],
            [190, 153, 153],
            [153, 153, 153],
            [250, 170, 30],
            [220, 220, 0],
            [107, 142, 35],
            [152, 251, 152],
            [0, 130, 180],
            [220, 20, 60],
            [255, 0, 0],
            [0, 0, 142],
            [0, 0, 70],
            [0, 60, 100],
            [0, 80, 100],
            [0, 0, 230],
            [119, 11, 32],
        ]

void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]

class_names = [
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

label_colours = dict(zip(range(19), colors))
ignore_index = 250
class_map = dict(zip(valid_classes, range(19)))
n_classes = 19
img_size = (1024, 2048)

def encode_instance_label(label):
    if label in void_classes: 
        return -1
    else:
        return class_map[label]

def encode_segmap(mask):
    # Put all void classes to zero
    for _voidc in void_classes:
        mask[mask == _voidc] = ignore_index
    for _validc in valid_classes:
        mask[mask == _validc] = class_map[_validc]
    return mask

def decode_segmap(temp):
    r = temp.copy()
    g = temp.copy()
    b = temp.copy()
    for l in range(0, 19):
        r[temp == l] = label_colours[l][0]
        g[temp == l] = label_colours[l][1]
        b[temp == l] = label_colours[l][2]

    rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
    rgb[:, :, 0] = r #/ 255.0
    rgb[:, :, 1] = g #/ 255.0
    rgb[:, :, 2] = b #/ 255.0
    return rgb

def instmap_to_segmap(inst):
    inst_seg = inst//1000
    inst_t = inst.copy()
    np.where(inst_t < 1000, inst_t, 0)
    inst_seg = inst_seg + inst_t
    #print("inst_seg", np.unique(inst_seg))
    return inst_seg

def decode_instmap(temp, diff_instance=False, needs_encoding=False):
    if needs_encoding == True:
        temp_sem = encode_segmap(temp//1000)
    else:
        #print("temp", np.unique(temp))
        temp_sem = temp//1000
        #print("temp_sem", np.unique(temp_sem))
    if diff_instance == True:
        temp_sem = (temp_sem + temp%1000) % 19
    return decode_segmap(temp_sem)

def save_samples_out(img, sem, inst, boxes, name="gd", val_dir="./val_samples/"):
    #print(img.shape, sem.shape, inst.shape, boxes.shape)
    im = img[...]
    im = np.transpose(im, (1, 2, 0))
    cv2.imwrite(val_dir+name+"_img.jpg", im)
    cv2.imwrite(val_dir+name+'_sem.jpg', decode_segmap(sem))
    cv2.imwrite(val_dir+name+'_inst.jpg', decode_instmap(inst, needs_encoding=False))
    cv2.imwrite(val_dir+name+'_instdiff.jpg', decode_instmap(inst, diff_instance=True, needs_encoding=False))
    im = cv2.UMat(im)
    for box in boxes:
        cv2.rectangle(im,(box[0],box[1]),(box[2],box[3]),(0,255,0),2) # add rectangle to image
    cv2.imwrite(val_dir+name+'_bounding_box.jpg', im)

def save_samples(img, sem, inst, boxes):
    im = np.transpose(img, (1, 2, 0))
    print("shape" ,im.shape)
    cv2.imwrite('org_img.jpg', im)
    #cv2.imwrite('decoded_sem.jpg', decode_segmap(sem))
    #cv2.imwrite('decoded_inst.jpg', decode_instmap(inst))
    #cv2.imwrite('decoded_instdiff.jpg', decode_instmap(inst, diff_instance=True))
    
    #for box in boxes:
    #    cv2.rectangle(im,(box[0],box[1]),(box[2],box[3]),(0,255,0),2) # add rectangle to image
    #cv2.imwrite('bounding_box.jpg', im)

class cityscapesTransforms(torch.nn.Module):
    def __init__(self, augmentation=False):
        self.augmentation = augmentation
        self.random_crops = [(1024,2048), (1024,1024), (384,1280), (720,1280)] # 1024×2048, 1024×1024, 384×1280, 720×1280
        super().__init__()

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

        instIds = torch.sort(torch.unique(ann))[0]
        for instId in instIds:
            if instId < 1000:  # group labels
                continue

            mask = ann == instId
            label = (instId / 1000.).int()
            label = encode_instance_label(label.item())
            if label < 0:
                continue
            box = self.mask_to_tight_box(mask)

            boxes.append(torch.FloatTensor(box))
            masks.append(mask)
            labels.append(label)

        if boxes == []:
            return [], [], []

        return torch.stack(boxes), torch.stack(masks), torch.LongTensor(labels)-11

    def aug_transform(self, image, mask, inst):
        # Random crop
        rand_size = self.random_crops[random.randint(0, 3)]
        i, j, h, w = transforms.RandomCrop.get_params(image, output_size=rand_size)
        image = TF.crop(image, i, j, h, w)
        mask = TF.crop(mask, i, j, h, w)
        inst = TF.crop(inst, i, j, h, w)

        # Random horizontal flipping
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
            inst = TF.hflip(inst)

        # Random vertical flipping
        if random.random() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)
            inst = TF.vflip(inst)
        
        return image, mask, inst
    
    def forward(self, img, lbls, norm = True):
        
        sem_lbl = lbls[0]
        inst_lbl = lbls[1]

        if self.augmentation == True:
            img, sem_lbl, inst_lbl = self.aug_transform(img, sem_lbl, inst_lbl)
        
        img = np.array(img, dtype=np.float64)
        img = img[:, :, ::-1]  # RGB -> BGR
        img = img.transpose(2, 0, 1) # NHWC -> NCHW

        sem_lbl = np.array(sem_lbl, dtype=np.uint8)
        inst_lbl = np.array(inst_lbl, dtype=np.int32)
        
        sem_lbl = encode_segmap(sem_lbl)
        inst_lbl = torch.from_numpy(inst_lbl).long()
        boxes, masks, labels = self.processBinayMasks(inst_lbl)

        if norm:
            img = img.astype(float) / 255.0

        sem_lbl = sem_lbl.astype(int)

        img = torch.from_numpy(img).float()
        sem_lbl = torch.from_numpy(sem_lbl).long()

        return img, [sem_lbl, boxes, masks, labels]