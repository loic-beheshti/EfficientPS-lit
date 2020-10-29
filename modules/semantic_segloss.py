
"""
Pixel-wise weighted log loss with hard mining
From => https://github.com/mapillary/seamseg/blob/master/seamseg/algos/semantic_seg.py
"""

from math import ceil

import torch
import torch.nn.functional as functional

class SemanticSegLoss:
    """Semantic segmentation loss
    Parameters
    ----------
    ohem : float or None
        Online hard example mining fraction, or `None` to disable OHEM
    ignore_index : int
        Index of the void class
    """

    def __init__(self, ohem=None, ignore_index=250):
        if ohem is not None and (ohem <= 0 or ohem > 1):
            raise ValueError("ohem should be in (0, 1]")
        self.ohem = ohem
        self.ignore_index = ignore_index

    def __call__(self, sem_logits, sem):
        """Compute the semantic segmentation loss
        Parameters
        ----------
        sem_logits : sequence of torch.Tensor
            A sequence of N tensors of segmentation logits with shapes C x H_i x W_i
        sem : sequence of torch.Tensor
            A sequence of N tensors of ground truth semantic segmentations with shapes H_i x W_i
        Returns
        -------
        sem_loss : torch.Tensor
            A scalar tensor with the computed loss
        """
        sem_loss = []
        for sem_logits_i, sem_i in zip(sem_logits, sem):

            sem_loss_i = functional.cross_entropy(
                sem_logits_i.unsqueeze(0), sem_i.unsqueeze(0), ignore_index=self.ignore_index, reduction="none")
            sem_loss_i = sem_loss_i.view(-1)

            if self.ohem is not None and self.ohem != 1:
                top_k = int(ceil(sem_loss_i.numel() * self.ohem))
                if top_k != sem_loss_i.numel():
                    sem_loss_i, _ = sem_loss_i.topk(top_k)

            sem_loss.append(sem_loss_i.mean())

        return sum(sem_loss) / len(sem_logits)