from collections import OrderedDict

import torch
from torch import nn
import torch.nn.functional as F

from torchvision.ops import misc as misc_nn_ops
from torchvision.ops import MultiScaleRoIAlign

#from ._utils import overwrite_eps
#from ..utils import load_state_dict_from_url

from torchvision.models.detection import MaskRCNN
from torch.jit.annotations import Tuple, List, Dict, Optional
import warnings
#from .backbone_utils import resnet_fpn_backbone


class BackboneDummy(nn.Module):
    """
    Dummy to use and modify mask rcnn from torchvision
    """
    def __init__(self, features = [], out_channels = 256):
        super().__init__()
    
        self.features = features
        self.out_channels = out_channels

    def store_features(self, features):
        self.features = features

    def forward(self):
        return self.features

class MyMaskRCNN(MaskRCNN):
    """
    Implements Mask R-CNN.
    The input to the model is expected to be a list of tensors, each of shape [C, H, W], one for each
    image, and should be in 0-1 range. Different images can have different sizes.
    The behavior of the model changes depending if it is in training or evaluation mode.
    During training, the model expects both the input tensors, as well as a targets (list of dictionary),
    containing:
        - boxes (FloatTensor[N, 4]): the ground-truth boxes in [x1, y1, x2, y2] format, with values of x
          between 0 and W and values of y between 0 and H
        - labels (Int64Tensor[N]): the class label for each ground-truth box
        - masks (UInt8Tensor[N, H, W]): the segmentation binary masks for each instance
    The model returns a Dict[Tensor] during training, containing the classification and regression
    losses for both the RPN and the R-CNN, and the mask loss.
    During inference, the model requires only the input tensors, and returns the post-processed
    predictions as a List[Dict[Tensor]], one for each input image. The fields of the Dict are as
    follows:
        - boxes (FloatTensor[N, 4]): the predicted boxes in [x1, y1, x2, y2] format, with values of x
          between 0 and W and values of y between 0 and H
        - labels (Int64Tensor[N]): the predicted labels for each image
        - scores (Tensor[N]): the scores or each prediction
        - masks (UInt8Tensor[N, 1, H, W]): the predicted masks for each instance, in 0-1 range. In order to
          obtain the final segmentation masks, the soft masks can be thresholded, generally
          with a value of 0.5 (mask >= 0.5)
    Arguments:
        backbone (nn.Module): the network used to compute the features for the model.
            It should contain a out_channels attribute, which indicates the number of output
            channels that each feature map has (and it should be the same for all feature maps).
            The backbone should return a single Tensor or and OrderedDict[Tensor].
        num_classes (int): number of output classes of the model (including the background).
            If box_predictor is specified, num_classes should be None.
        min_size (int): minimum size of the image to be rescaled before feeding it to the backbone
        max_size (int): maximum size of the image to be rescaled before feeding it to the backbone
        image_mean (Tuple[float, float, float]): mean values used for input normalization.
            They are generally the mean values of the dataset on which the backbone has been trained
            on
        image_std (Tuple[float, float, float]): std values used for input normalization.
            They are generally the std values of the dataset on which the backbone has been trained on
        rpn_anchor_generator (AnchorGenerator): module that generates the anchors for a set of feature
            maps.
        rpn_head (nn.Module): module that computes the objectness and regression deltas from the RPN
        rpn_pre_nms_top_n_train (int): number of proposals to keep before applying NMS during training
        rpn_pre_nms_top_n_test (int): number of proposals to keep before applying NMS during testing
        rpn_post_nms_top_n_train (int): number of proposals to keep after applying NMS during training
        rpn_post_nms_top_n_test (int): number of proposals to keep after applying NMS during testing
        rpn_nms_thresh (float): NMS threshold used for postprocessing the RPN proposals
        rpn_fg_iou_thresh (float): minimum IoU between the anchor and the GT box so that they can be
            considered as positive during training of the RPN.
        rpn_bg_iou_thresh (float): maximum IoU between the anchor and the GT box so that they can be
            considered as negative during training of the RPN.
        rpn_batch_size_per_image (int): number of anchors that are sampled during training of the RPN
            for computing the loss
        rpn_positive_fraction (float): proportion of positive anchors in a mini-batch during training
            of the RPN
        box_roi_pool (MultiScaleRoIAlign): the module which crops and resizes the feature maps in
            the locations indicated by the bounding boxes
        box_head (nn.Module): module that takes the cropped feature maps as input
        box_predictor (nn.Module): module that takes the output of box_head and returns the
            classification logits and box regression deltas.
        box_score_thresh (float): during inference, only return proposals with a classification score
            greater than box_score_thresh
        box_nms_thresh (float): NMS threshold for the prediction head. Used during inference
        box_detections_per_img (int): maximum number of detections per image, for all classes.
        box_fg_iou_thresh (float): minimum IoU between the proposals and the GT box so that they can be
            considered as positive during training of the classification head
        box_bg_iou_thresh (float): maximum IoU between the proposals and the GT box so that they can be
            considered as negative during training of the classification head
        box_batch_size_per_image (int): number of proposals that are sampled during training of the
            classification head
        box_positive_fraction (float): proportion of positive proposals in a mini-batch during training
            of the classification head
        bbox_reg_weights (Tuple[float, float, float, float]): weights for the encoding/decoding of the
            bounding boxes
        mask_roi_pool (MultiScaleRoIAlign): the module which crops and resizes the feature maps in
             the locations indicated by the bounding boxes, which will be used for the mask head.
        mask_head (nn.Module): module that takes the cropped feature maps as input
        mask_predictor (nn.Module): module that takes the output of the mask_head and returns the
            segmentation mask logits
    Example::
        >>> import torch
        >>> import torchvision
        >>> from torchvision.models.detection import MaskRCNN
        >>> from torchvision.models.detection.anchor_utils import AnchorGenerator
        >>>
        >>> # load a pre-trained model for classification and return
        >>> # only the features
        >>> backbone = torchvision.models.mobilenet_v2(pretrained=True).features
        >>> # MaskRCNN needs to know the number of
        >>> # output channels in a backbone. For mobilenet_v2, it's 1280
        >>> # so we need to add it here
        >>> backbone.out_channels = 1280
        >>>
        >>> # let's make the RPN generate 5 x 3 anchors per spatial
        >>> # location, with 5 different sizes and 3 different aspect
        >>> # ratios. We have a Tuple[Tuple[int]] because each feature
        >>> # map could potentially have different sizes and
        >>> # aspect ratios
        >>> anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
        >>>                                    aspect_ratios=((0.5, 1.0, 2.0),))
        >>>
        >>> # let's define what are the feature maps that we will
        >>> # use to perform the region of interest cropping, as well as
        >>> # the size of the crop after rescaling.
        >>> # if your backbone returns a Tensor, featmap_names is expected to
        >>> # be ['0']. More generally, the backbone should return an
        >>> # OrderedDict[Tensor], and in featmap_names you can choose which
        >>> # feature maps to use.
        >>> roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
        >>>                                                 output_size=7,
        >>>                                                 sampling_ratio=2)
        >>>
        >>> mask_roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
        >>>                                                      output_size=14,
        >>>                                                      sampling_ratio=2)
        >>> # put the pieces together inside a MaskRCNN model
        >>> model = MaskRCNN(backbone,
        >>>                  num_classes=2,
        >>>                  rpn_anchor_generator=anchor_generator,
        >>>                  box_roi_pool=roi_pooler,
        >>>                  mask_roi_pool=mask_roi_pooler)
        >>> model.eval()
        >>> x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
        >>> predictions = model(x)
    """
    def __init__(self, backbone, num_classes=None,
                 # transform parameters
                 min_size=800, max_size=1333,
                 image_mean=None, image_std=None,
                 # RPN parameters
                 rpn_anchor_generator=None, rpn_head=None,
                 rpn_pre_nms_top_n_train=2000, rpn_pre_nms_top_n_test=1000,
                 rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=1000,
                 rpn_nms_thresh=0.7,
                 rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,
                 rpn_batch_size_per_image=256, rpn_positive_fraction=0.5,
                 # Box parameters
                 box_roi_pool=None, box_head=None, box_predictor=None,
                 box_score_thresh=0.05, box_nms_thresh=0.5, box_detections_per_img=100,
                 box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5,
                 box_batch_size_per_image=512, box_positive_fraction=0.25,
                 bbox_reg_weights=None,
                 # Mask parameters
                 mask_roi_pool=None, mask_head=None, mask_predictor=None):

        backbone = BackboneDummy()

        super(MyMaskRCNN, self).__init__(
            backbone, num_classes,
            # transform parameters
            min_size, max_size,
            image_mean, image_std,
            # RPN-specific parameters
            rpn_anchor_generator, rpn_head,
            rpn_pre_nms_top_n_train, rpn_pre_nms_top_n_test,
            rpn_post_nms_top_n_train, rpn_post_nms_top_n_test,
            rpn_nms_thresh,
            rpn_fg_iou_thresh, rpn_bg_iou_thresh,
            rpn_batch_size_per_image, rpn_positive_fraction,
            # Box parameters
            box_roi_pool, box_head, box_predictor,
            box_score_thresh, box_nms_thresh, box_detections_per_img,
            box_fg_iou_thresh, box_bg_iou_thresh,
            box_batch_size_per_image, box_positive_fraction,
            bbox_reg_weights)
        
        


    def forward(self, images, features, targets=None):
            # type: (List[Tensor], Optional[List[Dict[str, Tensor]]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
            """
            Arguments:
                images (list[Tensor]): images to be processed
                targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)
            Returns:
                result (list[BoxList] or dict[Tensor]): the output from the model.
                    During training, it returns a dict[Tensor] which contains the losses.
                    During testing, it returns list[BoxList] contains additional fields
                    like `scores`, `labels` and `mask` (for Mask R-CNN models).
            """
            if self.training and targets is None:
                raise ValueError("In training mode, targets should be passed")
            if self.training:
                assert targets is not None
                for target in targets:
                    #print("target =", target)
                    boxes = target["boxes"]
                    if isinstance(boxes, torch.Tensor):
                        if len(boxes.shape) != 2 or boxes.shape[-1] != 4:
                            raise ValueError("Expected target boxes to be a tensor"
                                            "of shape [N, 4], got {:}.".format(
                                                boxes.shape))
                    else:
                        raise ValueError("Expected target boxes to be of type "
                                        "Tensor, got {:}.".format(type(boxes)))

            original_image_sizes = torch.jit.annotate(List[Tuple[int, int]], [])
            for img in images:
                val = img.shape[-2:]
                assert len(val) == 2
                original_image_sizes.append((val[0], val[1]))

            images, targets = self.transform(images, targets)

            # Check for degenerate boxes
            # TODO: Move this to a function
            if targets is not None:
                for target_idx, target in enumerate(targets):
                    boxes = target["boxes"]
                    degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                    if degenerate_boxes.any():
                        # print the first degenerate box
                        bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                        degen_bb: List[float] = boxes[bb_idx].tolist()
                        raise ValueError("All bounding boxes should have positive height and width."
                                        " Found invalid box {} for target at index {}."
                                        .format(degen_bb, target_idx))

            #features = self.backbone(images.tensors)
            #if isinstance(features, torch.Tensor):
            #    features = OrderedDict([('0', features)])

            f0, f1, f2, f3 = features
            features = OrderedDict([('0', f0), ('1', f1), ('2', f2), ('3', f3)])
            proposals, proposal_losses = self.rpn(images, features, targets)
            detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)
            detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)

            if torch.jit.is_scripting():
                if not self._has_warned:
                    warnings.warn("RCNN always returns a (Losses, Detections) tuple in scripting")
                    self._has_warned = True
                return (losses, detections)
            else:
                return self.eager_outputs(losses, detections)

