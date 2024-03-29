import collections
import geopandas as gpd
import lgblkb_tools
import numpy as np
import random
import rasterio
import torch
import torch.utils.data
import torchvision
import torchvision.transforms.functional as F

from PIL import Image, ImageDraw


def collate_fn(batch):
    return tuple(zip(*batch))


def get_instance_segmentation_model(num_classes):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 512
    model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(in_features_mask,
                                                                                              hidden_layer, num_classes)
    return model
