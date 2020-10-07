from collections import defaultdict
from lgblkb_tools import Folder
from lgblkb_tools.pathify import get_name

from app.black_box import transforms as T
from pycocotools import mask as coco_mask
import geopandas as gpd

import os
import numpy as np
import rasterio
import torch
import torchvision
from PIL import Image, ImageDraw
from pycocotools.coco import COCO
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


class FieldsDataset(torch.utils.data.Dataset):
    def __init__(self, folder_paths, transforms=None):
        self.folder_paths = folder_paths
        self.transforms = transforms
        # self.imgs = list()
        self.info = defaultdict(list)

        for folder_path in folder_paths:
            folder = Folder(folder_path)
            for photo_path in folder['Photos'].children:
                self.info['img'].append(photo_path)
                self.info['coors'].append(folder['Coordinates'][f'{get_name(photo_path)}.shp'])
                # self.imgs.extend(photo_path)  # Full path
                # self.coors.extend()  # Full path

        # for item in os.listdir(os.path.join(root, "Photos")):
        #     name, ext = os.path.splitext(item)
        #     self.imgs.append('{}.tif'.format(name))
        pass

    def __getitem__(self, idx):
        # img_path = os.path.join(self.root, "Photos", self.imgs[idx])
        img_path = self.info['img'][idx]
        src = rasterio.open(img_path)
        nir = src.read(4)
        red = src.read(3)
        green = src.read(2)
        src = np.dstack((nir, red, green))
        src = np.nan_to_num(src)
        src = (src * 255 / np.max(src)).astype('uint8')
        image = Image.fromarray(src)

        target = get_target(self.info, idx)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self):
        return len(self.info['img'])


# %%

# Defining the Mask R-CNN algorithm

def get_instance_segmentation_model(num_classes):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)
    return model


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


def annotate_image(image_info, ann_id):
    img, targets = image_info
    image_id = targets["image_id"].item()
    img_dict = dict(id=image_id, height=img.shape[-2], width=img.shape[-1])
    categories = set()
    # target_dataset['images'].append(img_dict)
    bboxes = targets["boxes"]
    bboxes[:, 2:] -= bboxes[:, :2]
    bboxes = bboxes.tolist()
    labels = targets['labels'].tolist()
    areas = targets['area'].tolist()
    iscrowd = targets['iscrowd'].tolist()
    if 'masks' in targets:
        masks = targets['masks'].permute(0, 2, 1).contiguous().permute(0, 2, 1)

    if 'keypoints' in targets:
        keypoints = targets['keypoints']
        keypoints = keypoints.reshape(keypoints.shape[0], -1).tolist()
    num_objs = len(bboxes)
    annotations = list()
    for i in range(num_objs):
        ann = dict(image_id=image_id, bbox=bboxes[i], id=ann_id,
                   category_id=labels[i], area=areas[i], iscrowd=iscrowd[i])
        categories.add(labels[i])
        if 'masks' in targets:
            ann["segmentation"] = coco_mask.encode(masks[i].numpy())
        if 'keypoints' in targets:
            ann['keypoints'] = keypoints[i]
            ann['num_keypoints'] = sum(k != 0 for k in keypoints[i][2::3])
        annotations.append(ann)
    return img_dict, categories, annotations


def convert_to_coco_api(ds):
    coco_ds = COCO()
    ann_id = 0
    target_dataset = {'images': [], 'categories': [], 'annotations': []}
    categories = set()
    for img_idx in range(len(ds)):
        print(f"[INFO] Converting to coco dataset - [{img_idx}/{len(ds)}]\r", end='')
        # annotate_image(ds[img_idx],ann_id)
        # find better way to get target
        # targets = ds.get_annotations(img_idx)
        img, targets = ds[img_idx]
        image_id = targets["image_id"].item()
        img_dict = {}
        img_dict['id'] = image_id
        img_dict['height'] = img.shape[-2]
        img_dict['width'] = img.shape[-1]
        target_dataset['images'].append(img_dict)
        bboxes = targets["boxes"]
        bboxes[:, 2:] -= bboxes[:, :2]
        bboxes = bboxes.tolist()
        labels = targets['labels'].tolist()
        areas = targets['area'].tolist()
        iscrowd = targets['iscrowd'].tolist()
        if 'masks' in targets:
            masks = targets['masks']
            # make masks Fortran contiguous for coco_mask
            masks = masks.permute(0, 2, 1).contiguous().permute(0, 2, 1)
        if 'keypoints' in targets:
            keypoints = targets['keypoints']
            keypoints = keypoints.reshape(keypoints.shape[0], -1).tolist()
        num_objs = len(bboxes)
        for i in range(num_objs):
            ann = {}
            ann['image_id'] = image_id
            ann['bbox'] = bboxes[i]
            ann['category_id'] = labels[i]
            categories.add(labels[i])
            ann['area'] = areas[i]
            ann['iscrowd'] = iscrowd[i]
            ann['id'] = ann_id
            if 'masks' in targets:
                ann["segmentation"] = coco_mask.encode(masks[i].numpy())
            if 'keypoints' in targets:
                ann['keypoints'] = keypoints[i]
                ann['num_keypoints'] = sum(k != 0 for k in keypoints[i][2::3])
            target_dataset['annotations'].append(ann)
            ann_id += 1
    print(f"[INFO] Converting to coco dataset - [{img_idx}/{len(ds)}]")
    target_dataset['categories'] = [{'id': i} for i in sorted(categories)]
    coco_ds.dataset = target_dataset
    coco_ds.createIndex()
    print(coco_ds)
    print(type(coco_ds))
    return coco_ds


def get_coco_api_from_dataset(target_dataset):
    for i in range(10):
        if isinstance(target_dataset, torchvision.datasets.CocoDetection):
            break
        if isinstance(target_dataset, torch.utils.data.Subset):
            target_dataset = target_dataset.dataset
    if isinstance(target_dataset, torchvision.datasets.CocoDetection):
        return target_dataset.coco
    return convert_to_coco_api(target_dataset)


# def get_target(root, image_name, idx):
def get_target(info, idx):
    shp_file = gpd.read_file(info['coors'][idx])
    image = rasterio.open(info['img'][idx])

    # name, ext = os.path.splitext(image_name)
    # shp_file = gpd.read_file(os.path.join(root, "Coordinates", '{}.shp'.format(name)))
    # image = rasterio.open(os.path.join(root, "Photos", image_name))

    left, bottom, right, top = image.bounds
    img_height, img_width = image.shape
    meters_in_pixel = (right - left) / img_width

    boxes = []
    masks = []

    for i in range(len(shp_file)):
        if shp_file.geometry[i] is None or shp_file.geometry[i].type == "Point":
            continue

        xmin = (shp_file.geometry[i].bounds[0] - left) / meters_in_pixel
        ymax = img_height - (shp_file.geometry[i].bounds[1] - bottom) / meters_in_pixel
        xmax = (shp_file.geometry[i].bounds[2] - left) / meters_in_pixel
        ymin = img_height - (shp_file.geometry[i].bounds[3] - bottom) / meters_in_pixel

        boxes.append([xmin, ymin, xmax, ymax])

        temp = []

        if shp_file.geometry[i].type == "Polygon":
            if shp_file.geometry[i].boundary.type == "MultiLineString":
                for j in range(len(shp_file.geometry[i].boundary)):
                    coord_len = len(shp_file.geometry[i].boundary[j].coords.xy[0])

                    for k in range(coord_len):
                        x_coord = (shp_file.geometry[i].boundary[j].coords.xy[0][k] - left) / meters_in_pixel
                        y_coord = (top - bottom) / meters_in_pixel - (
                                shp_file.geometry[i].boundary[j].coords.xy[1][k] - bottom) / meters_in_pixel

                        temp.append((x_coord, y_coord))
            else:
                coord_len = len(shp_file.geometry[i].boundary.coords.xy[0])

                for j in range(coord_len):
                    x_coord = (shp_file.geometry[i].boundary.coords.xy[0][j] - left) / meters_in_pixel
                    y_coord = (top - bottom) / meters_in_pixel - (
                            shp_file.geometry[i].boundary.coords.xy[1][j] - bottom) / meters_in_pixel

                    temp.append((x_coord, y_coord))
        elif shp_file.geometry[i].type == "MultiPolygon":
            for j in range(len(shp_file.geometry[i])):
                single_polygon = []
                if shp_file.geometry[i][j].boundary.type == "MultiLineString":
                    for k in range(len(shp_file.geometry[i][j].boundary)):
                        coord_len = len(shp_file.geometry[i][j].boundary[k].coords.xy[0])

                        for l in range(coord_len):
                            x_coord = (shp_file.geometry[i][j].boundary[k].coords.xy[0][l] - left) / meters_in_pixel
                            y_coord = (top - bottom) / meters_in_pixel - (
                                    shp_file.geometry[i][j].boundary[k].coords.xy[1][l] - bottom) / meters_in_pixel

                            single_polygon.append((x_coord, y_coord))
                else:
                    coord_len = len(shp_file.geometry[i][j].boundary.coords.xy[0])

                    for k in range(coord_len):
                        x_coord = (shp_file.geometry[i][j].boundary.coords.xy[0][k] - left) / meters_in_pixel
                        y_coord = (top - bottom) / meters_in_pixel - (
                                shp_file.geometry[i][j].boundary.coords.xy[1][k] - bottom) / meters_in_pixel

                        single_polygon.append((x_coord, y_coord))
                temp.append(single_polygon)

        img = Image.new('L', (img_width, img_height), 0)

        if shp_file.geometry[i].type == "Polygon":
            ImageDraw.Draw(img).polygon(temp, outline=False, fill=True)
        elif shp_file.geometry[i].type == "MultiPolygon":
            for polygon in temp:
                ImageDraw.Draw(img).polygon(polygon, outline=False, fill=True)

        mask = np.array(img)
        masks.append(mask)

    target = {}

    masks = np.array(masks)

    boxes = torch.as_tensor(boxes, dtype=torch.float32)
    masks = torch.as_tensor(masks, dtype=torch.uint8)

    target['area'] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
    target['labels'] = torch.ones((len(boxes),), dtype=torch.int64)

    target['boxes'] = boxes
    target['masks'] = masks
    target['image_id'] = torch.tensor([idx])
    target['iscrowd'] = torch.zeros((len(boxes),), dtype=torch.int64)

    return target


def get_transform():
    transforms = []
    transforms.append(T.ToTensor())
    return T.Compose(transforms)
