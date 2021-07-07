import collections
import copy
import json
import lgblkb_tools
import numpy as np
import pickle
import pycocotools.coco
import pycocotools.cocoeval
import pycocotools.mask
import torch
import torch.distributed

from functools import partial
from multiprocessing import Manager, Pool

import config

from training import NUM_WORKERS

categories = set()


def parallel_coco_execution(idx, ds, target_dataset):
    img, targets = ds[idx]
    image_id = targets['image_id'].item()
    img_dict = {'id': image_id, 'height': img.shape[-2], 'width': img.shape[-1]}
    target_dataset['images'].append(img_dict)
    bboxes = targets['boxes']
    bboxes[:, 2:] -= bboxes[:, :2]
    bboxes = bboxes.tolist()
    labels = targets['labels'].tolist()
    areas = targets['area'].tolist()
    iscrowd = targets['iscrowd'].tolist()

    if 'masks' in targets:
        masks = targets['masks'].permute(0, 2, 1).contiguous().permute(0, 2, 1)

    num_objs = len(bboxes)

    for i in range(num_objs):
        ann = {'image_id': image_id,
               'bbox': bboxes[i],
               'category_id': labels[i],
               'area': areas[i],
               'iscrowd': iscrowd[i],
               'id': idx}
        categories.add(labels[i])

        if 'masks' in targets:
            ann['segmentation'] = pycocotools.mask.encode(masks[i].numpy())

        target_dataset['annotations'].append(ann)

    length = len(target_dataset['images'])

    if length > 0 and length % 25 == 0:
        lgblkb_tools.logger.debug(f"Progress: [{length} out of 4200]")


@lgblkb_tools.logger.trace()
def convert_to_coco_api(ds, title):
    coco_ds = pycocotools.coco.COCO()
    ds_length = len(ds)

    with Manager() as manager:
        manager_dataset = manager.dict()
        manager_dataset['images'] = manager.list()
        manager_dataset['categories'] = manager.list()
        manager_dataset['annotations'] = manager.list()

        with Pool(NUM_WORKERS) as pool:
            pool.map(partial(
                parallel_coco_execution,
                ds=ds,
                target_dataset=manager_dataset
            ), range(ds_length))

        target_dataset = dict()
        target_dataset['images'] = list(manager_dataset['images'])
        target_dataset['categories'] = [{"supercategory": "field",
                                         "id": i,
                                         "name": "field"} for i in sorted(categories)]
        target_dataset['annotations'] = list(manager_dataset['annotations'])

    coco_ds.dataset = target_dataset
    coco_ds.createIndex()
    pickle.dump(coco_ds, open(title, 'wb'))


@lgblkb_tools.logger.trace()
def save_as_coco_dataset(data_loader_train, data_loader_valid, data_loader_test):
    convert_to_coco_api(data_loader_train.dataset, config.data_folder['Train']['coco_train.pickle'])
    convert_to_coco_api(data_loader_valid.dataset, config.data_folder['Valid']['coco_valid.pickle'])
    convert_to_coco_api(data_loader_test.dataset, config.data_folder['Test']['coco_test.pickle'])


class CocoEvaluator(object):
    def __init__(self, coco_gt, iou_types):
        assert isinstance(iou_types, (list, tuple))
        coco_gt = copy.deepcopy(coco_gt)
        self.coco_gt = coco_gt
        self.iou_types = iou_types
        self.coco_eval = {}

        for iou_type in self.iou_types:
            self.coco_eval[iou_type] = pycocotools.cocoeval.COCOeval(coco_gt, iouType=iou_type)

        self.img_ids = []
        self.eval_imgs = {k: [] for k in self.iou_types}
        self.res = None

    def update(self, predictions):
        img_ids = list(np.unique(list(predictions.keys())))
        self.img_ids.extend(img_ids)

        for iou_type in self.iou_types:
            results = self.prepare(predictions, iou_type)
            coco_dt = loadRes(self.coco_gt, results) if results else pycocotools.coco.COCO()
            coco_eval = self.coco_eval[iou_type]
            coco_eval.cocoDt = coco_dt
            coco_eval.params.imgIds = list(img_ids)
            img_ids, eval_imgs = evaluate(coco_eval)
            self.eval_imgs[iou_type].append(eval_imgs)

    def prepare(self, predictions, iou_type):
        if iou_type == "bbox":
            return self.prepare_for_coco_detection(predictions)
        elif iou_type == "segm":
            return self.prepare_for_coco_segmentation(predictions)
        else:
            raise ValueError(f"Unknown IoU type: {iou_type}")

    def prepare_for_coco_detection(self, predictions):
        coco_results = []

        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = convert_to_xywh(prediction['boxes']).tolist()
            scores = prediction['scores'].tolist()
            labels = prediction['labels'].tolist()

            coco_results.extend([{
                "image_id": original_id,
                "category_id": labels[k],
                "bbox": box,
                "score": scores[k]
            } for k, box in enumerate(boxes)])

        return coco_results

    def prepare_for_coco_segmentation(self, predictions):
        coco_results = []

        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            scores = prediction['scores'].tolist()
            labels = prediction['labels'].tolist()
            masks = prediction['masks'] > 0.5

            rles = [pycocotools.mask.encode(np.array(mask[0, :, :, np.newaxis], order='F'))[0] for mask in masks]

            for rle in rles:
                rle["counts"] = rle["counts"].decode("utf-8")

            coco_results.extend([{
                "image_id": original_id,
                "category_id": labels[k],
                "segmentation": rle,
                "score": scores[k]
            } for k, rle in enumerate(rles)])

        return coco_results

    def synchronize_between_processes(self):
        for iou_type in self.iou_types:
            self.eval_imgs[iou_type] = np.concatenate(self.eval_imgs[iou_type], 2)
            create_common_coco_eval(self.coco_eval[iou_type], self.img_ids, self.eval_imgs[iou_type])

    def accumulate(self):
        for coco_eval in self.coco_eval.values():
            coco_eval.accumulate()

    def summarize(self):
        temp_stats = None

        for iou_type, coco_eval in self.coco_eval.items():
            lgblkb_tools.logger.info(f"[INFO] IoU metric: {iou_type}")
            coco_eval.summarize()

            if iou_type == "segm":
                temp_stats = coco_eval.stats

        return temp_stats


def createIndex(self):
    anns, cats, imgs = {}, {}, {}
    imgToAnns, catToImgs = collections.defaultdict(list), collections.defaultdict(list)

    if 'annotations' in self.dataset:
        for ann in self.dataset['annotations']:
            imgToAnns[ann['image_id']].append(ann)
            anns[ann['id']] = ann

    if 'images' in self.dataset:
        for img in self.dataset['images']:
            imgs[img['id']] = img

    if 'categories' in self.dataset:
        for cat in self.dataset['categories']:
            cats[cat['id']] = cat

    if 'annotations' in self.dataset and 'categories' in self.dataset:
        for ann in self.dataset['annotations']:
            catToImgs[ann['category_id']].append(ann['image_id'])

    self.anns = anns
    self.imgToAnns = imgToAnns
    self.catToImgs = catToImgs
    self.imgs = imgs
    self.cats = cats


def loadRes(self, resFile):
    res = pycocotools.coco.COCO()
    res.dataset['images'] = [img for img in self.dataset['images']]

    if isinstance(resFile, torch._six.string_classes):
        anns = json.load(open(resFile))
    elif type(resFile) == np.ndarray:
        anns = self.loadNumpyAnnotations(resFile)
    else:
        anns = resFile

    assert type(anns) == list, 'results in not an array of objects'
    annsImgIds = [ann['image_id'] for ann in anns]
    assert set(annsImgIds) == (set(annsImgIds) & set(self.getImgIds())), \
        'Results do not correspond to current coco set'

    if 'caption' in anns[0]:
        imgIds = set([img['id'] for img in res.dataset['images']]) & set([ann['image_id'] for ann in anns])
        res.dataset['images'] = [img for img in res.dataset['images'] if img['id'] in imgIds]

        for id, ann in enumerate(anns):
            ann['id'] = id + 1

    elif 'bbox' in anns[0] and not anns[0]['bbox'] == []:
        res.dataset['categories'] = copy.deepcopy(self.dataset['categories'])

        for id, ann in enumerate(anns):
            bb = ann['bbox']
            x1, x2, y1, y2 = [bb[0], bb[0] + bb[2], bb[1], bb[1] + bb[3]]

            if 'segmentation' not in ann:
                ann['segmentation'] = [[x1, y1, x1, y2, x2, y2, x2, y1]]
            ann['area'] = bb[2] * bb[3]
            ann['id'] = id + 1
            ann['iscrowd'] = 0
    elif 'segmentation' in anns[0]:
        res.dataset['categories'] = copy.deepcopy(self.dataset['categories'])

        for id, ann in enumerate(anns):
            ann['area'] = pycocotools.mask.area(ann['segmentation'])
            if 'bbox' not in ann:
                ann['bbox'] = pycocotools.mask.toBbox(ann['segmentation'])
            ann['id'] = id + 1
            ann['iscrowd'] = 0

    res.dataset['annotations'] = anns
    createIndex(res)
    return res


def is_dist_avail_and_initialized():
    if not torch.distributed.is_available():
        return False
    if not torch.distributed.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return torch.distributed.get_world_size()


def all_gather(data):
    world_size = get_world_size()

    if world_size == 1:
        return [data]

    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    local_size = torch.tensor([tensor.numel()], device="cuda")
    size_list = [torch.tensor([0], device="cuda") for _ in range(world_size)]
    torch.distributed.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    tensor_list = []

    for _ in size_list:
        tensor_list.append(torch.empty((max_size,), dtype=torch.uint8, device="cuda"))
        
    if local_size != max_size:
        padding = torch.empty(size=(max_size - local_size,), dtype=torch.uint8, device="cuda")
        tensor = torch.cat((tensor, padding), dim=0)

    torch.distributed.all_gather(tensor_list, tensor)
    data_list = []

    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def merge(img_ids, eval_imgs):
    all_img_ids = all_gather(img_ids)
    all_eval_imgs = all_gather(eval_imgs)
    merged_img_ids = []

    for p in all_img_ids:
        merged_img_ids.extend(p)

    merged_eval_imgs = []

    for p in all_eval_imgs:
        merged_eval_imgs.append(p)

    merged_img_ids = np.array(merged_img_ids)
    merged_eval_imgs = np.concatenate(merged_eval_imgs, 2)
    merged_img_ids, idx = np.unique(merged_img_ids, return_index=True)
    merged_eval_imgs = merged_eval_imgs[..., idx]
    return merged_img_ids, merged_eval_imgs


def create_common_coco_eval(coco_eval, img_ids, eval_imgs):
    img_ids, eval_imgs = merge(img_ids, eval_imgs)
    img_ids = list(img_ids)
    eval_imgs = list(eval_imgs.flatten())
    coco_eval.evalImgs = eval_imgs
    coco_eval.params.imgIds = img_ids
    coco_eval._paramsEval = copy.deepcopy(coco_eval.params)


def evaluate(self):
    p = self.params

    if p.useSegm is not None:
        p.iouType = 'segm' if p.useSegm == 1 else 'bbox'
        lgblkb_tools.logger.error(f"useSegm (deprecated) is not None. Running {p.iouType} evaluation.")

    p.imgIds = list(np.unique(p.imgIds))

    if p.useCats:
        p.catIds = list(np.unique(p.catIds))

    p.maxDets = sorted(p.maxDets)
    self.params = p
    self._prepare()
    catIds = p.catIds if p.useCats else [-1]

    computeIoU = None

    if p.iouType == 'segm' or p.iouType == 'bbox':
        computeIoU = self.computeIoU

    self.ious = {
        (imgId, catId): computeIoU(imgId, catId)
        for imgId in p.imgIds
        for catId in catIds}

    evaluateImg = self.evaluateImg
    maxDet = p.maxDets[-1]
    evalImgs = [
        evaluateImg(imgId, catId, areaRng, maxDet)
        for catId in catIds
        for areaRng in p.areaRng
        for imgId in p.imgIds
    ]
    evalImgs = np.asarray(evalImgs).reshape(len(catIds), len(p.areaRng), len(p.imgIds))
    self._paramsEval = copy.deepcopy(self.params)
    return p.imgIds, evalImgs


def convert_to_xywh(boxes):
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymax), dim=1)
