import lgblkb_tools
import math
import os
import pickle
import sys
import time
import torch
import torch.utils.data
# import wandb

import coco_utils
import config
import metric_logger
import transforms
import utils

from dataset import FieldsDataset


class TrainingPipeline:
    def __init__(self, device, num_classes, train_folder, valid_folder, test_folder=None, batch_size=2):
        self._device = device
        self.train_folder = train_folder
        self.valid_folder = valid_folder
        self.test_folder = test_folder
        self.num_classes = num_classes
        self.batch_size = batch_size
        self._dataset_train = None
        self._dataset_valid = None
        self._dataset_test = None
        self.data_loader_train = None
        self.data_loader_valid = None
        self.data_loader_test = None
        self.model = None
        self.train_coco_ds = None
        self.valid_coco_ds = None
        self.test_coco_ds = None
        self.iou_types = None
        self._params = None
        self.valid_coco_evaluator = None
        self.optimizer = None
        self.lr_scheduler = None
        self.metric_logger_var = None

        self.max_mAP = 52.521
        lgblkb_tools.logger.info(f"Device used is: {self._device}")

        self.initialize_datasets()
        self.initialize_data_loaders()

        if not os.path.exists(config.data_folder['Train']['coco_train.pickle']):
            coco_utils.save_as_coco_dataset(self.data_loader_train, self.data_loader_valid, self.data_loader_test)

    def initialize_datasets(self):
        self._dataset_train = FieldsDataset(self.train_folder, transforms.get_train_transform())
        self._dataset_valid = FieldsDataset(self.valid_folder, transforms.get_test_transform())
        lgblkb_tools.logger.info(f"Length of Training dataset: {len(self._dataset_train)}")
        lgblkb_tools.logger.info(f"Length of Validation dataset: {len(self._dataset_valid)}")

        if self.test_folder is not None:
            self._dataset_test = FieldsDataset(self.test_folder, transforms.get_test_transform())
            lgblkb_tools.logger.info(f"Length of Testing dataset: {len(self._dataset_test)}")

    def initialize_data_loaders(self):
        self.data_loader_train = torch.utils.data.DataLoader(self._dataset_train,
                                                             batch_size=self.batch_size,
                                                             shuffle=True,
                                                             num_workers=self.batch_size,
                                                             collate_fn=utils.collate_fn)
        self.data_loader_valid = torch.utils.data.DataLoader(self._dataset_valid,
                                                             batch_size=self.batch_size,
                                                             shuffle=False,
                                                             num_workers=self.batch_size,
                                                             collate_fn=utils.collate_fn)

        if self.test_folder is not None:
            self.data_loader_test = torch.utils.data.DataLoader(self._dataset_test,
                                                                batch_size=self.batch_size,
                                                                shuffle=False,
                                                                num_workers=self.batch_size,
                                                                collate_fn=utils.collate_fn)

    def initialize_model(self):
        self.model = utils.get_instance_segmentation_model(self.num_classes)
        self.model.load_state_dict(torch.load(f"./data/Model/mAP_{str(self.max_mAP)}.pt"))
        self.model.to(self._device)
        # self.model = torch.nn.DataParallel(self.model)
        self.iou_types = ['bbox', 'segm']
        self._params = [p for p in self.model.parameters() if p.requires_grad]

    def initialize_tools(self, base_lr, max_lr):
        # wandb.init(project="Crop_Field_Segmentation")
        self.initialize_model()
        self.optimizer = torch.optim.SGD(self._params,
                                         lr=max_lr,
                                         momentum=0.9,
                                         weight_decay=0.0005)
        self.lr_scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=base_lr, max_lr=max_lr,
                                                              step_size_up=1, step_size_down=len(self._dataset_train)
                                                                                             // self.batch_size)
        self.valid_coco_ds = pickle.load(open(lgblkb_tools.Folder(self.valid_folder[0])['coco_valid.pickle'], 'rb'))
        lgblkb_tools.logger.info("Starting the training process")
        self.metric_logger_var = metric_logger.MetricLogger(delimiter="  ")
        self.metric_logger_var.add_meter('lr', metric_logger.SmoothedValue(window_size=1, fmt='{value:.6f}'))

        del self._dataset_train
        del self._dataset_valid
        del self._dataset_test

    def get_eval_stats(self, metric_temp_logger):
        metric_temp_logger.synchronize_between_processes()
        lgblkb_tools.logger.info(f"Averaged stats: {metric_temp_logger}")
        self.valid_coco_evaluator.synchronize_between_processes()
        self.valid_coco_evaluator.accumulate()
        stats = self.valid_coco_evaluator.summarize()
        # wandb.log({'Mask mAP': stats[0] * 100})
        return stats[0]

    def train(self, base_lr=0.000005, max_lr=0.005, num_epochs=50, print_freq=10):
        self.initialize_tools(base_lr=base_lr, max_lr=max_lr)

        for epoch in range(num_epochs):
            self.model.train()
            header = f'Epoch: [{epoch + 1}]'

            for iter_i, (images, targets) in enumerate(
                    self.metric_logger_var.log_every(self.data_loader_train, print_freq,
                                                     header)):
                images = list(image.to(self._device) for image in images)
                targets = [{k: v.to(self._device) for k, v in t.items()} for t in targets]
                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                loss_dict_reduced = metric_logger.reduce_dict(loss_dict)
                losses_reduced = sum(loss for loss in loss_dict_reduced.values())
                loss_value = losses_reduced.item()
                # wandb.log({'Train Loss': loss_value})

                if not math.isfinite(loss_value):
                    lgblkb_tools.logger.error(f"Loss is {loss_value}, stopping training.")
                    lgblkb_tools.logger.error(f"{loss_dict_reduced}")
                    sys.exit(1)

                self.optimizer.zero_grad()
                losses.backward()
                self.optimizer.step()
                self.lr_scheduler.step()

                self.metric_logger_var.update(loss=losses_reduced, **loss_dict_reduced)
                self.metric_logger_var.update(lr=self.optimizer.param_groups[0]["lr"])

            lgblkb_tools.logger.info("Evaluating...")

            self.valid_coco_evaluator = coco_utils.CocoEvaluator(self.valid_coco_ds, self.iou_types)

            with torch.no_grad():
                self.model.eval()
                metric_temp_logger = metric_logger.MetricLogger(delimiter="  ")
                header = "Test: "

                for image, targets in metric_temp_logger.log_every(self.data_loader_valid, print_freq, header):
                    image = list(image.to(self._device) for image in image)
                    targets = [{k: v.to(self._device) for k, v in t.items()} for t in targets]
                    torch.cuda.synchronize()
                    model_time = time.time()
                    outputs = self.model(image)
                    outputs = [{k: v.to(torch.device('cpu')) for k, v in t.items()} for t in outputs]
                    model_time = time.time() - model_time
                    res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
                    evaluator_time = time.time()
                    self.valid_coco_evaluator.update(res)
                    evaluator_time = time.time() - evaluator_time
                    metric_temp_logger.update(model_time=model_time, evaluator_time=evaluator_time)

                current_mAP = self.get_eval_stats(metric_temp_logger) * 100

                if current_mAP > self.max_mAP:
                    lgblkb_tools.logger.info(f"\n\nSaving the model. Mask mAP: {current_mAP}%\n\n")
                    self.max_mAP = current_mAP
                    torch.save(self.model.state_dict(), "./data/Model/mAP_{:.3f}.pt".format(current_mAP))

        lgblkb_tools.logger.info("\n\n\nFinished!")


def main():
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    num_classes = 2
    batch_size = 4
    train_folder_paths = config.data_folder.glob_search("**/Train")
    valid_folder_paths = config.data_folder.glob_search("**/Valid")
    test_folder_paths = config.data_folder.glob_search("**/Test")

    train_pipeline = TrainingPipeline(device=device,
                                      num_classes=num_classes,
                                      train_folder=train_folder_paths,
                                      valid_folder=valid_folder_paths,
                                      test_folder=test_folder_paths,
                                      batch_size=batch_size)
    train_pipeline.train()


if __name__ == "__main__":
    main()
