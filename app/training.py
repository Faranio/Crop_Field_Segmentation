import wandb
import pickle
import math
import sys
import time
import geojson
from pathlib import Path

import shapely.affinity as shaff
import shapely.geometry as shg
from functools import partial

from lgblkb_tools import Folder, logger
import numpy as np
import pandas as pd
from lgblkb_tools.gdal_datasets import GdalMan, DataSet
from lgblkb_tools.pathify import get_name
from lgblkb_tools.telegram_notify import TheChat
from telegram import Bot
import rasterio.plot
import rasterio.mask
import rasterio.features
import geopandas as gpd
from random import random
from shapely.strtree import STRtree
from shapely.geometry import Polygon
from rasterio.warp import calculate_default_transform, reproject, Resampling

import os
from app.black_box import utils
import torch
import random
import rasterio
from matplotlib import patches
from app.black_box.coco_eval import CocoEvaluator
from matplotlib import pyplot as plt
from PIL import Image

# region pandas options:
# pd.set_option('display.max_rows', None)
from app.app import data_folder, cache_folder
from app.torched import FieldsDataset, get_instance_segmentation_model, get_coco_api_from_dataset, \
    _get_iou_types, get_transform

pd.set_option('display.max_columns', None)
# pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 25)
# endregion

notify = partial(TheChat(r'-432356237', bot=Bot('1084990340:AAFCm3odQOHCPDTqTStT_KQwidCaOrXwJNc')).send_message)


def get_records(dataset_dir, image_name, image_id, record_id):
    image, ext = os.path.splitext(image_name)
    shp_file = image + ".shp"
    shp_file = gpd.read_file(os.path.join(dataset_dir, "Coordinates", shp_file))
    image = rasterio.open(os.path.join(dataset_dir, "Photos", image_name))

    left, bottom, right, top = image.bounds
    img_height, img_width = image.shape
    meters_in_pixel = (right - left) / img_width

    records = []

    for i in range(len(shp_file)):
        if shp_file.geometry[i] is None:
            continue

        record = {
            "segmentation": [],
            "area": 0,
            "iscrowd": 0,
            "image_id": image_id,
            "bbox": [],
            "category_id": 1,
            "id": record_id
        }

        x = (shp_file.geometry[i].bounds[0] - left) / meters_in_pixel
        y = (top - shp_file.geometry[i].bounds[3]) / meters_in_pixel
        bbox_width = (shp_file.geometry[i].bounds[2] - shp_file.geometry[i].bounds[0]) / meters_in_pixel
        bbox_height = (shp_file.geometry[i].bounds[3] - shp_file.geometry[i].bounds[1]) / meters_in_pixel

        record['bbox'] = [x, y, bbox_width, bbox_height]

        if shp_file.geometry[i].type == "Polygon":

            if shp_file.geometry[i].boundary.type == "MultiLineString":
                total_area = 0

                for j in range(len(shp_file.geometry[i].boundary)):
                    coord_len = len(shp_file.geometry[i].boundary[j].coords.xy[0])

                    temp = []

                    for k in range(coord_len):
                        x_coord = (shp_file.geometry[i].boundary[j].coords.xy[0][k] - left) / meters_in_pixel
                        y_coord = (top - bottom) / meters_in_pixel - (
                                shp_file.geometry[i].boundary[j].coords.xy[1][k] - bottom) / meters_in_pixel

                        temp.append((x_coord, y_coord))

                    poly = Polygon(temp)
                    total_area += poly.area

                    temp = []

                    for k in range(coord_len):
                        x_coord = (shp_file.geometry[i].boundary[j].coords.xy[0][k] - left) / meters_in_pixel
                        y_coord = (top - bottom) / meters_in_pixel - (
                                shp_file.geometry[i].boundary[j].coords.xy[1][k] - bottom) / meters_in_pixel

                        temp.append(x_coord)
                        temp.append(y_coord)

                    record['segmentation'].append(temp)

                record['area'] = total_area
            else:
                coord_len = len(shp_file.geometry[i].boundary.coords.xy[0])

                temp = []

                for j in range(coord_len):
                    x_coord = (shp_file.geometry[i].boundary.coords.xy[0][j] - left) / meters_in_pixel
                    y_coord = (top - bottom) / meters_in_pixel - (
                            shp_file.geometry[i].boundary.coords.xy[1][j] - bottom) / meters_in_pixel

                    temp.append((x_coord, y_coord))

                poly = Polygon(temp)
                record['area'] = poly.area

                temp = []

                for j in range(coord_len):
                    x_coord = (shp_file.geometry[i].boundary.coords.xy[0][j] - left) / meters_in_pixel
                    y_coord = (top - bottom) / meters_in_pixel - (
                            shp_file.geometry[i].boundary.coords.xy[1][j] - bottom) / meters_in_pixel

                    temp.append(x_coord)
                    temp.append(y_coord)

                record['segmentation'].append(temp)
        elif shp_file.geometry[i].type == "MultiPolygon":
            total_area = 0

            for j in range(len(shp_file.geometry[i])):
                coord_len = len(shp_file.geometry[i][j].boundary.coords.xy[0])

                temp = []

                for k in range(coord_len):
                    x_coord = (shp_file.geometry[i][j].boundary.coords.xy[0][k] - left) / meters_in_pixel
                    y_coord = (top - bottom) / meters_in_pixel - (
                            shp_file.geometry[i][j].boundary.coords.xy[1][k] - bottom) / meters_in_pixel

                    temp.append((x_coord, y_coord))

                poly = Polygon(temp)
                total_area += poly.area

                temp = []

                for k in range(coord_len):
                    x_coord = (shp_file.geometry[i][j].boundary.coords.xy[0][k] - left) / meters_in_pixel
                    y_coord = (top - bottom) / meters_in_pixel - (
                            shp_file.geometry[i][j].boundary.coords.xy[1][k] - bottom) / meters_in_pixel

                    temp.append(x_coord)
                    temp.append(y_coord)

                record['segmentation'].append(temp)

            record['area'] = total_area

        record_id += 1
        records.append(record)

    return records, record_id


# Defining a method for plotting .tif and .shp file together on the same figure

def plot_tif_shp(tif_file="new.tif", shp_file="fields_oktyabr.shp"):
    '''
    Plotting .tif and .shp files together on one figure.
    '''
    fig, ax = plt.subplots(1, figsize=(5, 5))
    src = rasterio.open(tif_file)
    ax = rasterio.plot.show(src, ax=ax)
    shp_file = gpd.read_file(shp_file)

    for i in range(len(shp_file.geometry)):
        if shp_file.geometry[i].type == "MultiPolygon":
            for j in range(len(shp_file["geometry"][i])):
                if shp_file["geometry"][i][j].boundary.type == "MultiLineString":
                    for k in range(len(shp_file["geometry"][i][j].boundary)):
                        bbox = shp_file.geometry[i][j].boundary[k].bounds
                        location = (bbox[0], bbox[1])
                        width = bbox[2] - bbox[0]
                        height = bbox[3] - bbox[1]
                        rect = patches.Rectangle(location, width, height, edgecolor='b', facecolor='None')
                        ax.plot(shp_file.geometry[i][j].boundary[k].coords.xy[0],
                                shp_file.geometry[i][j].boundary[k].coords.xy[1], color='r', linewidth=3)
                        ax.add_patch(rect)
                else:
                    bbox = shp_file.geometry[i][j].boundary.bounds
                    location = (bbox[0], bbox[1])
                    width = bbox[2] - bbox[0]
                    height = bbox[3] - bbox[1]
                    rect = patches.Rectangle(location, width, height, edgecolor='b', facecolor='None')
                    ax.plot(shp_file.geometry[i][j].boundary.coords.xy[0],
                            shp_file.geometry[i][j].boundary.coords.xy[1], color='r', linewidth=3)
                    ax.add_patch(rect)
        elif shp_file.geometry[i].type == "Polygon":
            if shp_file.geometry[i].boundary.type == "MultiLineString":
                for j in range(len(shp_file["geometry"][i].boundary)):
                    bbox = shp_file.geometry[i].boundary[j].bounds
                    location = (bbox[0], bbox[1])
                    width = bbox[2] - bbox[0]
                    height = bbox[3] - bbox[1]
                    rect = patches.Rectangle(location, width, height, edgecolor='b', facecolor='None')
                    ax.plot(shp_file.geometry[i].boundary[j].coords.xy[0],
                            shp_file.geometry[i].boundary[j].coords.xy[1], color='r', linewidth=3)
                    ax.add_patch(rect)
            else:
                bbox = shp_file.geometry[i].boundary.bounds
                location = (bbox[0], bbox[1])
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                rect = patches.Rectangle(location, width, height, edgecolor='b', facecolor='None')
                ax.plot(shp_file.geometry[i].boundary.coords.xy[0], shp_file.geometry[i].boundary.coords.xy[1],
                        color='r', linewidth=3)
                ax.add_patch(rect)

    plt.show()


# Defining a method for converting the .tif file to the EPSG:3857 format

def convert_to_epsg(tif_file="RGB.tif", out_tif_file="new.tif"):
    src = rasterio.open(tif_file)

    dst_crs = 'EPSG:3857'

    transform, width, height = calculate_default_transform(src.crs, dst_crs, src.width, src.height, *src.bounds)
    kwargs = src.meta.copy()
    kwargs.update({
        'crs': dst_crs,
        'transform': transform,
        'width': width,
        'height': height
    })

    dst = rasterio.open(out_tif_file, 'w', **kwargs)

    for i in range(1, src.count + 1):
        reproject(source=rasterio.band(src, i),
                  destination=rasterio.band(dst, i),
                  src_transform=src.transform,
                  src_crs=src.crs,
                  dst_transform=transform,
                  dst_crs=dst_crs,
                  resampling=Resampling.nearest)


def get_georeferenced_vectors(vectors: gpd.GeoDataFrame, transform, raster_shape):
    vectors.geometry = vectors.geometry.map(
        lambda g: shaff.scale(g, xfact=transform[1], yfact=abs(transform[5]), origin=(0, 0))
    )
    vectors.geometry = vectors.geometry.map(
        lambda g: shaff.translate(g, transform[0], transform[3] + transform[5] * raster_shape[0]))
    return vectors


def vectorize(array, rasterio_transform, out_geojson_path=''):
    logger.debug("array:\n%s", array)
    # array=DataSet(raster_filepath).array
    image = Image.fromarray(array)
    # temp_folder = Folder.mkdtemp()
    temp_folder = cache_folder['vectors']
    raster_filepath = Path(temp_folder['raster_file.bmp'])
    if out_geojson_path:
        out_geojson_path = Path(out_geojson_path)
    else:
        out_geojson_path = Path(raster_filepath).with_suffix('.geojson')
    image.save(raster_filepath)
    cmd = f"potrace -b geojson {raster_filepath} -o {out_geojson_path} && " \
          f"cat {out_geojson_path} | simplify-geojson -t 5 > temp.geojson && " \
          f"mv temp.geojson {out_geojson_path}"
    logger.debug("cmd: %s", cmd)
    os.system(cmd)
    geometry = gpd.read_file(out_geojson_path).iloc[0].geometry
    logger.debug("geometry: %s", geometry)
    vectors = gpd.GeoDataFrame([geometry], crs='epsg:3857', columns=['geometry'])
    tif_polygon = get_georeferenced_vectors(
        vectors, [rasterio_transform[2], *rasterio_transform[:2],
                  rasterio_transform[5], *rasterio_transform[3:5]],
        array.shape).iloc[0].geometry

    # temp_folder.delete()
    return tif_polygon


# Cropping the .tif image with given width and height into tiles (.tif and corresponding .shp files)
@logger.trace()
def crop_tif(label_geometries, tif_file, out_folder, width=5000, height=5000):
    ds = DataSet(tif_file)
    src = rasterio.open(tif_file)

    # tiff_masks = src.read_masks()
    vector_infos = list(rasterio.features.shapes((np.flipud(ds.array) != 0).astype(np.int32)))

    polygons = [shg.shape(vector) for vector, value in vector_infos if value > 0]
    assert len(polygons) == 1
    polygon = polygons[0]

    temp_folder = cache_folder['vectors']
    raw_geom_path = temp_folder['raw_geom.geojson']
    geojson.dump(geojson.Feature(geometry=polygon), open(raw_geom_path, 'w'))

    out_json_path = temp_folder["simplified_geom.geojson"]
    os.system(f'simplify-geojson {raw_geom_path} -t 5 > {out_json_path}')
    unreferenced_poly = gpd.read_file(out_json_path).iloc[0].geometry
    valid_image_polygon = get_georeferenced_vectors(gpd.GeoDataFrame([unreferenced_poly],
                                                                     crs='epsg:3857', columns=['geometry']),
                                                    ds.transform, ds.array.shape).iloc[0].geometry

    tree = STRtree(label_geometries.geometry)
    valid_fields = list()
    intersecting_field_multipolygons = tree.query(valid_image_polygon)
    for intersecting_field_multipolygon in intersecting_field_multipolygons:
        resultant_poly = intersecting_field_multipolygon.intersection(valid_image_polygon)
        if resultant_poly.area >= intersecting_field_multipolygon.area / 10:
            valid_fields.append(resultant_poly)

    tree = STRtree(valid_fields)

    valid_image_polygon: shg.Polygon = valid_image_polygon
    max_left, max_bottom, max_right, max_up = valid_image_polygon.bounds
    left, top = max_left, max_up
    # sys.exit()
    # max_left, max_up = src.transform * (0, 0)
    # max_right, max_bottom = src.transform * (src.width, src.height)

    print("Adding shapes...")
    count = 0

    while True:
        if count % 100 == 0:
            logger.debug("count: %s", count)

        right = left + width
        bottom = top - height

        tile_poly = Polygon([[left, bottom], [left, top], [right, top], [right, bottom]])
        intersections = tree.query(tile_poly)

        temp_coords = []
        for multipolygon in intersections:
            # for polygon in multipolygon:
            coord = multipolygon.intersection(tile_poly)
            if coord.area >= multipolygon.area / 10:
                temp_coords.append(coord)

        if len(temp_coords):
            temp = [{'type': 'Polygon', 'coordinates': [[(left, top, 0.0),
                                                         (right, top, 0.0),
                                                         (right, bottom, 0.0),
                                                         (left, bottom, 0.0)]]}]

            out_image, out_transform = rasterio.mask.mask(src, temp, crop=True)
            out_meta = src.meta

            out_meta.update({"driver": "GTiff",
                             "height": out_image.shape[1],
                             "width": out_image.shape[2],
                             "transform": out_transform})

            if random() > 0.20:
                # save_folder = os.path.join(out_folder, "Train")
                save_folder = out_folder['Train']
            else:
                save_folder = out_folder['Test']
            tile_name = f"tile_{count}"
            image_path = save_folder['Photos'][tile_name + '.tiff']
            if not os.path.exists(image_path):
                dest = rasterio.open(image_path, "w", **out_meta)
                dest.write(out_image)

            temp_coords = gpd.GeoDataFrame(geometry=temp_coords)
            shp_filepath = save_folder['Coordinates'][tile_name + '.shp']
            if not os.path.exists(shp_filepath):
                temp_coords.to_file(shp_filepath, driver='ESRI Shapefile')
            count += 1

        left += width / 2

        if left >= max_right:
            left = max_left
            top -= height / 2

        if top <= max_bottom:
            break

    print(f"Tiles created - {count - 1}")


training_folder = data_folder['training']
tiles_folder: Folder = training_folder['tiles']
label_geometries = gpd.read_file(data_folder['final.geojson'])


@logger.trace()
def create_dataset():
    safe_folder_paths = data_folder['raw2'].children

    gm = GdalMan(q=True, lazy=True)

    for safe_folder_path in safe_folder_paths:
        logger.debug("safe_folder_path: %s", safe_folder_path)
        safe_folder = Folder(safe_folder_path)
        working_folder = Folder(cache_folder.get_filepath(safe_folder.name))['training_data']
        band_paths = [safe_folder.glob_search(f'**/*_B0{band_num}_10m.jp2')[0] for band_num in [2, 3, 4, 8]]
        gm.gdal_merge(*band_paths, separate=True, out_filepath=working_folder['combined_bands.tiff'])
        gm.gdalwarp(gm.path, t_srs='epsg:3857', ot='Float32',
                    out_filepath=working_folder['epsg_converted.tiff'])
        # gm.gdal_translate(gm.path,working_folder['translated.tiff'],a_nodata='0')
        crop_tif(label_geometries=label_geometries, tif_file=gm.path,
                 out_folder=tiles_folder[get_name(safe_folder.path)], width=10000, height=10000)


@logger.trace()
def main():
    train_folder_paths = tiles_folder.glob_search('**/Train')
    test_folder_paths = tiles_folder.glob_search('**/Test')

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"[INFO] Device used is: {device}")
    num_classes = 2
    dataset = FieldsDataset(train_folder_paths, get_transform())
    dataset_test = FieldsDataset(test_folder_paths, get_transform())

    print(f"[INFO] Length of Training dataset: {len(dataset)}")
    print(f"[INFO] Length of Testing dataset: {len(dataset_test)}")

    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=1,
                                              shuffle=True,
                                              num_workers=0,
                                              collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(dataset_test,
                                                   batch_size=1,
                                                   shuffle=False,
                                                   num_workers=0,
                                                   collate_fn=utils.collate_fn)

    model = get_instance_segmentation_model(num_classes)
    # model.load_state_dict(torch.load("Models/model/MODEL_7.pt"))
    model.to(device)
    wandb.init(project="crop_field_segmentation")

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params,
                                lr=0.0005,
                                momentum=0.9,
                                weight_decay=0.0005)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    num_epochs = 15
    max_mAP = 0.05
    print_freq = 10

    # coco = get_coco_api_from_dataset(data_loader_test.dataset, )
    # pickle.dump(coco, open(data_folder['coco_obj.pickle'], 'wb'))
    coco = pickle.load(open(data_folder['coco_obj.pickle'], 'rb'))
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    print("[INFO] Starting the training process...")

    for epoch in range(num_epochs):

        # Train
        model.to(device)
        model.train()
        metric_logger = utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        header = 'Epoch: [{}]'.format(epoch + 1)

        lr_temp_scheduler = None
        if epoch == 0:
            warmup_factor = 1. / 1000
            warmup_iters = min(1000, len(data_loader) - 1)

            lr_temp_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

        for images, targets in metric_logger.log_every(data_loader, print_freq, header):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)

            losses = sum(loss for loss in loss_dict.values())

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = utils.reduce_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())

            loss_value = losses_reduced.item()
            wandb.log({'epoch': epoch, 'Train loss': loss_value})

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                print(loss_dict_reduced)
                sys.exit(1)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            if lr_temp_scheduler is not None:
                lr_temp_scheduler.step()

            metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        # Evaluate
        logger.debug('Evaluating...')
        with torch.no_grad():

            # n_threads = torch.get_num_threads()
            # torch.set_num_threads(1)
            # cpu_device = torch.device("cpu")
            # model.to(cpu_device)
            model.eval()
            metric_temp_logger = utils.MetricLogger(delimiter="  ")
            header = 'Test:'

            for image, targets in metric_temp_logger.log_every(data_loader_test, print_freq, header):
                image = list(img.to(device) for img in image)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                torch.cuda.synchronize()
                model_time = time.time()
                outputs = model(image)

                outputs = [{k: v.to(device) for k, v in t.items()} for t in outputs]
                model_time = time.time() - model_time

                res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
                evaluator_time = time.time()
                coco_evaluator.update(res)
                evaluator_time = time.time() - evaluator_time
                metric_temp_logger.update(model_time=model_time, evaluator_time=evaluator_time)

            # gather the stats from all processes
            metric_temp_logger.synchronize_between_processes()
            print("Averaged stats:", metric_temp_logger)
            coco_evaluator.synchronize_between_processes()

            # accumulate predictions from all images
            coco_evaluator.accumulate()
            stats = coco_evaluator.summarize()

            if stats[0] > max_mAP:
                print("\n\nSaving the model. Mask mAP: {:.3f}%\n\n".format(stats[0] * 100))
                max_mAP = stats[0]
                torch.save(model.state_dict(), "Max_Model.pt")

            # torch.set_num_threads(n_threads)

        lr_scheduler.step()

    print("\n\n\nFinished!")

    pass


if __name__ == '__main__':
    main()
