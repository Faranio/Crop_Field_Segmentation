from pathlib import Path

import more_itertools as mit
import os

import geopandas as god
import matplotlib.pyplot as plt
import json
import numpy as np
import rasterio
import rasterio.mask
import torch
import torchvision
from PIL import Image
from lgblkb_tools import Folder, logger
from lgblkb_tools.gdal_datasets import GdalMan
from lgblkb_tools.pathify import get_name
from rasterio.warp import calculate_default_transform, reproject, Resampling
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from app.app import s2_storage_folder, data_folder, cache_folder

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
os.environ['TORCH_HOME'] = data_folder.path


@logger.trace()
def get_instance_segmentation_model(num_classes):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    return model


model = get_instance_segmentation_model(2)


@logger.trace()
def convert_to_epsg(tif_file, out_tif_file):
    src = rasterio.open(tif_file)
    dst_crs = "EPSG:3857"
    transform, width, height = calculate_default_transform(src.crs,
                                                           dst_crs,
                                                           src.width,
                                                           src.height,
                                                           *src.bounds)
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


@logger.trace()
def crop_tif(tif_file, width=20000, height=20000, out_folder='Temp', limit=-1):
    out_folder = Folder(out_folder)
    src = rasterio.open(tif_file)
    max_left, max_up = src.transform * (0, 0)
    max_right, max_bottom = src.transform * (src.width, src.height)
    left, up = max_left, max_up
    tile_count = 0
    while True:
        tile_path = out_folder[f"{tile_count}.tiff"]
        if os.path.exists(tile_path): continue
        tile_count += 1
        if tile_count == limit:
            break

        if tile_count % 25 == 0:
            logger.debug("tile_count: %s", tile_count)

        temp = [{'type': 'Polygon', 'coordinates': [[(left, up, 0.0),
                                                     (left + width, up, 0.0),
                                                     (left + width, up - height, 0.0),
                                                     (left, up - height, 0.0)]]}]
        out_image, out_transform = rasterio.mask.mask(src, temp, crop=True)
        out_meta = src.meta
        out_meta.update({"driver": "GTiff",
                         "height": out_image.shape[1],
                         "width": out_image.shape[2],
                         "transform": out_transform})

        dest = rasterio.open(tile_path, "w", **out_meta)
        dest.write(out_image)

        left += width / 2

        if left >= max_right:
            left = max_left
            up -= height / 2

        if up <= max_bottom:
            break


@logger.trace()
def process_tile(working_dir, tile_path, confidence, threshold):
    logger.debug("tile_path: %s", tile_path)
    masks = {}
    uint8_type = True
    tile = rasterio.open(os.path.join(working_dir, tile_path))

    if tile.dtypes[0] != 'uint8':
        uint8_type = False

    array = np.dstack((tile.read(4), tile.read(3), tile.read(2)))
    array = np.nan_to_num(array)

    if not uint8_type:
        array = array.astype(np.float32, order='C') / 32768.0
        array = (array * 255 / np.max(array)).astype('uint8')

    image = Image.fromarray(array)
    tensor = transforms.ToTensor()(image)

    with torch.no_grad():
        prediction = model([tensor.to(device)])

    temp = []

    for i in range(len(prediction[0]['masks'])):
        if prediction[0]['scores'][i] > confidence:
            mask = prediction[0]['masks'][i, 0].mul(255).byte().cpu().numpy()
            mask = mask < threshold
            mask = mask.astype('uint8') * 255
            temp.append(mask)

    masks[tile_path] = temp
    return temp


@logger.trace()
def tile_pipeline(image_path, confidence, threshold, tile_width=20000, tile_height=20000, working_dir='Temp'):
    crop_tif(image_path, tile_width, tile_height, out_folder=working_dir, limit=10)
    masks = {}

    for tile_path in os.listdir(working_dir):
        masks[tile_path] = process_tile(working_dir, tile_path, confidence, threshold)
    return masks


@logger.trace()
def image_pipeline(image_path, confidence, threshold):
    masks = {image_path: []}
    uint8_type = True

    image = rasterio.open(image_path)

    if image.dtypes[0] != 'uint8':
        uint8_type = False

    array = np.dstack((image.read(4), image.read(3), image.read(2)))
    array = np.nan_to_num(array)

    if not uint8_type:
        array = array.astype(np.float32, order='C') / 32768.0
        array = (array * 255 / np.max(array)).astype('uint8')

    image = Image.fromarray(array)
    tensor = transforms.ToTensor()(image)

    with torch.no_grad():
        prediction = model([tensor.to(device)])

    for i in range(len(prediction[0]['masks'])):
        if prediction[0]['scores'][i] > confidence:
            mask = prediction[0]['masks'][i, 0].mul(255).byte().cpu().numpy()
            mask = mask < threshold
            mask = mask.astype('uint8') * 255
            masks[image_path].append(mask)

    return masks


@logger.trace()
def remove_temp_files():
    main_dir = "Temp"

    for file in os.listdir(main_dir):
        os.remove(os.path.join(main_dir, file))


@logger.trace()
def get_mask_info(image_path, model_path, confidence=0.6, threshold=100, tile_width=20000, tile_height=20000,
                  working_dir='Temp'):  # <----- previously this function was named main()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    convert_to_epsg(image_path, image_path)
    image = rasterio.open(image_path)

    if image.bounds[2] - image.bounds[0] > tile_width or image.bounds[3] - image.bounds[1] > tile_height:
        masks = tile_pipeline(image_path, confidence, threshold, tile_width, tile_height, working_dir=working_dir)
    else:
        masks = image_pipeline(image_path, confidence, threshold)

    # remove_temp_files()

    return masks


@logger.trace()
def main():
    model_path = data_folder['model']['MODEL_7.pt']
    safe_folder: Folder = Folder(s2_storage_folder['unzipped_scenes'].get_filepath(
        'S2A_MSIL2A_20200625T060641_N0214_R134_T43UDV_20200625T084444.SAFE'), reactive=False, assert_exists=True)
    band_paths = [safe_folder.glob_search(f'**/*_B0{band_num}_10m.jp2')[0] for band_num in [2, 3, 4, 8]]
    gm = GdalMan(q=True, lazy=True)
    working_folder = Folder(cache_folder.get_filepath(safe_folder.name))
    masks_folder = working_folder['Masks']
    out_filepath = working_folder['combined_bands.tiff']
    gm.gdal_merge(*band_paths, separate=True, out_filepath=out_filepath)
    logger.debug("out_filepath: %s", out_filepath)

    mask_info = get_mask_info(image_path=out_filepath,
                              confidence=0.6,
                              model_path=model_path,
                              tile_width=20000,
                              tile_height=20000,
                              working_dir=working_folder['tilings'])
    working_folder['tilings'].clear()
    for k, masks, in mask_info.items():
        logger.debug("k: %s", k)
        logger.debug("len(masks): %s", len(masks))
        for mask_i, mask in enumerate(masks):
            logger.debug("mask.shape: %s", mask.shape)
            image = Image.fromarray(mask)

            raster_filepath = Path(masks_folder[f'{get_name(k)}_{mask_i}.bmp'])
            image.save(raster_filepath)
            cmd = f"potrace -b geojson {raster_filepath} -o {Path(raster_filepath).with_suffix('.geojson')} &&" \
                  f"cat {Path(raster_filepath).with_suffix('.geojson')} | simplify-geojson -t 5 > temp.geojson &&" \
                  f"mv temp.geojson {Path(raster_filepath).with_suffix('.geojson')} &&" \
                  f"rm {raster_filepath}"
            logger.debug("cmd: %s", cmd)
            os.system(cmd)
            
            tile_path = Path(working_folder[f"tilings/{Path(raster_filepath).with_suffix('.tif')}"])
            
            src = rasterio.open(tile_path)
            image_left, image_bottom = src.bounds[:2]
            
            with open(f"{Path(raster_filepath).with_suffix('.geojson')}", 'r') as file:
                shp_file = json.load(file)

            for i in range(len(shp_file['features'])):
                for j in range(len(shp_file['features'][i]['geometry']['coordinates'])):
                    for l in range(len(shp_file['features'][i]['geometry']['coordinates'][j])):
                        x = shp_file['features'][i]['geometry']['coordinates'][j][l][0] * tile_width / src.shape[
                            0] + image_left
                        y = shp_file['features'][i]['geometry']['coordinates'][j][l][1] * tile_height / src.shape[
                            1] + image_bottom
                        shp_file['features'][i]['geometry']['coordinates'][j][l] = [x, y]

            with open(f"{Path(raster_filepath).with_suffix('.geojson')}", 'w+') as f:
                json.dump(shp_file, f, indent=2)
            
            os.remove(raster_filepath)


pass

if __name__ == "__main__":
    main()
