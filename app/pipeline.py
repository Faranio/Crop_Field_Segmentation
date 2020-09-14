import os

import matplotlib.pyplot as plt
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
def get_instance_segmentation_model(num_classes):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    return model


@logger.trace()
def crop_tif(tif_file, width=15000, height=15000, out_dir='Temp'):
    try:
        os.mkdir(out_dir)
    except FileExistsError:
        pass

    src = rasterio.open(tif_file)
    tif_name = 0
    max_left, max_up = src.transform * (0, 0)
    max_right, max_bottom = src.transform * (src.width, src.height)
    left, up = max_left, max_up
    tile_count = 0
    while True:
        tile_count += 1
        if tile_count % 25 == 0:
            logger.debug("tile_count: %s", tile_count)
        # right = left + width
        # bottom = up - height

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

        dest = rasterio.open(os.path.join(out_dir, f"{tif_name}.tif"), "w", **out_meta)
        tif_name += 1
        dest.write(out_image)

        left += width / 2

        if left >= max_right:
            left = max_left
            up -= height / 2

        if up <= max_bottom:
            break


@logger.trace()
def process_tile(working_dir, tile_path, model, confidence, threshold):
    logger.debug("tile_path: %s", tile_path)
    uint8_type = True
    tile = rasterio.open(os.path.join(working_dir, tile_path))

    if tile.dtypes[0] != 'uint8':
        uint8_type = False

    array = np.dstack((tile.read(4), tile.read(3), tile.read(2)))
    array = np.nan_to_num(array)

    if not uint8_type:
        array = (array * 255 / np.max(array)).astype('uint8')

    image = Image.fromarray(array)
    tensor = transforms.ToTensor()(image)

    with torch.no_grad():
        prediction = model([tensor.to(device)])

    temp = []

    for i in range(len(prediction[0]['masks'])):
        if prediction[0]['scores'][i] > confidence:
            mask = prediction[0]['masks'][i, 0].mul(255).byte().cpu().numpy()
            mask[mask < threshold] = 0
            mask[mask >= threshold] = 1
            # mask = mask.tolist()
            temp.append(mask)

    # masks[tile_path] = temp
    return temp


@logger.trace()
def tile_pipeline(image_path, model, confidence, threshold, working_dir='Temp'):
    crop_tif(image_path, out_dir=working_dir)
    masks = {}

    for tile_path in os.listdir(working_dir):
        masks[tile_path] = process_tile(working_dir, tile_path, model, confidence, threshold)
    return masks


@logger.trace()
def image_pipeline(image_path, model, confidence, threshold):
    masks = {image_path: []}
    uint8_type = True

    image = rasterio.open(image_path)

    if image.dtypes[0] != 'uint8':
        uint8_type = False

    array = np.dstack((image.read(4), image.read(3), image.read(2)))
    array = np.nan_to_num(array)

    if not uint8_type:
        array = (array * 255 / np.max(array)).astype('uint8')

    image = Image.fromarray(array)
    tensor = transforms.ToTensor()(image)

    with torch.no_grad():
        prediction = model([tensor.to(device)])

    for i in range(len(prediction[0]['masks'])):
        if prediction[0]['scores'][i] > confidence:
            mask = prediction[0]['masks'][i, 0].mul(255).byte().cpu().numpy()
            mask[mask < threshold] = 0
            mask[mask >= threshold] = 1
            # mask = mask.tolist()
            masks[image_path].append(mask)

    return masks


@logger.trace()
def remove_temp_files():
    main_dir = "Temp"

    for file in os.listdir(main_dir):
        os.remove(os.path.join(main_dir, file))


@logger.trace()
def get_mask_info(image_path, model_path, confidence=0.6, threshold=100,
                  working_dir='Temp'):  # <----- previously this function was named main()
    model = get_instance_segmentation_model(2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    convert_to_epsg(image_path, image_path)
    image = rasterio.open(image_path)

    if image.bounds[2] - image.bounds[0] > 20000 or image.bounds[3] - image.bounds[1] > 20000:
        masks = tile_pipeline(image_path, model, confidence, threshold, working_dir=working_dir)
    else:
        masks = image_pipeline(image_path, model, confidence, threshold)

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
    out_filepath = working_folder['combined_bands.tiff']
    gm.gdal_merge(*band_paths, separate=True, out_filepath=out_filepath)
    logger.debug("out_filepath: %s", out_filepath)

    mask_info = get_mask_info(image_path=out_filepath,
                              confidence=0.4,
                              model_path=model_path,
                              working_dir=working_folder['tilings'])
    for k, masks, in mask_info.items():
        logger.debug("k: %s", k)
        logger.debug("len(masks): %s", len(masks))
        for mask_i, mask in enumerate(masks):
            logger.debug("mask.shape: %s", mask.shape)
            plt.imshow(mask)
            plt.savefig(f'{get_name(k)}_{mask_i}.png')
            plt.close()

    pass


if __name__ == "__main__":
    main()
