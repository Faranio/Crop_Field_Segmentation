import geopandas as gpd
import json
import numpy as np
import os
import rasterio
import rasterio.mask
import shapely
import shapely.errors
import torch

from lgblkb_tools import Folder, logger
from lgblkb_tools.pathify import get_name
from pathlib import Path
from PIL import Image
from rasterio.warp import calculate_default_transform, reproject, Resampling
from shapely.geometry import shape
from torchvision import transforms

from config import cache_folder, data_folder
from utils import get_instance_segmentation_model


class InstanceSegmentationModel:
    def __init__(self, model_checkpoint, num_classes, device='cpu'):
        self.num_classes = num_classes
        self.model_checkpoint = model_checkpoint
        self.device = device
        self.model = None
        self.initialize_model()

    def initialize_model(self):
        self.model = get_instance_segmentation_model(self.num_classes)
        self.model.load_state_dict(torch.load(self.model_checkpoint, map_location=torch.device(self.device)))
        self.model.eval()


def convert_crs(tif_file, out_tif_file, crs="EPSG:3857"):
    source_file = rasterio.open(tif_file)
    transform, width, height = calculate_default_transform(source_file.crs,
                                                           crs,
                                                           source_file.width,
                                                           source_file.height,
                                                           *source_file.bounds)
    kwargs = source_file.meta.copy()
    kwargs.update({
        'crs': crs,
        'transform': transform,
        'width': width,
        'height': height
    })

    destination_file = rasterio.open(out_tif_file, 'w', **kwargs)

    for band_idx in range(1, source_file.count + 1):
        reproject(
            source=rasterio.band(source_file, band_idx),
            destination=rasterio.band(destination_file, band_idx),
            src_transform=source_file.transform,
            src_crs=source_file.crs,
            dst_transform=transform,
            dst_crs=crs,
            resampling=Resampling.nearest
        )

    source_file.close()
    destination_file.close()


def convert_raster_to_vector(raster_filepath, mask):
    image = Image.fromarray(mask)
    image.save(raster_filepath)


def convert_vector_to_geojson(raster_filepath):
    geojson_filepath = Path(raster_filepath).with_suffix('.geojson')
    command = f"potrace -b geojson {raster_filepath} -o {geojson_filepath}"
    os.system(command)
    return geojson_filepath


def adjust_coordinates(geojson_coordinates, image_left, image_bottom, source_file, tile_width, tile_height):
    for feature_idx in range(len(geojson_coordinates['features'])):
        for coord_idx in range(len(geojson_coordinates['features'][feature_idx]['geometry']['coordinates'])):
            for i in range(len(geojson_coordinates['features'][feature_idx]['geometry']['coordinates'][coord_idx])):
                x = geojson_coordinates['features'][feature_idx]['geometry']['coordinates'][coord_idx][i][0] \
                    * tile_width / source_file.shape[0] + image_left
                y = geojson_coordinates['features'][feature_idx]['geometry']['coordinates'][coord_idx][i][1] \
                    * tile_height / source_file.shape[1] + image_bottom
                geojson_coordinates['features'][feature_idx]['geometry']['coordinates'][coord_idx][i] = [x, y]

    return geojson_coordinates


def save_geojson_coordinates(mask_idx, mask, masks_folder, tiles_folder, tile_path, tile_width, tile_height):
    raster_filepath = Path(masks_folder[f'{get_name(tile_path)}_{mask_idx}.bmp'])
    convert_raster_to_vector(raster_filepath, mask)
    geojson_filepath = convert_vector_to_geojson(raster_filepath)
    tile_path = tiles_folder[f'{get_name(tile_path)}.tiff']
    source_file = rasterio.open(tile_path)
    image_left, image_bottom = source_file.bounds[:2]

    with open(geojson_filepath, 'r') as file:
        geojson_coordinates = json.load(file)

    geojson_coordinates = adjust_coordinates(geojson_coordinates, image_left, image_bottom, source_file, tile_width,
                                             tile_height)

    with open(geojson_filepath, 'w+') as file:
        json.dump(geojson_coordinates, file, indent=2)

    source_file.close()


def crop_tif(tif_file, tile_width=20000, tile_height=20000, tile_stride_factor=2, out_folder='Temp'):
    out_folder = Folder(out_folder)
    source_file = rasterio.open(tif_file)
    max_left, max_top = source_file.transform * (0, 0)
    max_right, max_bottom = source_file.transform * (source_file.width, source_file.height)
    left, top = max_left, max_top
    tile_count = 0
    horizontal_last = False
    vertical_last = False

    while True:
        tile_path = out_folder[f'{tile_count}.tiff']
        tile_count += 1

        if tile_count % 25 == 0:
            logger.info(f'Tile count: {tile_count}')

        if os.path.exists(tile_path):
            continue

        tile_region = [{'type': 'Polygon', 'coordinates': [[(left, top, 0.0),
                                                            (left + tile_width, top, 0.0),
                                                            (left + tile_width, top - tile_height, 0.0),
                                                            (left, top - tile_height, 0.0)]]}]
        out_image, out_transform = rasterio.mask.mask(source_file, tile_region, crop=True)
        out_meta = source_file.meta
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform
        })

        tile = rasterio.open(tile_path, 'w', **out_meta)
        tile.write(out_image)
        tile.close()

        left += tile_width / 2

        if horizontal_last or left >= max_right:
            left = max_left
            top -= tile_height / tile_stride_factor
            horizontal_last = False
        elif left + tile_width >= max_right:
            left = max_right - tile_width
            horizontal_last = True

        if (vertical_last and horizontal_last) or top <= max_bottom:
            break
        elif top - tile_height <= max_bottom:
            top = max_bottom + tile_height
            vertical_last = True

    source_file.close()
    print(f"Tiles created - {tile_count - 1}")


def process_tile(tiles_folder, tile_path, confidence, mask_pixel_threshold, model):
    uint8_type = True
    tile = rasterio.open(os.path.join(tiles_folder, tile_path))

    if tile.dtypes[0] != 'uint8':
        uint8_type = False

    red = tile.read(3)
    green = tile.read(2)
    blue = tile.read(1)
    array = np.dstack((red, green, blue))
    array = np.nan_to_num(array)

    if np.max(array) == 0:
        return []

    if not uint8_type:
        array = array.astype(np.float32, order='C') / 32768.0
        array = (array * 255 / np.max(array)).astype('uint8')

    tensor = transforms.ToTensor()(array)

    with torch.no_grad():
        prediction = model.forward([tensor])

    masks = []

    for idx in range(len(prediction[0]['scores'])):
        score = prediction[0]['scores'][idx]

        if score > confidence:
            mask = prediction[0]['masks'][idx, 0].mul(255).byte().cpu().numpy()
            mask = mask < mask_pixel_threshold
            mask = mask.astype('uint8') * 255
            masks.append(mask)

    tile.close()
    return masks


def predict_masks(image_path, confidence, working_folder, mask_pixel_threshold, num_classes, tile_width, tile_height):
    segmentation_model = InstanceSegmentationModel(
        model_checkpoint=data_folder['Model']['mAP_60.382.pt'],
        num_classes=num_classes,
        device='cuda'
    )
    masks_folder = working_folder['Masks']
    tiles_folder = working_folder['Tiles']
    crop_tif(image_path, tile_width, tile_height, out_folder=tiles_folder)

    for tile_idx, tile_path in enumerate(os.listdir(tiles_folder)):
        if tile_idx % 10 == 0:
            logger.info(f"Tile index: {tile_idx}")

        masks = process_tile(tiles_folder, tile_path, confidence, mask_pixel_threshold,
                                 model=segmentation_model.model)
        logger.info(f"{tile_path}: {len(masks)}")

        for mask_idx, mask in enumerate(masks):
            save_geojson_coordinates(mask_idx, mask, masks_folder, tiles_folder, tile_path, tile_width, tile_height)


def show_image_and_tile_shapes(image_path, tile_width, tile_height):
    image = rasterio.open(image_path)
    logger.info(f"[Tile width] X [Tile height]: [{tile_width}] X [{tile_height}]")
    logger.info(f"[Image tile width] X [Image tile height]: [{image.bounds[2] - image.bounds[0]}] X "
                f"[{image.bounds[3] - image.bounds[1]}]")
    logger.info(f"[Image pixel width] X [Image pixel height]: [{image.width}] X [{image.height}]")
    image.close()


def get_single_wkt_from_masks(masks_folder, intersection_threshold):
    shapes = []

    for geojson in os.listdir(masks_folder):
        name, ext = os.path.splitext(geojson)

        if ext == ".geojson":
            try:
                shp_file = gpd.read_file(os.path.join(masks_folder, geojson))

                for i in range(len(shp_file.geometry)):
                    polygon = shape(shp_file.geometry[i])
                    shapes.append(polygon)
            except ValueError:
                continue

    multipolygon = shapely.geometry.MultiPolygon(shapes)
    sorted_polygons = sorted(multipolygon, key=lambda polygon: polygon.area)
    shapes = []

    for i in range(len(sorted_polygons)):
        append = True

        for j in range(i + 1, len(sorted_polygons)):
            try:
                area = sorted_polygons[i].intersection(sorted_polygons[j]).area
            except shapely.errors.TopologicalError:
                append = False
                break

            inter1 = area / sorted_polygons[i].area
            inter2 = area / sorted_polygons[j].area

            if inter1 > intersection_threshold or inter2 > intersection_threshold:
                append = False
                break

        if append:
            shapes.append(sorted_polygons[i])

    wkt = shapely.geometry.MultiPolygon(shapes).wkt
    return wkt


def predict_regions(tif_file_name, tile_width=20000, tile_height=20000, confidence=0.7):
    logger.info(f"Image path: {tif_file_name}")
    temp_crs_converted_file_name = 'tif_file_with_epsg_3857.tiff'
    tif_file_folder = Folder(tif_file_name)
    working_folder = Folder(cache_folder.get_filepath(tif_file_folder.name))
    masks_folder = working_folder['Masks']
    out_filepath = working_folder[temp_crs_converted_file_name]
    show_image_and_tile_shapes(out_filepath, tile_width, tile_height)
    convert_crs(tif_file_name, out_filepath)
    predict_masks(
        image_path=out_filepath,
        confidence=confidence,
        mask_pixel_threshold=80,
        num_classes=2,
        tile_width=tile_width,
        tile_height=tile_height,
        working_folder=working_folder
    )
    multipolygon_wkt = get_single_wkt_from_masks(
        masks_folder=masks_folder,
        intersection_threshold=0.8
    )
    working_folder.clear()
    logger.info(f"Multipolygon WKT: {multipolygon_wkt}")
    return multipolygon_wkt


if __name__ == "__main__":
    predict_regions(
        tif_file_name="data/Tombov/tombov_rgbnir.tiff",
        tile_width=20000,
        tile_height=20000,
        confidence=0.7
    )
