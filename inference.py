import geopandas as gpd
import json
import numpy as np
import os
import rasterio
import rasterio.mask
import shapely
import shapely.errors
import shapely.wkt
import torch

from lgblkb_tools import Folder, logger
from lgblkb_tools.pathify import get_name
from pathlib import Path
from PIL import Image
from rasterio.warp import calculate_default_transform, reproject, Resampling
from shapely.geometry import shape
from torchvision import transforms

from config import cache_folder, data_folder, model_path
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


segmentation_model = InstanceSegmentationModel(
    model_checkpoint=model_path,
    num_classes=2,
    device='cuda'
)


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
    image.close()


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
    return geojson_filepath.name.split('/')[-1]


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

        left += tile_width / tile_stride_factor

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
    print(f"Tiles created - {tile_count}")


def process_tile(tiles_folder, tile_path, confidence, mask_pixel_threshold):
    uint8_type = True
    tile = rasterio.open(os.path.join(tiles_folder, tile_path))

    if tile.dtypes[0] != 'uint8':
        uint8_type = False

    tile = np.dstack((tile.read(3), tile.read(2), tile.read(1)))
    tile = np.nan_to_num(tile)

    if np.max(tile) == 0:
        return []

    if not uint8_type:
        tile = tile.astype(np.float32, order='C') / 32768.0
        tile = (tile * 255 / np.max(tile)).astype('uint8')

    tile = transforms.ToTensor()(tile)

    with torch.no_grad():
        prediction = segmentation_model.model.forward([tile])

    masks = []

    for idx in range(len(prediction[0]['scores'])):
        score = prediction[0]['scores'][idx].item()

        if score > confidence:
            mask = prediction[0]['masks'][idx, 0].mul(255).byte().cpu().numpy()
            mask = mask < mask_pixel_threshold
            mask = mask.astype('uint8') * 255
            masks.append([mask, score])

    return masks


def predict_masks(image_path, confidence, working_folder, mask_pixel_threshold, tile_stride_factor, tile_width,
                  tile_height):
    masks_folder = working_folder['Masks']
    tiles_folder = working_folder['Tiles']
    crop_tif(image_path, tile_width, tile_height, tile_stride_factor=tile_stride_factor, out_folder=tiles_folder)
    masks_confidence_mapping = {}

    for tile_idx, tile_path in enumerate(os.listdir(tiles_folder)):
        if tile_idx % 10 == 0:
            logger.info(f"Tile index: {tile_idx}")

        masks = process_tile(tiles_folder, tile_path, confidence, mask_pixel_threshold)
        logger.info(f"{tile_path}: {len(masks)}")

        for mask_idx, mask in enumerate(masks):
            geojson_file_name = save_geojson_coordinates(mask_idx, mask[0], masks_folder, tiles_folder, tile_path,
                                                         tile_width, tile_height)
            masks_confidence_mapping[geojson_file_name] = mask[1]

    return masks_confidence_mapping


def show_image_and_tile_shapes(image_path, tile_width, tile_height):
    image = rasterio.open(image_path)
    logger.info(f"[Tile width] X [Tile height]: [{tile_width}] X [{tile_height}]")
    logger.info(f"[Image tile width] X [Image tile height]: [{image.bounds[2] - image.bounds[0]}] X "
                f"[{image.bounds[3] - image.bounds[1]}]")
    logger.info(f"[Image pixel width] X [Image pixel height]: [{image.width}] X [{image.height}]")
    image.close()


def read_shapes_from_geojson(masks_folder, confidence_mapping):
    shapes = []

    for geojson in os.listdir(masks_folder):
        name, ext = os.path.splitext(geojson)

        if ext == ".geojson":
            try:
                shp_file = gpd.read_file(os.path.join(masks_folder, geojson))

                for i in range(len(shp_file.geometry)):
                    polygon = shape(shp_file.geometry[i])
                    shapes.append([polygon, confidence_mapping[geojson]])
            except ValueError:
                continue

    shapes = sorted(shapes, key=lambda polygon: polygon[1])
    return shapes


def remove_overlapping_shapes(sorted_polygons, threshold):
    shapes = []

    for i in range(len(sorted_polygons)):
        append = True

        for j in range(i + 1, len(sorted_polygons)):
            try:
                area = sorted_polygons[i][0].intersection(sorted_polygons[j][0]).area
            except shapely.errors.TopologicalError:
                append = False
                break

            inter1 = area / sorted_polygons[i][0].area
            inter2 = area / sorted_polygons[j][0].area

            if inter1 > (1 - threshold) or inter2 > (1 - threshold):
                append = False
                break

        if append:
            shapes.append(sorted_polygons[i][0])

    return shapes


def remove_intersections(figures):
    shapes = []

    for i in range(len(figures)):
        shape = figures[i]

        for j in range(i + 1, len(figures)):
            shape = shape.symmetric_difference(figures[j]).difference(figures[j])

        if shape.geom_type == "MultiPolygon":
            for polygon in shape:
                shapes.append(polygon)
        else:
            shapes.append(shape)

    return shapes


def get_single_wkt_from_masks(masks_folder, intersection_threshold, confidence_mapping):
    shapes = read_shapes_from_geojson(masks_folder, confidence_mapping)
    shapes = remove_overlapping_shapes(shapes, intersection_threshold)
    shapes = remove_intersections(shapes)

    logger.info(f'Number of fields found: {len(shapes)}')
    wkt = shapely.geometry.MultiPolygon(shapes).wkt
    return wkt


def predict_regions(tif_file_name, tile_width=20000, tile_height=20000, confidence=0.7, intersection_threshold=0.8,
                    mask_pixel_threshold=80, tile_stride_factor=2):
    logger.info(f"Image path: {tif_file_name}")
    temp_crs_converted_file_name = 'tif_file_with_epsg_3857.tiff'
    tif_file_folder = Folder(tif_file_name)
    working_folder = Folder(cache_folder.get_filepath(tif_file_folder.name))
    masks_folder = working_folder['Masks']
    out_filepath = working_folder[temp_crs_converted_file_name]
    convert_crs(tif_file_name, out_filepath)
    show_image_and_tile_shapes(out_filepath, tile_width, tile_height)
    mapping = predict_masks(
        image_path=out_filepath,
        confidence=confidence,
        mask_pixel_threshold=mask_pixel_threshold,
        tile_stride_factor=tile_stride_factor,
        tile_width=tile_width,
        tile_height=tile_height,
        working_folder=working_folder
    )
    multipolygon_wkt = get_single_wkt_from_masks(
        masks_folder=masks_folder,
        intersection_threshold=intersection_threshold,
        confidence_mapping=mapping
    )
    working_folder.clear()
    return multipolygon_wkt


def save_wkt(wkt: str, filepath, crs='EPSG:3857', driver='GeoJSON'):
    gpd.GeoSeries(shapely.wkt.loads(wkt), crs=crs).to_file(filepath, driver)


if __name__ == "__main__":
    file_name = "krasnodar_8bit"
    wkt = predict_regions(
        tif_file_name=data_folder['Krasnodar'][f'{file_name}.tif'],
        tile_width=20000,
        tile_height=20000,
        confidence=0.5,
        intersection_threshold=0.5,
        tile_stride_factor=2,
        mask_pixel_threshold=80
    )
    save_wkt(wkt, f'{file_name}.gpkg')
