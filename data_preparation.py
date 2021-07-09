import geopandas as gpd
import os
import random
import rasterio
import rasterio.mask
import shapely
import shapely.geometry
import shapely.strtree
import shapely.wkt

import config

from inference import convert_crs


def crop_tif(label_geometries, tif_file, out_folder, width=7500, height=7500, tile_stride_factor=2, count=0):
    """
    Create a dataset of tiles by cropping .tif files with corresponding field geometries.
    """
    ext_len = 3
    polygon_validity_threshold = 0.9
    src = rasterio.open(tif_file)

    with open(tif_file[:-ext_len] + "txt", 'r') as file:
        valid_image_polygon = shapely.wkt.loads(file.read())

    tree = shapely.strtree.STRtree(label_geometries.geometry)
    valid_fields = list()
    intersecting_field_multipolygons = tree.query(valid_image_polygon)

    for intersecting_field_multipolygon in intersecting_field_multipolygons:
        resultant_poly = intersecting_field_multipolygon.intersection(valid_image_polygon)

        if resultant_poly.area >= intersecting_field_multipolygon.area * polygon_validity_threshold:
            valid_fields.append(resultant_poly)

    tree = shapely.strtree.STRtree(valid_fields)
    valid_image_polygon: shapely.geometry.Polygon = valid_image_polygon
    max_left, max_bottom, max_right, max_up = valid_image_polygon.bounds
    left, top = max_left, max_up

    print("[INFO] Adding shapes...")
    h_last = False
    v_last = False

    while True:
        if count % 100 == 0 and count > 0:
            print(f"[INFO] Count: {count}")

        right = left + width
        bottom = top - height
        tile_poly = shapely.geometry.Polygon([[left, bottom], [left, top], [right, top], [right, bottom]])
        intersections = tree.query(tile_poly)
        temp_coords = []

        for multipolygon in intersections:
            coord = multipolygon.intersection(tile_poly)

            if coord.area >= multipolygon.area * polygon_validity_threshold:
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

            if random.random() < 0.02:
                save_folder = out_folder['Test']
            elif random.random() < 0.10:
                save_folder = out_folder['Valid']
            else:
                save_folder = out_folder['Train']

            tile_name = f"tile_{count}"
            image_path = save_folder['Photos'][tile_name + ".tif"]

            if not os.path.exists(image_path):
                dest = rasterio.open(image_path, "w", **out_meta)
                dest.write(out_image)

            temp_coords = gpd.GeoDataFrame(geometry=temp_coords)
            shp_filepath = save_folder['Coordinates'][tile_name + '.shp']

            if not os.path.exists(shp_filepath):
                temp_coords.to_file(shp_filepath, driver='ESRI Shapefile')

            count += 1

        left += width / tile_stride_factor

        if h_last or left >= max_right:
            left = max_left
            top -= height / tile_stride_factor
            h_last = False
        elif left + width >= max_right:
            left = max_right - width
            h_last = True

        if (v_last and h_last) or top <= max_bottom:
            break
        elif top - height <= max_bottom:
            top = max_bottom + height
            v_last = True

    src.close()
    print(f"Tiles created - {count - 1}")
    return count


def crop_dataset_into_tiles():
    """
    Main loop for cropping .tif files into tiles with field coordinates.
    """
    dataset_folder = config.data_folder
    labels_file = config.data_folder['labels.geojson']
    count = 0
    labels_df = gpd.read_file(labels_file)

    for file in os.listdir(dataset_folder):
        name, ext = os.path.splitext(file)

        if ext == ".tif":
            tif_file = dataset_folder[file]
            convert_crs(tif_file, tif_file)
            count = crop_tif(labels_df, tif_file, config.data_folder, count=count)
