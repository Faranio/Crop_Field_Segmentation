import json
import os
from pathlib import Path
from shutil import copyfile

import geopandas as gpd
import numpy as np
import rasterio
import rasterio.mask
import shapely
import shapely.errors
import shapely.geometry as shg
import shapely.wkt as shwkt
import torch
import torchvision
from PIL import Image
from fiona.crs import from_epsg
from lgblkb_tools import Folder, logger
from lgblkb_tools.gdal_datasets import GdalMan
from lgblkb_tools.pathify import get_name
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
from shapely.geometry import shape
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from app import data_folder, cache_folder

# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# logger.debug("device: %s", device)
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


model_path = data_folder['model']['Max_Model.pt']


@logger.trace()
def convert_wkt_to_epsg(roi_wkt):
    dst_crs = "EPSG:3857"
    src = shapely.wkt.loads(roi_wkt)
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

    dst = rasterio.open("Temp.tif", 'w', **kwargs)

    for i in range(1, src.count + 1):
        reproject(source=rasterio.band(src, i),
                  destination=rasterio.band(dst, i),
                  src_transform=src.transform,
                  src_crs=src.crs,
                  dst_transform=transform,
                  dst_crs=dst_crs,
                  resampling=Resampling.nearest)

    result = dst.wkt
    os.remove("Temp.tif")
    dst.close()
    return result


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

    src.close()
    dst.close()


@logger.trace()
def crop_tif(tif_file, width=20000, height=20000, out_folder='Temp', limit=-1):
    logger.debug("tif_file: %s", tif_file)
    out_folder = Folder(out_folder)
    src = rasterio.open(tif_file)
    max_left, max_up = src.transform * (0, 0)
    max_right, max_bottom = src.transform * (src.width, src.height)
    left, up = max_left, max_up
    tile_count = 0
    h_last = False
    v_last = False
    first_pass = True

    while True:
        tile_path = out_folder[f"{tile_count}.tiff"]
        tile_count += 1
        if tile_count % 25 == 0:
            logger.debug("tile_count: %s", tile_count)
        if tile_count == limit:
            break
        if os.path.exists(tile_path):
            continue

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
        dest.close()

        if first_pass:
            first_pass = False
            logger.info(f"Tile shape: {out_image.shape}")

        left += width / 2

        if h_last or left >= max_right:
            left = max_left
            up -= height / 2
            h_last = False
        elif left + width >= max_right:
            left = max_right - width
            h_last = True

        if v_last or up <= max_bottom:
            break
        elif up - height <= max_bottom:
            up = max_bottom + height
            v_last = True

    src.close()


def process_tile(working_dir, tile_path, confidence, threshold, model):
    uint8_type = True
    tile = rasterio.open(os.path.join(working_dir, tile_path))

    if tile.dtypes[0] != 'uint8':
        uint8_type = False

    array = np.dstack((tile.read(4), tile.read(3), tile.read(2)))
    array = np.nan_to_num(array)

    if np.max(array) == 0:
        return []

    if not uint8_type:
        array = array.astype(np.float32, order='C') / 32768.0
        array = (array * 255 / np.max(array)).astype('uint8')

    tensor = transforms.ToTensor()(array)

    with torch.no_grad():
        prediction = model.forward([tensor])

    temp = []

    for i in range(len(prediction[0]['scores'])):
        if prediction[0]['scores'][i] > confidence:
            mask = prediction[0]['masks'][i, 0].mul(255).byte().cpu().numpy()
            mask = mask < threshold
            mask = mask.astype('uint8') * 255
            temp.append(mask)

    tile.close()
    return temp


@logger.trace()
def tile_pipeline(image_path, confidence, threshold, model, working_folder, masks_folder, tile_width=20000,
                  tile_height=20000, tilings_folder='Temp'):
    crop_tif(image_path, tile_width, tile_height, out_folder=tilings_folder)

    for tile_i, tile_path in enumerate(os.listdir(tilings_folder)):
        if tile_i % 10 == 0:
            logger.debug("tile_i: %s", tile_i)
        temp = process_tile(tilings_folder, tile_path, confidence, threshold, model=model)

        logger.debug("%s: %s", tile_path, len(temp))

        for mask_i, mask in enumerate(temp):
            raster_filepath = Path(masks_folder[f'{get_name(tile_path)}_{mask_i}.bmp'])
            image = Image.fromarray(mask)
            image.save(raster_filepath)

            output_path = Path(raster_filepath).with_suffix('.geojson')
            cmd = f"potrace -b geojson {raster_filepath} -o {output_path} && " \
                  f"cat {output_path} | simplify-geojson -t 5 > temp.geojson && " \
                  f"mv temp.geojson {output_path}"
            os.system(cmd)

            tile_path = working_folder[f"tilings/{get_name(tile_path)}.tiff"]
            src = rasterio.open(tile_path)
            image_left, image_bottom = src.bounds[:2]

            with open(output_path, 'r') as file:
                shp_file = json.load(file)

            for i in range(len(shp_file['features'])):
                for j in range(len(shp_file['features'][i]['geometry']['coordinates'])):
                    for l in range(len(shp_file['features'][i]['geometry']['coordinates'][j])):
                        x = shp_file['features'][i]['geometry']['coordinates'][j][l][0] * tile_width / \
                            src.shape[
                                0] + image_left
                        y = shp_file['features'][i]['geometry']['coordinates'][j][l][1] * tile_height / \
                            src.shape[
                                1] + image_bottom
                        shp_file['features'][i]['geometry']['coordinates'][j][l] = [x, y]

            with open(output_path, 'w+') as f:
                json.dump(shp_file, f, indent=2)

            src.close()


@logger.trace()
def image_pipeline(image_path, confidence, threshold, model):
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
        prediction = model([tensor])

    for i in range(len(prediction[0]['masks'])):
        if prediction[0]['scores'][i] > confidence:
            mask = prediction[0]['masks'][i, 0].mul(255).byte().cpu().numpy()
            mask = mask < threshold
            mask = mask.astype('uint8') * 255
            masks[image_path].append(mask)

    image.close()
    return masks


@logger.trace()
def get_mask_info(image_path, confidence, working_folder, masks_folder, threshold=100, tile_width=20000,
                  tile_height=20000, tilings_folder='Temp'):
    image = rasterio.open(image_path)

    model = get_instance_segmentation_model(2)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    logger.debug(f"Tile width: {tile_width}")
    logger.debug(f"Tile height: {tile_height}")
    logger.debug(f"Image tile width: {image.bounds[2] - image.bounds[0]}")
    logger.debug(f"Image width: {image.width}")
    logger.debug(f"Image tile height: {image.bounds[3] - image.bounds[1]}")
    logger.debug(f"Image height: {image.height}")
    image.close()

    tile_pipeline(image_path=image_path, confidence=confidence, threshold=threshold, model=model, tile_width=tile_width,
                  tile_height=tile_height, working_folder=working_folder, masks_folder=masks_folder,
                  tilings_folder=tilings_folder)


# if image.bounds[2] - image.bounds[0] > tile_width or image.bounds[3] - image.bounds[1] > tile_height:
# 	tile_pipeline(image_path, confidence, threshold, model, tile_width, tile_height, working_folder=working_folder,
# 	              masks_folder=masks_folder, working_dir=working_dir)
# else:
# 	masks = image_pipeline(image_path, confidence, threshold, model)

# return masks


@logger.trace()
def get_wkt(folder, threshold=0.8):
    shapes = []

    for geojson in os.listdir(folder):
        name, ext = os.path.splitext(geojson)

        if ext == ".geojson":
            try:
                shp_file = gpd.read_file(os.path.join(folder, geojson))
                for i in range(len(shp_file.geometry)):
                    polygon = shape(shp_file.geometry[i])
                    shapes.append(polygon)
            except ValueError:
                continue

    multipolygon = shapely.geometry.MultiPolygon(shapes)
    logger.debug(f"shapes: {shapes}")
    return multipolygon.wkt


@logger.trace()
def crop_wkt(roi_wkt, folder):
    out_tif = folder['combined_bands.tiff']
    crs = 3857

    #######################################################TEMP#########################################################
    with rasterio.open(out_tif) as data:
        data_mask = data.dataset_mask()

        for geom, val in rasterio.features.shapes(
                data_mask, transform=data.transform):
            geom = rasterio.warp.transform_geom(
                data.crs, crs, geom, precision=6)

            poly = shapely.geometry.Polygon(geom['coordinates'][0])

        g1 = shapely.wkt.loads(roi_wkt)
        g2 = shapely.wkt.loads(poly.wkt)
        geo1 = gpd.GeoDataFrame({'geometry': g1}, index=[0], crs=from_epsg(4326)).to_crs(crs)
        geo2 = gpd.GeoDataFrame({'geometry': g2}, index=[0], crs=crs).to_crs(crs)
        g1 = geo1.iloc[0]['geometry']
        g2 = geo2.iloc[0]['geometry']

        logger.info(f"CRS: {crs}")
        logger.info(f"Intersection area: {g2.intersection(g1).area}")
        # logger.info(f"geo1: {geo1.iloc[0]['geometry']}")
        # logger.info(f"geo2: {geo2.iloc[0]['geometry']}")

        #######################################################TEMP#####################################################

        if g2.intersection(g1).area <= 0:
            return False

        coords = [json.loads(geo1.to_json())['features'][0]['geometry']]
        out_img, out_transform = mask(data, shapes=coords, crop=True)
        out_meta = data.meta.copy()
        out_meta.update({"driver": "GTiff",
                         "height": out_img.shape[1],
                         "width": out_img.shape[2],
                         "transform": out_transform,
                         "crs": 'EPSG:3857'})

        with rasterio.open(out_tif, "w", **out_meta) as dest:
            dest.write(out_img)

    return True


@logger.trace()
def segment_safe_product(safe_folder_path, roi_wkt, tile_width_height=(20000, 20000), confidence=0.5):
    safe_folder = Folder(safe_folder_path)
    # band_paths = [safe_folder.glob_search(f'**/*_B0{band_num}_10m.jp2')[0] for band_num in [2, 3, 4, 8]]
    working_folder = Folder(cache_folder.get_filepath(safe_folder.name))
    # out_filepath = GdalMan(q=True, lazy=True). \
    # 	gdal_merge(*band_paths, separate=True,
    #                out_filepath=working_folder['combined_bands.tiff']).out_filepath
    copyfile(safe_folder_path, working_folder['combined_bands.tiff'])
    out_filepath = working_folder['combined_bands.tiff']

    convert_to_epsg(out_filepath, out_filepath)
    flag = crop_wkt(roi_wkt, folder=working_folder)

    if not flag:
        working_folder.clear()
        return ''

    logger.debug("combined_bands: %s", out_filepath)
    tile_width, tile_height = tile_width_height
    tilings_folder = working_folder['tilings'].clear()
    get_mask_info(image_path=out_filepath, confidence=confidence, tile_width=tile_width, tile_height=tile_height,
                  tilings_folder=tilings_folder, working_folder=working_folder, masks_folder=working_folder['Masks'])

    multipolygon_wkt = get_wkt(folder=working_folder['Masks'])
    working_folder.clear()

    logger.debug(f"multipolygon_wkt: {multipolygon_wkt}")

    return multipolygon_wkt


def remove_overlaps(wkt=None, overlap_threshold=0.7, input_file=None, output_file=None):
    if input_file is not None:
        with open(input_file, 'r') as file:
            wkt = file.read()

    multipolygon = gpd.GeoDataFrame(dict(geometry=[shwkt.loads(wkt)]), crs='epsg:3857')

    temp = shape(multipolygon.iloc[0].geometry)
    temp = sorted(temp, key=lambda polygon: polygon.area)
    shapes = []

    for i in range(len(temp)):
        print("Removing overlaps: {}/{}".format(i + 1, len(temp)))
        append = True

        for j in range(i + 1, len(temp)):
            try:
                area = temp[i].intersection(temp[j]).area
            except shapely.errors.TopologicalError:
                append = False
                break

            inter1 = area / temp[i].area
            inter2 = area / temp[j].area

            if inter1 > overlap_threshold or inter2 > overlap_threshold:
                append = False
                break

        if append:
            shapes.append(temp[i])

    out = shg.MultiPolygon(shapes)

    if output_file is not None:
        with open(output_file, 'w') as file:
            file.write(out.wkt)

    return out.wkt


@logger.trace()
def predict(image_path, tile_width_height=(20000, 20000), confidence=0.5):
    logger.info(f"Image path: {image_path}")
    safe_folder = Folder(image_path)
    working_folder = Folder(cache_folder.get_filepath(safe_folder.name))
    convert_to_epsg(image_path, working_folder['combined_bands.tiff'])
    out_filepath = working_folder['combined_bands.tiff']
    tile_width, tile_height = tile_width_height
    tilings_folder = working_folder['tilings'].clear()
    get_mask_info(image_path=out_filepath, confidence=confidence, tile_width=tile_width, tile_height=tile_height,
                  tilings_folder=tilings_folder, working_folder=working_folder, masks_folder=working_folder['Masks'])

    multipolygon_wkt = get_wkt(folder=working_folder['Masks'])
    multipolygon_wkt = remove_overlaps(multipolygon_wkt)
    working_folder.clear()
    logger.info(f"Multipolygon WKT: {multipolygon_wkt}")

    return multipolygon_wkt


def save_wkt(wkt: str, filepath, crs='epsg:3857', driver='GeoJSON'):
    gpd.GeoSeries(shapely.wkt.loads(wkt), crs=crs).to_file(filepath, driver)


def main():
    wkt = predict("../data/Uzbekistan/images/1.tif", tile_width_height=(2000, 2000), confidence=0.70)
    save_wkt(wkt, "result_60percent.gpkg")

    # wkts = []
    #
    # for file in os.listdir(data_folder['Uzbekistan']['results'].path):
    #     with open(os.path.join(data_folder['Uzbekistan']['results'].path, file), 'r') as wkt_file:
    #         wkts.append(wkt_file.read())
    #
    # multipolygons = gpd.GeoDataFrame(dict(geometry=[shwkt.loads(x) for x in wkts]), crs='epsg:3857')
    # multipolygons = multipolygons[multipolygons.area != 0]
    # logger.debug("multipolygons:\n%s", multipolygons)
    #
    # shapes = []
    #
    # for multipolygon in multipolygons.geometry:
    #     multipolygon = shape(multipolygon)
    #     remove_overlaps(multipolygon.wkt, overlap_threshold=0.3)
    #
    # out = shg.MultiPolygon(shapes)
    #
    # with open(os.path.join(data_folder['Uzbekistan']['results'].path, "overlap30percent.txt"), 'w') as file:
    #     file.write(out.wkt)
    #
    # return

    # input_file = os.path.join(data_folder['Uzbekistan']['results'].path, 'confidence70percent.txt')
    # output_file = os.path.join(data_folder['Uzbekistan']['results'].path, 'conf70over30.txt')
    # wkt = remove_overlaps(input_file=input_file, output_file=output_file, overlap_threshold=0.3)
    # print(wkt)
    #
    # return

    # uzbekistan_wkt = "MULTIPOLYGON (((66.4322809392397 40.5908540617513,66.4448999392514 40.6027220617624," \
    #                  "66.4485159392549 40.6566540618126,66.3366079391506 40.6621360618177,66.3087459391246 " \
    #                  "40.6686740618238,66.3195719391347 40.6525000618088,66.3175579391328 40.6387020617959," \
    #                  "66.3079829391239 40.6125180617715,66.2806089390985 40.5909530617514,66.2799449390978 " \
    #                  "40.5624500617249,66.2547759390744 40.5624460617249,66.2529439390726 40.5559950617189," \
    #                  "66.2667609390856 40.5484120617118,66.2660059390848 40.5402220617042,66.2379909390588 " \
    #                  "40.5130380616789,66.2561029390756 40.4980460616648,66.2405919390612 40.4792970616474," \
    #                  "66.2387229390594 40.4456210616161,66.2654409390843 40.441287061612,66.2644879390834 " \
    #                  "40.424030061596,66.2282709390497 40.4276580615993,66.213606939036 40.4205470615927," \
    #                  "66.2085569390313 40.3938400615679,66.2453679390656 40.3932380615673,66.2299879390513 " \
    #                  "40.3537400615305,66.254424939074 40.3490180615261,66.2459029390661 40.3167570614961," \
    #                  "66.2248449390465 40.2972250614778,66.1729499389982 40.2673950614501,66.1924889390164 " \
    #                  "40.243743061428,66.1864999390108 40.2235330614092,66.1462859389733 40.1904750613784," \
    #                  "66.1232979389519 40.154537061345,66.1091529389387 40.086921061282,66.1463469389734 " \
    #                  "40.0850440612803,66.1655569389913 40.1007270612948,66.175940939001 40.0923570612871," \
    #                  "66.1590109389852 40.0800890612757,66.1584469389847 40.0524440612499,66.1415779389689 " \
    #                  "40.0419120612401,66.1411429389685 40.0207480612204,66.1515269389783 40.012802061213," \
    #                  "66.1418449389692 40.0008540612018,66.1249919389535 39.990318061192,66.106162938936 " \
    #                  "39.9919050611935,66.0722269389043 39.9872390611892,66.0563499388896 39.9688940611721," \
    #                  "66.028998938864 39.960220061164,66.007437938844 39.9639960611675,65.973067938812 " \
    #                  "39.9644880611679,65.948409938789 39.9501450611547,65.9217139387641 39.9448960611497," \
    #                  "65.8796459387249 39.9497870611542,65.8468319386944 39.9441680611491,65.8130489386629 " \
    #                  "39.9463340611511,65.7855979386374 39.9644050611679,65.7662649386194 39.9702600611734," \
    #                  "65.7302239385859 39.9711340611742,65.7047799385622 39.9757650611785,65.6661829385262 " \
    #                  "39.9943690611958,65.6528929385138 39.9962530611975,65.6413039385031 40.0015670612025," \
    #                  "65.6421579385038 40.023601061223,65.609084938473 40.042980061241,65.5836479384493 " \
    #                  "40.0523410612498,65.5804359384463 40.0618850612587,65.5710669384376 40.0680350612645," \
    #                  "65.5455319384138 40.0700300612663,65.5333929384025 40.0766440612724,65.5241389383939 " \
    #                  "40.0935970612882,65.5063929383774 40.0972320612916,65.4873499383597 40.0844610612796," \
    #                  "65.4422529383176 40.0200720612198,65.4341269383101 39.9855760611876,65.4729759383463 " \
    #                  "39.986068061188,65.4862439383587 39.9807510611831,65.4789349383518 39.9726060611755," \
    #                  "65.47702693835 39.9488560611534,65.4475169383226 39.9374800611428,65.4478829383228 " \
    #                  "39.9193190611259,65.4749289383481 39.9078210611152,65.4562069383306 39.8656460610759," \
    #                  "65.4637829383377 39.8482850610598,65.4504769383253 39.8466870610583,65.4173879382945 " \
    #                  "39.8608320610715,65.3708869382512 39.8629830610735,65.3455199382276 39.8770440610865," \
    #                  "65.325568938209 39.8772200610867,65.3090659381936 39.8911810610997,65.2635189381512 " \
    #                  "39.8768610610863,65.2833629381697 39.8637420610742,65.2721399381592 39.8448180610565," \
    #                  "65.2610159381489 39.8379820610501,65.2355419381251 39.8364600610488,65.2045439380963 " \
    #                  "39.8384200610505,65.1823339380756 39.8299440610427,65.1701659380642 39.8326220610452," \
    #                  "65.1424629380384 39.8284980610413,65.1754299380692 39.7876350610032,65.2040549380958 " \
    #                  "39.74122206096,65.2476569381364 39.6703870608941,65.2341529381238 39.635913060862," \
    #                  "65.214972938106 39.5712280608018,65.2170099381079 39.5452690607775,65.233328938123 " \
    #                  "39.5122940607468,65.2364729381259 39.4889290607251,65.2468099381356 39.4712820607086," \
    #                  "65.2517309381402 39.4628790607008,65.2680809381554 39.4394070606789,65.2754049381623 " \
    #                  "39.4340130606739,65.2909919381768 39.4346080606744,65.3193579382032 39.4293360606696," \
    #                  "65.3614949382424 39.4275390606679,65.4036479382817 39.4271620606676,65.4521859383269 " \
    #                  "39.4231140606637,65.488806938361 39.4206000606614,65.5116879383824 39.4189370606599," \
    #                  "65.5291669383986 39.424510060665,65.5419919384106 39.4251020606656,65.5594859384268 " \
    #                  "39.4321130606722,65.5742719384406 39.4427560606821,65.5871499384526 39.4469370606859," \
    #                  "65.6100459384739 39.4459640606851,65.6300499384926 39.4327880606728,65.6465449385079 " \
    #                  "39.4326130606726,65.6584089385189 39.4288740606691,65.6813119385403 39.4286190606689," \
    #                  "65.6969519385549 39.4334710606734,65.7089459385661 39.4390900606786,65.7330619385884 " \
    #                  "39.4596710606978,65.7488549386032 39.4745860607117,65.7490689386033 39.4904130607264," \
    #                  "65.7455059386001 39.4976570607332,65.7364499385916 39.5063970607413,65.7365789385918 " \
    #                  "39.5157500607501,65.7512579386054 39.5155670607499,65.7539439386079 39.5105050607452," \
    #                  "65.7657159386189 39.499561060735,65.7776409386301 39.5001330607355,65.7823019386343 " \
    #                  "39.5051190607401,65.7887869386404 39.5100700607447,65.7924109386438 39.5064270607414," \
    #                  "65.796904938648 39.500617060736,65.8013299386521 39.4897680607258,65.8057929386562 " \
    #                  "39.4818000607185,65.8148419386647 39.4730520607103,65.8201589386696 39.4607500606989," \
    #                  "65.8337779386823 39.4526630606913,65.8509059386982 39.4337420606737,65.8672559387135 " \
    #                  "39.4248920606654,65.8780509387235 39.4125130606539,65.8952709387395 39.400775060643," \
    #                  "65.904334938748 39.3949050606375,65.9235679387659 39.3946380606372,65.9238509387662 " \
    #                  "39.3557850606011,65.9356609387772 39.3505890605962,65.948355938789 39.34393606059," \
    #                  "65.9592199387991 39.3373140605839,65.9597999387997 39.3171650605651,65.9661019388055 " \
    #                  "39.3113210605596,65.9758909388147 39.2953600605448,65.9895169388273 39.2901300605399," \
    #                  "66.0048819388416 39.2798380605303,66.0194389388552 39.2753100605261,66.0404429388748 " \
    #                  "39.2742840605252,66.0623929388951 39.2746880605256,66.0770329389088 39.2759050605267," \
    #                  "66.0882029389193 39.2865330605365,66.102988938933 39.2949440605445,66.1130979389424 " \
    #                  "39.2976640605469,66.122260938951 39.2982480605475,66.1236189389522 39.3212540605689," \
    #                  "66.1248009389533 39.3356200605823,66.1442789389714 39.3489870605947,66.1498099389766 " \
    #                  "39.3510620605967,66.1735909389987 39.3506850605963,66.1762309390012 39.3448980605909," \
    #                  "66.1890789390132 39.3475720605935,66.2075189390304 39.3537520605992,66.2186879390407 " \
    #                  "39.3629260606077,66.2325279390537 39.3684570606128,66.2381129390588 39.3726800606168," \
    #                  "66.2546079390742 39.3738470606179,66.2600319390793 39.3701660606144,66.2700499390886 " \
    #                  "39.3678390606123,66.2746119390928 39.3677670606122,66.2763289390945 39.3619840606069," \
    #                  "66.2761909390943 39.3555030606008,66.2806239390985 39.3489600605947,66.2823939391001 " \
    #                  "39.346054060592,66.288763939106 39.3445090605905,66.2951269391119 39.3429640605891," \
    #                  "66.3125299391282 39.3433910605895,66.3152309391307 39.3419070605881,66.317748939133 " \
    #                  "39.3310660605781,66.3194349391346 39.324573060572,66.3247599391395 39.3172790605652," \
    #                  "66.3332969391475 39.3308020605778,66.3407129391544 39.3357120605823,66.3418569391555 " \
    #                  "39.3457640605918,66.3647839391768 39.3482470605941,66.3658519391778 39.3547050606," \
    #                  "66.371382939183 39.3567610606019,66.3730999391846 39.3516960605973,66.3822779391931 " \
    #                  "39.3529770605985,66.3997419392093 39.3562690606015,66.4070809392162 39.3568570606021," \
    #                  "66.4180519392264 39.3566580606019,66.4181739392265 39.3616980606066,66.4273369392351 " \
    #                  "39.3622470606072,66.427229939235 39.3579360606031,66.4363859392435 39.357776060603," \
    #                  "66.4404979392474 39.3382790605848,66.4688109392737 39.3363300605829,66.4823149392863 " \
    #                  "39.3274420605747,66.4894559392929 39.3201250605679,66.5030359393056 39.3148380605629," \
    #                  "66.5130919393149 39.3146510605628,66.5250469393261 39.3173060605652,66.5252069393263 " \
    #                  "39.3237720605712,66.5265419393275 39.3402930605867,66.5294259393301 39.345993060592," \
    #                  "66.5376349393378 39.3451230605911,66.5438379393435 39.3370890605837,66.5492169393486 " \
    #                  "39.3326680605795,66.5600579393586 39.3274300605747,66.5737529393714 39.3264500605737," \
    #                  "66.5877149393845 39.3355290605822,66.5942229393905 39.3397210605861,66.6153029394101 " \
    #                  "39.3414720605877,66.6275709394216 39.3556130606009,66.6407619394339 39.369739060614," \
    #                  "66.6467129394393 39.3868860606301,66.6562189394483 39.4003600606426,66.681746939472 " \
    #                  "39.3962590606388,66.7038259394926 39.4001230606423,66.7213129395088 39.4033660606454," \
    #                  "66.7414009395275 39.4015120606437,66.757971939543 39.4047690606467,66.7663339395508 " \
    #                  "39.4089080606506,66.7699429395542 39.4066770606484,66.7664259395509 39.4011110606433," \
    #                  "66.7807149395642 39.4018240606439,66.7856669395687 39.3935690606363,66.7853619395685 " \
    #                  "39.3828730606263,66.8177709395987 39.3827050606262,66.8426739396219 39.3699450606143," \
    #                  "66.8448409396239 39.3775360606214,66.8598629396379 39.3812860606248,66.8740379396511 " \
    #                  "39.3811220606247,66.8716879396489 39.3721160606163,66.8904569396664 39.3692390606136," \
    #                  "66.8788749396556 39.3501350605958,66.895422939671 39.3431890605893,66.8840629396604 " \
    #                  "39.3310810605781,66.8835059396599 39.3125720605609,66.8972469396727 39.2822150605326," \
    #                  "66.8727179396499 39.249397060502,66.8764949396534 39.2357330604893,66.891158939667 " \
    #                  "39.2362400604897,66.9005429396758 39.2529140605052,66.9242549396979 39.2416830604948," \
    #                  "66.9290459397023 39.2444570604974,66.9454419397176 39.2502740605028,66.9665449397372 " \
    #                  "39.2395010604928,66.9754329397455 39.2561870605083,66.9980999397666 39.2453800604982," \
    #                  "66.9971229397657 39.2309910604848,67.0227199397896 39.2304070604843,67.0373989398032 " \
    #                  "39.2317160604855,67.045744939811 39.2475770605003,67.067206939831 39.264366060516," \
    #                  "67.0676799398315 39.2948220605443,67.057570939822 39.3057630605545,67.0657039398296 " \
    #                  "39.3146240605627,67.0840139398467 39.3141970605623,67.0978539398595 39.321281060569," \
    #                  "67.1028129398641 39.3129340605612,67.1330939398924 39.310562060559,67.1785959399348 " \
    #                  "39.3098750605583,67.1839209399396 39.2973930605467,67.1802519399363 39.2818450605321," \
    #                  "67.2037269399582 39.2800330605305,67.2091059399632 39.2844310605346,67.2101589399641 " \
    #                  "39.2996330605487,67.2308189399834 39.3069380605556,67.2517469400029 39.2771980605279," \
    #                  "67.2773969400267 39.2777860605285,67.2874749400361 39.281646060532,67.3094399400566 " \
    #                  "39.2815010605318,67.3258509400719 39.2724380605234,67.3498759400942 39.2858160605359," \
    #                  "67.3654089401087 39.2673030605187,67.3933019401348 39.2708540605219,67.3871299401289 " \
    #                  "39.2874250605374,67.387259940129 39.2950930605445,67.3922799401338 39.2998350605489," \
    #                  "67.4158399401557 39.3053350605541,67.4307399401696 39.3099670605584,67.4642559402008 " \
    #                  "39.319183060567,67.4853129402204 39.3227800605703,67.4939189402284 39.3207620605684," \
    #                  "67.5087049402422 39.3186750605665,67.5197979402526 39.3175880605655,67.5236509402561 " \
    #                  "39.3252140605726,67.5189659402518 39.3386910605852,67.5089639402424 39.3548080606001," \
    #                  "67.4985959402328 39.3715240606158,67.4807809402161 39.4100720606517,67.437598940176 " \
    #                  "39.482452060719,67.4254829401647 39.4969590607326,67.4257119401649 39.5097650607444," \
    #                  "67.425879940165 39.5190040607531,67.4335549401722 39.5323370607655,67.4434959401814 " \
    #                  "39.5341450607672,67.4583049401952 39.5311080607644,67.4854349402204 39.5250510607587," \
    #                  "67.4916219402263 39.5249820607586,67.5004569402345 39.5335080607666,67.5001369402342 " \
    #                  "39.536087060769,67.4801779402156 39.5502390607821,67.4848089402199 39.5648230607958," \
    #                  "67.489211940224 39.5880310608173,67.4679479402042 39.596214060825,67.4631189401997 " \
    #                  "39.5828280608125,67.4481269401858 39.5857620608152,67.4444349401824 39.6019170608303," \
    #                  "67.4067379401473 39.603767060832,67.4005579401415 39.6115370608393,67.4041739401448 " \
    #                  "39.6215890608487,67.3979559401391 39.6285130608551,67.4104079401506 39.643398060869," \
    #                  "67.4076839401481 39.6569970608816,67.3943019401356 39.6742550608977,67.4149999401549 " \
    #                  "39.6787790609019,67.4196389401593 39.6871100609097,67.4427559401808 39.6983220609201," \
    #                  "67.4554289401926 39.7182690609387,67.468810940205 39.7288970609485,67.473074940209 " \
    #                  "39.7549810609729,67.4874259402224 39.7621950609796,67.5177449402506 39.7630610609803," \
    #                  "67.5356669402672 39.7515750609697,67.5422739402734 39.7547720609726,67.5424489402736 " \
    #                  "39.7589980609766,67.5532529402837 39.7587010609763,67.5449519402759 39.7673790609844," \
    #                  "67.5452649402762 39.7749820609915,67.5398629402712 39.7751310609916,67.532118940264 " \
    #                  "39.7711140609879,67.5237499402562 39.7781100609944,67.5035929402374 39.7879600610036," \
    #                  "67.4814979402168 39.8037790610183,67.4744939402103 39.8183320610319,67.4578619401948 " \
    #                  "39.8356890610481,67.4428779401809 39.8403160610523,67.4199669401596 39.8637500610742," \
    #                  "67.4331889401719 39.8693120610793,67.42056194016 39.8789520610883,67.3820029401242 " \
    #                  "39.8622240610727,67.3693079401124 39.8701660610801,67.324866940071 39.8696280610796," \
    #                  "67.2919919400404 39.8594930610702,67.2726359400223 39.8633680610738,67.247534939999 " \
    #                  "39.8580810610688,67.2389369399909 39.8599890610706,67.2220069399752 39.8714020610813," \
    #                  "67.2162169399698 39.8909830610995,67.1933889399485 39.8890220610977,67.1905889399459 " \
    #                  "39.8713370610812,67.1539909399118 39.8781540610876,67.1258159398856 39.8779940610874," \
    #                  "67.0959699398578 39.8922300611007,67.0878439398502 39.9084890611158,67.082228939845 " \
    #                  "39.9348290611404,67.0724019398358 39.9325290611382,67.0538019398185 39.9278860611339," \
    #                  "67.0475079398126 39.9339590611395,67.0466989398119 39.9424320611474,67.035056939801 " \
    #                  "39.9511560611555,67.0529399398177 39.9667960611701,67.0822979398451 39.9686390611718," \
    #                  "67.0635829398277 39.9927550611943,67.0591269398235 40.0216020612212,67.048964939814 " \
    #                  "40.0421330612403,67.0284719397949 40.0468360612447,67.0105589397782 40.06331206126," \
    #                  "66.9955589397643 40.0695830612659,66.9729689397432 40.0768580612727,66.9336919397066 " \
    #                  "40.071838061268,66.935156939708 40.0836330612789,66.9648049397357 40.0931090612877," \
    #                  "66.9783319397483 40.1080130613016,66.986639939756 40.1289630613211,66.9873879397567 " \
    #                  "40.1517630613423,66.9766609397466 40.1562380613466,66.9651709397359 40.1708750613602," \
    #                  "66.9511409397229 40.1745790613636,66.9385979397112 40.1909210613789,66.9125209396869 " \
    #                  "40.1923560613802,66.8860999396623 40.1827960613712,66.8638069396416 40.2010450613883," \
    #                  "66.8896019396657 40.2249900614106,66.8791419396559 40.2387460614234,66.8817819396583 " \
    #                  "40.2530590614367,66.8753809396523 40.2574340614408,66.8735189396506 40.2676160614503," \
    #                  "66.8967049396722 40.2772440614593,66.8995429396749 40.3321300615104,66.8914409396673 " \
    #                  "40.3517600615287,66.9080499396828 40.3589930615354,66.9138179396881 40.3690070615447," \
    #                  "66.9066839396815 40.384380061559,66.8867029396628 40.4084960615815,66.8928679396686 " \
    #                  "40.4303430616018,66.85129493963 40.4287330616003,66.8280559396083 40.4190900615913," \
    #                  "66.8062969395881 40.4220960615941,66.7884749395714 40.4195590615918,66.771895939556 " \
    #                  "40.4177890615901,66.752623939538 40.4181900615905,66.7445669395306 40.4247550615966," \
    #                  "66.7309949395178 40.4314300616029,66.7230519395104 40.4422560616129,66.7091519394975 " \
    #                  "40.4382850616093,66.6951369394845 40.4300460616016,66.6839439394741 40.4238850615958," \
    #                  "66.6673039394585 40.4199600615922,66.6562949394483 40.4201810615924,66.653792939446 " \
    #                  "40.4287600616004,66.6652289394567 40.443450061614,66.6740949394649 40.4645840616337," \
    #                  "66.6679149394591 40.4752690616437,66.6498559394423 40.4842600616521,66.627989939422 " \
    #                  "40.4910960616584,66.6170879394118 40.4955820616626,66.6032399393989 40.4998930616667," \
    #                  "66.6311559394249 40.5187790616842,66.5954739393917 40.5220830616872,66.5742179393719 " \
    #                  "40.5423690617061,66.5619269393604 40.5633460617258,66.5520549393512 40.5704490617323," \
    #                  "66.5050499393075 40.5700560617319,66.4951929392983 40.5789450617402,66.4808339392849 " \
    #                  "40.5856360617465,66.469168939274 40.5888780617494,66.4500119392562 40.584049061745," \
    #                  "66.4322809392397 40.5908540617513))) "
    #
    # multipolygons = []
    # confidence = 0.7
    #
    # for file in os.listdir(data_folder['Uzbekistan']['images'].path):
    #     if file.endswith(".tif"):
    #         current_file = os.path.join(data_folder['Uzbekistan']['images'].path, file)
    #         logger.debug(f"Current file: {file}")
    #         regions = segment_safe_product(safe_folder_path=current_file, roi_wkt=uzbekistan_wkt,
    #                                        tile_width_height=(1200, 1200), confidence=confidence)
    #
    #         name, ext = os.path.splitext(file)
    #
    #         with open(os.path.join(data_folder['Uzbekistan']['results'].path, "{}.txt".format(name)), 'w') as results:
    #             results.write(regions)
    #
    #         multipolygons.append(regions)
    #
    # multipolygons = gpd.GeoDataFrame(dict(geometry=[shwkt.loads(x) for x in multipolygons]), crs='epsg:3857')
    # multipolygons = multipolygons[multipolygons.area != 0]
    # logger.debug("multipolygons:\n%s", multipolygons)
    #
    # shapes = []
    #
    # for multipolygon in multipolygons.geometry:
    #     multipolygon = shape(multipolygon)
    #     for polygon in multipolygon:
    #         shapes.append(polygon)
    #
    # out = shg.MultiPolygon(shapes)
    #
    # with open(os.path.join(data_folder['Uzbekistan']['results'].path, "confidence70percent.txt"), 'w') as file:
    #     file.write(out.wkt)


if __name__ == "__main__":
    main()
