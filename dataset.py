import collections
import geopandas as gpd
import lgblkb_tools
import numpy as np
import rasterio
import torch

from PIL import Image, ImageDraw


def get_target(info, idx):
    shp_file = gpd.read_file(info['coords'][idx])
    image = rasterio.open(info['img'][idx])
    left, bottom, right, top = image.bounds
    img_height, img_width = image.shape
    image.close()
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
                        y_coord = (top - bottom) / meters_in_pixel - (shp_file.geometry[i].boundary[j].coords.xy[1][k]
                                                                      - bottom) / meters_in_pixel
                        temp.append((x_coord, y_coord))
            else:
                coord_len = len(shp_file.geometry[i].boundary.coords.xy[0])

                for j in range(coord_len):
                    x_coord = (shp_file.geometry[i].boundary.coords.xy[0][j] - left) / meters_in_pixel
                    y_coord = (top - bottom) / meters_in_pixel - (shp_file.geometry[i].boundary.coords.xy[1][j] -
                                                                  bottom) / meters_in_pixel
                    temp.append((x_coord, y_coord))
        elif shp_file.geometry[i].type == "MultiPolygon":
            for j in range(len(shp_file.geometry[i])):
                single_polygon = []

                if shp_file.geometry[i][j].boundary.type == "MultiLineString":
                    for k in range(len(shp_file.geometry[i][j].boundary)):
                        coord_len = len(shp_file.geometry[i][j].boundary[k].coords.xy[0])

                        for l in range(coord_len):
                            x_coord = (shp_file.geometry[i][j].boundary[k].coords.xy[0][l] - left) / meters_in_pixel
                            y_coord = (top - bottom) / meters_in_pixel - \
                                      (shp_file.geometry[i][j].boundary[k].coords.xy[1][l] - bottom) / meters_in_pixel
                            single_polygon.append((x_coord, y_coord))
                else:
                    coord_len = len(shp_file.geometry[i][j].boundary.coords.xy[0])

                    for k in range(coord_len):
                        x_coord = (shp_file.geometry[i][j].boundary.coords.xy[0][k] - left) / meters_in_pixel
                        y_coord = (top - bottom) / meters_in_pixel - (shp_file.geometry[i][j].boundary.coords.xy[1][k]
                                                                      - bottom) / meters_in_pixel
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


class FieldsDataset(torch.utils.data.Dataset):
    def __init__(self, folder_paths, transforms=None):
        self.folder_paths = folder_paths
        self.transforms = transforms
        self.info = collections.defaultdict(list)

        for folder_path in folder_paths:
            folder = lgblkb_tools.Folder(folder_path)

            for photo_path in folder['Photos'].children:
                self.info['img'].append(photo_path)
                self.info['coords'].append(folder['Coordinates'][f'{lgblkb_tools.pathify.get_name(photo_path)}.shp'])

    def __getitem__(self, idx):
        img_path = self.info['img'][idx]
        tile = rasterio.open(img_path)
        src = np.dstack((tile.read(3), tile.read(2), tile.read(1)))
        tile.close()
        src = np.nan_to_num(src)
        src = (src * 255 / np.max(src)).astype('uint8')
        src = Image.fromarray(src)
        target = get_target(self.info, idx)

        if self.transforms is not None:
            src, target = self.transforms(src, target)

        return src, target

    def __len__(self):
        return len(self.info['img'])
