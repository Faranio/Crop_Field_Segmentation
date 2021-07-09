# This script is used only 1 time for making inference for Russian and Uzbekistani regions.

import os
import pandas as pd
from lgblkb_tools import logger

from config import data_folder
from inference import predict_safe_regions, save_wkt


def predict(file_name, folder_name):
    products = pd.read_csv(data_folder['SAFE_Names'][file_name]).drop("Unnamed: 0", axis=1)['title']

    for row in products:
        make_single_prediction(row, folder_name)


@logger.trace()
def make_single_prediction(row, folder_name):
    gpkg_filepath = f'/home/faranio/Desktop/Egistic/Crop_Field_Segmentation/data/{folder_name}/{row}.gpkg'

    if os.path.exists(gpkg_filepath):
        return

    safe_filepath = f'/storage/caches/imagiflow/safe_products/{row}.SAFE'
    wkt = predict_safe_regions(
        safe_folder_path=safe_filepath,
        tile_width=7000,
        tile_height=7000,
        confidence=0.5,
        intersection_threshold=0.5
    )
    save_wkt(
        wkt=wkt,
        filepath=gpkg_filepath
    )


if __name__ == "__main__":
    file_names = [('russia_products.csv', 'Russia')]  # [("toshkent_products.csv", "Tashkent"),

    for file_name in file_names:
        predict(*file_name)
