import pandas as pd
import sys

from config import data_folder
from inference import predict_safe_regions, save_wkt


def download_safe_data(file_name, folder_name):
    products = pd.read_csv(data_folder['SAFE_Names'][file_name]).drop("Unnamed: 0", axis=1)['title']
    print(f'\nSAFE Names from file {file_name}:')

    for row in products:
        safe_folder_path = f'/storage/caches/imagiflow/safe_products/{row}.SAFE'
        wkt = predict_safe_regions(
            safe_folder_path=safe_folder_path,
            tile_width=5000,
            tile_height=5000
        )
        save_wkt(
            wkt=wkt,
            filepath=f'/home/faranio/Desktop/Egistic/Crop_Field_Segmentation/data/{folder_name}/{row}.gpkg'
        )


def main(file_names):
    for file_name in file_names:
        download_safe_data(*file_name)


if __name__ == "__main__":
    file_names = [("toshkent_products.csv", "Tashkent"), ("russia_products.csv", "Russia")]
    main(file_names)
