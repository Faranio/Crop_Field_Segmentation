import pandas as pd

from config import data_folder
from inference import predict_safe_regions, save_wkt


def download_safe_data(file_name, folder_name):
    products = pd.read_csv(data_folder['SAFE_Names'][file_name]).drop("Unnamed: 0", axis=1)['title']

    for row in products:
        safe_folder_path = f'/storage/caches/imagiflow/safe_products/{row}.SAFE'
        wkt = predict_safe_regions(
            safe_folder_path=safe_folder_path,
            tile_width=6000,
            tile_height=6000
        )
        save_wkt(
            wkt=wkt,
            filepath=f'/home/faranio/Desktop/Egistic/Crop_Field_Segmentation/data/{folder_name}/{row}.gpkg'
        )


if __name__ == "__main__":
    download_safe_data("toshkent_products.csv", "Tashkent")
