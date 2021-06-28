import pandas as pd
from lgblkb_tools import logger

from config import data_folder
from inference import predict_safe_regions, save_wkt


def download_safe_data(file_name, folder_name):
    products = pd.read_csv(data_folder['SAFE_Names'][file_name]).drop("Unnamed: 0", axis=1)['title']

    for row in products:
        make_single_prediction(row, folder_name)


@logger.trace()
def make_single_prediction(row, folder_name):
    file_name = f'/storage/caches/imagiflow/safe_products/{row}.SAFE'
    wkt = predict_safe_regions(
        safe_folder_path=file_name,
        tile_width=5000,
        tile_height=5000,
        confidence=0.6,
        intersection_threshold=0.5
    )
    save_wkt(
        wkt=wkt,
        filepath=f'/home/faranio/Desktop/Egistic/Crop_Field_Segmentation/data/{folder_name}/{row}.gpkg'
    )


if __name__ == "__main__":
    file_names = [("toshkent_products.csv", "Tashkent")]#, ('russia_products.csv', 'Russia')]

    for file_name in file_names:
        download_safe_data(*file_name)
