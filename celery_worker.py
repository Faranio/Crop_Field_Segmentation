import celery
import pandas as pd

from celery import Celery

from config import broker_url, data_folder
from inference import predict_safe_regions, save_wkt

app = Celery('celery_worker', broker=broker_url)


def make_safe_file_predictions(file_name):
    products = pd.read_csv(data_folder['SAFE_Names'][file_name]).drop("Unnamed: 0", axis=1)['title']

    for row in products:
        make_single_prediction.delay(row).forget()


@app.task(queue='crop_field_segmentation')
def make_single_prediction(row):
    file_name = f'/storage/caches/imagiflow/safe_products/{row}.SAFE'
    wkt = predict_safe_regions(
        safe_folder_path=file_name,
        tile_width=6000,
        tile_height=6000
    )
    save_wkt(
        wkt=wkt,
        filepath=f'/home/faranio/Desktop/Egistic/Crop_Field_Segmentation/data/Russia/{row}.gpkg'
    )
