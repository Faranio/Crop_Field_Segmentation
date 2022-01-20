import os

from celery import Celery
from lgblkb_tools import logger

from config import broker_url, data_folder, test_folder, settings
from inference import predict_safe_regions

app = Celery('Crop_Field_Segmentation', broker=broker_url, backend='rpc://')
app.config_from_object(settings.CELERY.config)


@app.task(name='crop_field_segmentation', queue='crop_field_segmentation')
def perform_predictions(file_paths):
    for file_path in file_paths:
        file_name = file_path.split('/')[-1]
        
        if os.path.exists(data_folder['03_results'][f'{file_name}.gpkg']):
            continue

        predict_safe_regions(
            region='UA',
            safe_folder_path=file_path,
            tile_width=20000,
            tile_height=20000,
            confidence=0.8,
            intersection_threshold=0.5,
            save=True
        )


files = os.listdir(test_folder)
file_paths = []

for file in files:
    name, ext = os.path.splitext(file)
    
    if ext == ".tiff":
        file_paths.append(os.path.join(test_folder, file))
    
perform_predictions(file_paths)
