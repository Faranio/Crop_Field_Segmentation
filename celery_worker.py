import celery
import os

from celery import Celery

from config import broker_url, settings
from inference import predict_safe_regions

app = Celery('Crop_Field_Segmentation', broker=broker_url, backend='rpc://')
app.config_from_object(settings.CELERY.config)


@app.task(name='crop_field_segmentation', queue='crop_field_segmentation')
def make_prediction(file_paths):
    celery.group([predict_safe_regions.s(
        safe_folder_path=file_path,
        tile_width=20000,
        tile_height=20000,
        confidence=0.8,
        intersection_threshold=0.5,
        save=True
    ) for file_path in file_paths]).delay().get(disable_sync_subtasks=False)


folder_path = '/usr/src/app/data/04_test'
safe_file_paths = os.listdir(folder_path)
resultant_file_paths = []

for file in safe_file_paths:
    keywords = [f'20211{x}' for x in [
        '024',
        '025',
        '026',
        '027',
        '028',
        '029',
        '030',
        '031',
        '101'
    ]]
    
    for keyword in keywords:
        if keyword in file:
            cur_file_path = os.path.join(folder_path, file)
            resultant_file_paths.append(cur_file_path)
            
print(resultant_file_paths)

# make_prediction(resultant_file_paths)
