import celery

from celery import Celery

from config import broker_url, settings
from inference import predict_safe_regions

app = Celery('Crop_Field_Segmentation', broker=broker_url, backend='rpc://')
app.config_from_object(settings.CELERY.config)
app = Celery('celery_worker', broker=broker_url, backend='rpc://')


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
