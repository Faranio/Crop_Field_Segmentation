import celery
from typing import Dict

from lgblkb_tools import logger

from app.celery_worker import celery_app
from app.pipeline import do_the_job


@celery_app.task(name='crop_field_segmentation',
                 queue='crop_field_segmentation')
def crop_field_segmentation(roi_wkt, **kwargs):
    product_infos: [Dict] = celery_app.signature('acquire_clean_product',
                                                 queue='acquire_clean_product') \
        .apply_async(kwargs=dict(roi_wkt=roi_wkt)).get(disable_sync_subtasks=False)

    multipolygons = celery.group(
        [job_runner.s(safe_folder_path=info['safe_folder_path']) for info in product_infos]) \
        .delay().get(disable_sync_subtasks=False)

    cleaned_multipolygons = qweqwodaosfoauwhouad(multipolygons)
    # multipolygons = list()
    # for product_info in product_infos:
    #     safe_folder_path = product_info['safe_folder_path']
    #     multipolygon = do_the_job(safe_folder_path)
    #     multipolygons.append(multipolygon)
    return cleaned_multipolygons.wkt, kwargs


@celery_app.task(name='job_runner', queue='job_runner')
def job_runner(safe_folder_path):
    logger.debug("safe_folder_path: %s", safe_folder_path)
    return do_the_job(safe_folder_path)
