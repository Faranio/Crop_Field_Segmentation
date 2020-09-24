from pathlib import Path
from pprint import pformat

import shapely.geometry as shg
import shapely.wkt as shwkt
import geopandas as gpd
import more_itertools as mit
import celery
from typing import Dict

from lgblkb_tools import logger
from lgblkb_tools.telegram_notify import egistic_notify

from app.app import s2_storage_folder
from app.celery_worker import celery_app
from app.pipeline import segment_safe_product, remove_overlaps


def celery_hook(payload, task_name, queue='', signature_options=None, apply_async=None):
    queue = queue or task_name
    celery_app.signature(task_name, queue=queue, **(signature_options or {})) \
        .apply_async(**dict(dict(serializer='json'), **(apply_async or {})), kwargs=payload)


@celery_app.task(name='crop_field_segmentation',
                 queue='crop_field_segmentation')
def crop_field_segmentation(roi_wkt, hook=None, **kwargs):
    product_infos: [Dict] = celery_app.signature('acquire_roi_products',
                                                 queue='acquire_roi_products') \
        .apply_async(kwargs=dict(roi_wkt=roi_wkt)).get(disable_sync_subtasks=False)
    multipolygons = celery.group(
        [image_segmentor.s(info['safe_folder']) for info in product_infos]) \
        .delay().get(disable_sync_subtasks=False)
    multipolygons = gpd.GeoDataFrame(dict(geometry=[shwkt.loads(x) for x in multipolygons]), crs='epsg:3857')
    out = shg.MultiPolygon(remove_overlaps(multipolygons.geometry, []))
    # multipolygons = list()
    # for product_info in product_infos:
    #     safe_folder_path = product_info['safe_folder_path']
    #     multipolygon = do_the_job(safe_folder_path)
    #     multipolygons.append(multipolygon)
    if hook:
        celery_hook(dict(fields=out.wkt, **kwargs), **hook)
    egistic_notify.send_message(f"""Crop field segmentation done:
meta = {pformat(kwargs)}
""")
    return out.wkt


@celery_app.task(name='image_segmentor', queue='image_segmentor')
def image_segmentor(safe_folder_path):
    safe_folder_path = s2_storage_folder.path + safe_folder_path
    logger.debug("safe_folder_path: %s", safe_folder_path)
    return segment_safe_product(safe_folder_path).wkt
