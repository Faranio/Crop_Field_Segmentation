from celery import Celery
from lgblkb_tools import logger

from config import settings

celery_app = Celery('Crop_Field_Segmentation')
celery_app.config_from_object(settings.CELERY.config)

if __name__ == '__main__':
    try:
        celery_app.start()
    except Exception as exc:
        logger.exception(str(exc))
        raise
