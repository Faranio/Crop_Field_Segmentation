import geopandas as gpd
import shapely.wkt as shwkt
from lgblkb_tools import logger
from app.celery_tasks import crop_field_segmentation


@logger.trace()
def main():
    # roi_wkt = 'POLYGON ((67.5 56.090427143991526, 73.970947265625 56.090427143991526, 73.970947265625
    # 56.93298739609704, 67.5 56.93298739609704, 67.5 56.090427143991526))'
    roi_wkt = 'POLYGON ((7514065.628545966 7576438.243626668, 8234408.183105467 7576438.243626668, 8234408.183105467 ' \
              '7746434.194532901, 7514065.628545966 7746434.194532901, 7514065.628545966 7576438.243626668)) '
    # roi_geom = gpd.GeoDataFrame(dict(geometry=[shwkt.loads(roi_wkt)]),
    #                             crs='epsg:4326').to_crs('epsg:3857').iloc[0].geometry
    fields = crop_field_segmentation.delay(roi_wkt).get()
    logger.debug("fields:\n%s", fields)
    pass


if __name__ == '__main__':
    main()
