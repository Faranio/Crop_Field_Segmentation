import geopandas as gpd
import os
import psycopg2
import shapely.wkb as shwkb


def fetch_dataframe():
    """
    Fetch the dataset with all of the fields in Kazakhstan.
    """
    filename = "labels.geojson"

    if os.path.exists(filename):
        df = gpd.read_file(filename)
    else:
        # Connect to Egistic_2.0 database
        conn = psycopg2.connect()
        cur = conn.cursor()
        sql = '''SELECT id, geometry FROM divided_fields;'''
        cur.execute(sql)
        df = cur.fetchall()
        cur.close()
        conn.close()

        df = gpd.GeoDataFrame(df, columns=['id', 'geometry'], crs='EPSG:3857')
        df['geometry'] = df['geometry'].map(lambda geometry: shwkb.loads(geometry, hex=True))
        df.to_file(filename, driver='GeoJSON')

    return df
