import geopandas as gpd
import os
import psycopg2
import shapely.wkb as shwkb

from shapely.ops import unary_union


def fetch_dataframe():
    filename = "labels.geojson"

    if os.path.exists(filename):
        df = gpd.read_file(filename)
    else:
        conn = psycopg2.connect(
            dbname='egistic_2.0',
            user='docker',
            password='PNdvVpM3VQoMOVOeu8YCbGc69eo2X3iC',
            host='35.154.215.124',
            port='8050'
        )
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


def save_dataframe():
    df = fetch_dataframe()
    polygons = []

    for idx, row in df.iterrows():
        for polygon in row['geometry']:
            polygons.append(polygon)

    result = unary_union(polygons)
    print(len(result), type(result), type(result[0]), result[0])

    with open('resultant_wkt.txt', 'w') as file:
        file.write(result.wkt)


if __name__ == "__main__":
    save_dataframe()
