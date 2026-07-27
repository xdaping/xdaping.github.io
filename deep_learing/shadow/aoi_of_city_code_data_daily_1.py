# encoding=utf-8
import argparse
import time
from datetime import datetime, timedelta

import shapely.wkt as wkt
from rtree import index
from pyspark.sql.types import *
from jobs.spark_job import SparkJob
from pyproj import Geod

geod = Geod(ellps="WGS84")

class DataProcess(object):

    def __init__(self, sc, sql_context):
        self.sc = sc
        self.sqlContext = sql_context

    def data_to_hive(self, results, output_table, dt):
        #print('************data_to_hive **************', output_table, dt)
        #print('type bill_time:', type(results))

        fields = [
            StructField("aoi_id", IntegerType(), True),
            StructField("aoi_polygon", StringType(), True),
            StructField("code", IntegerType(), True),
            StructField("name", StringType(), True),
            StructField("level", IntegerType(), True),
            StructField("parentcode", IntegerType(), True),
            StructField("area", DoubleType(), True)
        ]

        dataf = self.sqlContext.createDataFrame(results, schema=StructType(fields))
        #print(dataf.take(1))
        #print()
        dataf.registerTempTable("tmp")
        self.sqlContext.sql("set hive.exec.dynamic.partition=true")
        self.sqlContext.sql("set hive.exec.parallel = true")
        self.sqlContext.sql("set hive.exec.dynamic.partition.mode = nonstrict")
        self.sqlContext.sql("set spark.sql.shuffle.partitions = 2000")
        self.sqlContext.sql(
            f"""
            CREATE TABLE IF NOT EXISTS {output_table}
            (
                `aoi_id`              bigint        COMMENT '',
                `aoi_polygon`         string        COMMENT '',
                `code`                bigint        COMMENT '',
                `name`                string        COMMENT '',
                `level`               bigint        COMMENT '',
                `parentcode`          bigint        COMMENT '',
                `area`                double        COMMENT ''           
            ) COMMENT ''
            PARTITIONED BY (dt string)
            ROW FORMAT DELIMITED FIELDS TERMINATED BY '\t'
            STORED AS ORC
            """)

        self.sqlContext.sql(
            f"""insert  overwrite table {output_table} partition(dt='{dt}')
                    select * from tmp""")
        return

    def process(self, aoi_sql_query, code_sql_query, output_table, dt):
        #print(f"*****任务开始*****")
        #start = time.time()

        def get_rtree_of_city(brd_geom_list):
            # code,geometry,parentcode,name,level
            rtree_index = index.Index()
            rtree_dict = {}
            for row in brd_geom_list:
                code_id = int(row[0])
                code_p = wkt.loads(row[1])
                rtree_index.insert(code_id, code_p.bounds)
                rtree_dict[code_id] = [code_p, int(row[2]), row[3], int(row[4])]
            return rtree_index, rtree_dict

        def aoi_to_city(itr):
            b_city_geom_list = brd_city_geom_list.value
            rtree_index_city, rtree_dict_city = get_rtree_of_city(b_city_geom_list)

            aoi_city_list = []
            for row in itr:  #row.bm_aoi_id, row.polygon
                aoi_id = row.bm_aoi_id
                aoi_geom = wkt.loads(row.polygon)

                max_area = -1
                max_code_id = -1
                max_name = ''
                max_level = -1
                max_parentcode_id = -1
                block_match_citys = set(rtree_index_city.intersection(aoi_geom.bounds))
                for code_id in block_match_citys:
                    city_geom, parentcode_id, name, level = rtree_dict_city[code_id]
                    if city_geom.intersects(aoi_geom):
                        area = city_geom.intersection(aoi_geom).area
                        if area > max_area:
                            max_area = area
                            max_code_id = code_id
                            max_name = name
                            max_level = level
                            max_parentcode_id = parentcode_id

                aoi_city_list.append(
                    [aoi_id, row.polygon, max_code_id, max_name, max_level, max_parentcode_id,
                     round(abs(geod.geometry_area_perimeter(aoi_geom)[0]), 2)])

            return aoi_city_list




        city_geom_list = self.sqlContext.sql(code_sql_query).collect()
        #print(len(city_geom_list), city_geom_list[:1])
        brd_city_geom_list = self.sc.broadcast(city_geom_list)

        aoi_city_rdd = self.sqlContext.sql(aoi_sql_query).rdd.repartition(500).mapPartitions(aoi_to_city)

        self.data_to_hive(aoi_city_rdd, output_table, dt)

        #end = time.time()
        #print("end - start", end - start)

    def run_city_county(self, spark_job, dt):


        aoi_sql_query = f"""select bm_aoi_id, polygon
                                    from mart_peisonglbs.bm_aoi_release_data
                                    where dt='{dt}'"""

        code_sql_query = f"""SELECT code,geometry,parentcode,name,level
                             FROM mart_peisongpa.district_domestic_total_level_6_20250416"""

        print(aoi_sql_query)
        print(code_sql_query)

        # temp_spl = f"""ALTER TABLE mart_peisongpa.aoi_of_city_code_data_daily ADD COLUMNS (area DOUBLE)"""
        # self.sqlContext.sql(temp_spl)

        output_table = "mart_peisongpa.aoi_of_city_code_data_daily"

        self.process(aoi_sql_query, code_sql_query, output_table, dt)


def process(args):
    """Spark任务接口 """

    parser = argparse.ArgumentParser()
    parser.add_argument('--dt', type=str, default='20250707', required=True)
    args, unknown = parser.parse_known_args()

    spark_job = SparkJob()
    job = DataProcess(spark_job.sc, spark_job.sqlContext)

    dt = args.dt
    job.run_city_county(spark_job, dt)

