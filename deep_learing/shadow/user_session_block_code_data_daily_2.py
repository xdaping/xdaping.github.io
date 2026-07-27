# encoding=utf-8
import argparse
import time
from datetime import datetime, timedelta

import shapely.wkt as wkt
from rtree import index
from pyspark.sql.types import *
from jobs.spark_job import SparkJob
from jobs.algorithms import polygon_geohasher


class DataProcess(object):

    def __init__(self, sc, sql_context):
        self.sc = sc
        self.sqlContext = sql_context

    def data_to_hive(self, results, output_table, dt):
        print('************data_to_hive **************', output_table, dt)
        #print('type bill_time:', type(results))

        fields = [
            StructField("blockid", StringType(), True),
            StructField("detailed_category", StringType(), True),
            StructField("block_geom_wkt", StringType(), True),
            StructField("code", IntegerType(), True),
            StructField("name", StringType(), True),
            StructField("level", IntegerType(), True),
            StructField("parentcode", IntegerType(), True)
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
                `blockid`             string        COMMENT '',
                `detailed_category`   string        COMMENT '',
                `block_geom_wkt`      string        COMMENT '',
                `code`                bigint        COMMENT '',
                `name`                string        COMMENT '',
                `level`               bigint        COMMENT '',
                `parentcode`          bigint        COMMENT ''           
            ) COMMENT ''
            PARTITIONED BY (dt string)
            ROW FORMAT DELIMITED FIELDS TERMINATED BY '\t'
            STORED AS ORC
            """)

        self.sqlContext.sql(
            f"""insert  overwrite table {output_table} partition(dt='{dt}')
                    select * from tmp""")
        return

    def process(self, block_sql_query, dau_sql_query, city_geom_query, output_table, dt):
        print(f"*****任务开始*****")
        #start = time.time()

        def get_rtree_of_geohash(brd_geom_list):
            # geohash_id
            rtree_index = index.Index()
            for row in brd_geom_list:
                geohash_i = polygon_geohasher.str_to_int(row[0])
                geohash_p = polygon_geohasher.geohash_to_polygon(row[0])
                rtree_index.insert(geohash_i, geohash_p.bounds)
            return rtree_index

        def get_rtree_of_city(brd_geom_list):
            #code,geometry,parentcode,name,level
            rtree_index = index.Index()
            rtree_dict = {}
            for row in brd_geom_list:
                code_id = int(row[0])
                code_p = wkt.loads(row[1])
                rtree_index.insert(code_id, code_p.bounds)
                rtree_dict[code_id] = [code_p, int(row[2]), row[3], int(row[4])]
            return rtree_index, rtree_dict

        def block_to_user(itr):
            b_user_geohash_list = brd_user_geohash_list.value
            rtree_index_geohash = get_rtree_of_geohash(b_user_geohash_list)

            b_city_geom_list = brd_city_geom_list.value
            rtree_index_city, rtree_dict_city = get_rtree_of_city(b_city_geom_list)

            user_block_list = []
            for row in itr:  #row.blockid, row.detailed_category, row.block_geom_wkt
                block_geom = wkt.loads(row.block_geom_wkt)
                block_match_users = set(rtree_index_geohash.intersection(block_geom.bounds))

                for geohash_i in block_match_users:
                    geohash_p = polygon_geohasher.geohash_to_polygon(polygon_geohasher.int_to_str(geohash_i))
                    if geohash_p.intersects(block_geom):

                        max_area = -1
                        max_code_id = -1
                        max_name = ''
                        max_level = -1
                        max_parentcode_id = -1
                        block_match_citys = set(rtree_index_city.intersection(block_geom.bounds))
                        for code_id in block_match_citys:
                            city_geom, parentcode_id, name, level = rtree_dict_city[code_id]
                            if city_geom.intersects(block_geom):
                                area = city_geom.intersection(block_geom).area
                                if area > max_area:
                                    max_area = area
                                    max_code_id = code_id
                                    max_name = name
                                    max_level = level
                                    max_parentcode_id = parentcode_id

                        user_block_list.append(
                            [row.blockid, row.detailed_category, row.block_geom_wkt, max_code_id, max_name, max_level, max_parentcode_id])

                        break

            return user_block_list

        #
        user_geohash_list = self.sqlContext.sql(dau_sql_query).collect()
        #print(user_geohash_list[:5], len(user_geohash_list))
        brd_user_geohash_list = self.sc.broadcast(user_geohash_list)

        city_geom_list = self.sqlContext.sql(city_geom_query).collect()
        #print(city_geom_list[:5], len(city_geom_list))
        brd_city_geom_list = self.sc.broadcast(city_geom_list)

        user_block_rdd = self.sqlContext.sql(block_sql_query).rdd.repartition(1000).mapPartitions(block_to_user)

        self.data_to_hive(user_block_rdd, output_table, dt)

        #end = time.time()
        #print("end - start", end - start)

    def run_city_county(self, spark_job, dau_b_dt, dau_e_dt, dt):

        block_sql_query = f"""select blockid, detailed_category, block_geom_wkt
                            from mart_peisongpa.block_of_city_code_data
                            where ver='block_20250401_code_20250416' and detailed_category!='水系'"""

        dau_sql_query = f"""select user_geohash_id from 
                (select mt_geohash(latitude,longitude,6) as user_geohash_id
                from mart_waimai.topic_flow_sdk_effective_union_d
                where dt between '{dau_b_dt}' and '{dau_e_dt}'              
                and is_dau=1 and is_catering_dau=1 and is_effective_dau=1  
                and user_id>0 and longitude > 0 and latitude>0) a 
                where user_geohash_id is not null 
                group by 1 having count(1)>=7"""

        city_geom_query = f"""SELECT code,geometry,parentcode,name,level
                             FROM mart_peisongpa.district_domestic_total_level_6_20250416"""

        print(block_sql_query)
        print(dau_sql_query)
        print(city_geom_query)

        output_table = "mart_peisongpa.user_session_block_code_data_daily"



        self.process(block_sql_query, dau_sql_query, city_geom_query, output_table, dt)


def process(args):
    """Spark任务接口 """

    parser = argparse.ArgumentParser()
    parser.add_argument('--dt', type=str, default='20250707', required=True)
    parser.add_argument("--lasting_days", type=int, default=30, required=True)
    args, unknown = parser.parse_known_args()

    spark_job = SparkJob()
    job = DataProcess(spark_job.sc, spark_job.sqlContext)

    dau_e_dt = args.dt
    lasting_days = args.lasting_days

    end_date = datetime.strptime(dau_e_dt, "%Y%m%d")
    start_date = end_date - timedelta(days=lasting_days - 1)  # 包含起始日
    dau_b_dt = start_date.strftime("%Y%m%d")


    job.run_city_county(spark_job, dau_b_dt, dau_e_dt, args.dt)
