# encoding=utf-8
import argparse
import time
from rtree import index
from pyspark.sql.types import *
from shapely import wkt, ops
from utils.base_util import BaseUtil
from jobs.spark_job import SparkJob


class DataProcess(object):

    def __init__(self, sc, sql_context):
        self.sc = sc
        self.sqlContext = sql_context

    def data_to_hive(self, results, output_table, dt):
        print('************data_to_hive **************', output_table, dt)
        print('type bill_time:', type(results))

        fields = [
            StructField("blockid", StringType(), True),
            StructField("detailed_category", StringType(), True),
            StructField("part", IntegerType(), True),
            StructField("raw_shadow_p_wkt", StringType(), True),
            StructField("code", IntegerType(), True),
            StructField("name", StringType(), True),
            StructField("level", IntegerType(), True),
            StructField("parentcode", IntegerType(), True)
        ]

        dataf = self.sqlContext.createDataFrame(results, schema=StructType(fields))
        # print(dataf.take(1))
        print()
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
                `part`                bigint        COMMENT '',
                `raw_shadow_p_wkt`    string        COMMENT '',
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

    def process(self, block_sql_query, aoi_sql_query, output_table, ver):
        print(f"*****任务开始*****")
        start = time.time()

        def get_rtree_of_polygon(res_arr):
            # bm_aoi_id, polygon
            rtree_index = index.Index()
            rtree_dict = {}
            for row in res_arr:
                aoi_id = int(row[0])
                aoi_p = wkt.loads(row[1])
                rtree_dict[aoi_id] = aoi_p
                rtree_index.insert(aoi_id, aoi_p.bounds)
            return rtree_index, rtree_dict

        def is_small_pencil_block(raw_shadow, detailed_category):
            if detailed_category in ['居住', '办公', '商业']:
                return False

            if raw_shadow.area * 1e10 < 500 or (
                    raw_shadow.length * 1.0 / raw_shadow.area * (raw_shadow.length / 20) > (20 / 9)
                    and raw_shadow.area * 1e10 <= 5000) or (
                    raw_shadow.length * 1.0 / raw_shadow.area * (raw_shadow.length / 20) > 10
                    and raw_shadow.area * 1e10 <= 100000):
                if raw_shadow.length * 1.0 / raw_shadow.area * (raw_shadow.length / 20) <= 2:
                    return False
                return True
            else:
                return False

        def get_raw_shaodow(itr):
            b_aoi_list = brd_aoi_list.value
            rtree_index, aoi_dict = get_rtree_of_polygon(b_aoi_list)

            shadow_list = []
            for row in itr:  # blockid,detailed_category,block_geom_wkt,code,name,level,parentcode
                part = 1
                blockid = row.blockid
                block_geom = wkt.loads(row.block_geom_wkt)
                detailed_category = row.detailed_category

                block_match_aois = set(rtree_index.intersection(block_geom.bounds))
                match_aoi_list = [aoi_dict[i] for i in block_match_aois if aoi_dict[i].intersects(block_geom)]

                if len(match_aoi_list) == 0:
                    shadow = block_geom
                    if is_small_pencil_block(shadow, detailed_category):
                        continue
                    shadow_list.append(
                        [blockid, detailed_category, part, str(shadow), row.code, row.name, row.level, row.parentcode])
                    part += 1
                    continue

                aois_geom = ops.unary_union(match_aoi_list)

                raw_shadow = block_geom.difference(aois_geom.buffer(1e-5))

                raw_shadow = raw_shadow.buffer(-1e-4).buffer(1e-4).simplify(1e-5)

                if raw_shadow.is_empty:
                    continue

                # continue
                if raw_shadow.geom_type == "MultiPolygon":
                    for geom in raw_shadow.geoms:  # 遍历所有子多边形
                        shadow = geom
                        if is_small_pencil_block(geom, detailed_category):
                            continue
                        shadow_list.append(
                            [blockid, detailed_category, part, str(shadow), row.code, row.name, row.level,
                             row.parentcode])
                        part += 1
                else:
                    shadow = raw_shadow
                    if is_small_pencil_block(shadow, detailed_category):
                        continue
                    shadow_list.append(
                        [blockid, detailed_category, part, str(shadow), row.code, row.name, row.level, row.parentcode])
                    part += 1

            return shadow_list

        #
        aoi_list = self.sqlContext.sql(aoi_sql_query).collect()
        brd_aoi_list = self.sc.broadcast(aoi_list)

        raw_shadow_rdd = self.sqlContext.sql(block_sql_query).rdd.repartition(500).mapPartitions(get_raw_shaodow)

        self.data_to_hive(raw_shadow_rdd, output_table, ver)

        end = time.time()
        print("end - start", end - start)

    def run_city_county(self, spark_job, dt):

        block_sql_query = f"""select blockid,detailed_category,block_geom_wkt,code,name,level,parentcode  
                            from mart_peisongpa.user_session_block_code_data_daily
                            where dt='{dt}'"""

        aoi_sql_query = f"""select bm_aoi_id, polygon
                            from mart_peisonglbs.bm_aoi_release_data
                            where dt='{dt}'"""

        print(block_sql_query)
        print(aoi_sql_query)

        output_table = "mart_peisongpa.raw_shadow_of_code_block_aoi_daily"
        # spark_job.drop_table(output_table)

        self.process(block_sql_query, aoi_sql_query, output_table, dt)


def process(args):
    """Spark任务接口 """
    parser = argparse.ArgumentParser()
    parser.add_argument('--dt', type=str, default='20250707', required=True)
    args, unknown = parser.parse_known_args()

    spark_job = SparkJob()
    job = DataProcess(spark_job.sc, spark_job.sqlContext)

    dt = args.dt

    job.run_city_county(spark_job, dt)
