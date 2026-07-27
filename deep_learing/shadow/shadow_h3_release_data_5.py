# encoding=utf-8
import argparse
import time
from collections import defaultdict

import shapely
from pyproj import Geod
from pyspark.sql.types import *

from jobs.spark_job import SparkJob
from shapely import ops
from utils.base_util import BaseUtil
from pyspark.rdd import Partitioner
import shapely
from rtree import index
from shapely import wkt
from shapely.geometry import Polygon, MultiPolygon, LineString
import h3
from shapely.ops import nearest_points
import networkx as nx

precisions = 9
geod = Geod(ellps="WGS84")




class DataProcess(object):

    def __init__(self, sc, sql_context):
        self.sc = sc
        self.sqlContext = sql_context

    def data_to_hive(self, results, output_table, dt):
        print('************data_to_hive **************', output_table, dt)
        print('type bill_time:', type(results))

        fields = [
            StructField("shadow_id", LongType(), True),
            StructField("h3_code", StringType(), True),
            StructField("shadow_p_wkt", StringType(), True),
            StructField("area", DoubleType(), True),
            StructField("heat", IntegerType(), True),
            StructField("blockid", StringType(), True),
            StructField("detailed_category", StringType(), True),
            StructField("code", IntegerType(), True),
            StructField("name", StringType(), True),
            StructField("level", IntegerType(), True),
            StructField("parentcode", IntegerType(), True)
        ]

        dataf = self.sqlContext.createDataFrame(results, schema=StructType(fields))
        #print(dataf.take(1))
        print()
        dataf.registerTempTable("tmp")
        self.sqlContext.sql("set hive.exec.dynamic.partition=true")
        self.sqlContext.sql("set hive.exec.parallel = true")
        self.sqlContext.sql("set hive.exec.dynamic.partition.mode = nonstrict")
        self.sqlContext.sql("set spark.sql.shuffle.partitions = 4000")
        self.sqlContext.sql("set spark.shuffle.compress=true")
        self.sqlContext.sql("set spark.io.compression.codec=snappy")

        self.sqlContext.sql(
            f"""
            CREATE TABLE IF NOT EXISTS {output_table}
            (
                `shadow_id`                bigint        COMMENT '',
                `h3_code`                  string        COMMENT '',
                `shadow_p_wkt`             string        COMMENT '',
                `area`                     double        COMMENT '',
                `heat`                     bigint        COMMENT '',
                `blockid`                  string        COMMENT '',
                `detailed_category`        string        COMMENT '',
                `code`                     bigint        COMMENT '',
                `name`                     string        COMMENT '',
                `level`                    bigint        COMMENT '',
                `parentcode`               bigint        COMMENT ''          
            ) COMMENT ''
            PARTITIONED BY (dt string)
            ROW FORMAT DELIMITED FIELDS TERMINATED BY '\t'
            STORED AS ORC
            """)
        self.sqlContext.sql(
            f"""insert  overwrite table {output_table} partition(dt='{dt}')
                    select * from tmp""")
        return

    def process(self, raw_shadow_sql_query, aoi_change_sql_query, output_table, ver_dt):
        print(f"*****任务开始*****")
        start = time.time()

        def get_rtree_of_aoi(brd_geom_list, code):
            # code,aoi,polygon,change_type
            rtree_index = index.Index()
            rtree_dict = {}
            for row in brd_geom_list:
                if int(code) == int(row[0]) and int(row[3]) in (1, 3):  # 增删
                    aoi_id = int(row[1])
                    aoi_p = wkt.loads(row[2])
                    rtree_index.insert(aoi_id, aoi_p.bounds)
                    rtree_dict[aoi_id] = aoi_p
            return rtree_index, rtree_dict

        def convert_to_simple_polygon(polygon):
            exterior = polygon.exterior
            interiors = list(polygon.interiors)

            for interior in interiors:
                point1, point2 = nearest_points(interior, exterior)
                line_b = LineString([(point1.x, point1.y), (point2.x, point2.y)]).buffer(0.00001, resolution=2)
                polygon = polygon.difference(line_b)
            return polygon

        def fix_simple_polygon(shadow_p_wkt_raw):

            cnt = len(shadow_p_wkt_raw.exterior.coords)

            if cnt >= 500:
                shadow_p_wkt_raw = shadow_p_wkt_raw.simplify(0.000001).buffer(0)
                cnt = len(shadow_p_wkt_raw.exterior.coords)
                if cnt >= 500:
                    shadow_p_wkt_raw = shadow_p_wkt_raw.simplify(0.000002).buffer(0)
                    cnt = len(shadow_p_wkt_raw.exterior.coords)
                    if cnt >= 500:
                        shadow_p_wkt_raw = shadow_p_wkt_raw.simplify(0.000003).buffer(0)

            shadow_p_wkt = wkt.loads(wkt.dumps(shadow_p_wkt_raw, rounding_precision=6)).buffer(0)

            if isinstance(shadow_p_wkt, MultiPolygon):
                # 取面积最大的polygon
                max_area = 0
                max_poly = None
                for poly in shadow_p_wkt.geoms:
                    if poly.is_valid and poly.area > max_area:
                        max_area = poly.area
                        max_poly = poly
                if max_poly is not None:
                    shadow_p_wkt = max_poly

            return shadow_p_wkt

        def fix_col_info_with_aoi(col_info, rtree_index_aoi, rtree_dict_aoi):
            if len(rtree_dict_aoi) > 0:
                shadow_p_wkt = wkt.loads(col_info[2]).buffer(0)
                try:
                    match_aois = set(rtree_index_aoi.intersection(shadow_p_wkt.bounds))
                    inter_aoi_p_list = []

                    for match_aoi_id in match_aois:
                        match_aoi_wkt = rtree_dict_aoi[match_aoi_id]
                        if shadow_p_wkt.intersects(match_aoi_wkt):

                                inter_raito = shadow_p_wkt.intersection(match_aoi_wkt).area / shadow_p_wkt.area
                                if inter_raito > 0.1:
                                   inter_aoi_p_list.append(match_aoi_wkt)


                    if len(inter_aoi_p_list) > 0:

                        inter_aoi_p = ops.unary_union(inter_aoi_p_list)
                        current_poly = shadow_p_wkt.difference(inter_aoi_p)

                        if isinstance(current_poly, MultiPolygon):
                            # 取面积最大的polygon
                            max_area = 0
                            max_poly = None
                            for poly in current_poly.geoms:
                                if poly.is_valid and poly.area > max_area:
                                    max_area = poly.area
                                    max_poly = poly
                            if max_poly is not None:
                                current_poly = max_poly

                        if current_poly.interiors.__len__() > 0:
                            current_poly = convert_to_simple_polygon(current_poly)

                        col_info[2] = str(current_poly)
                        col_info[3] = round(abs(geod.geometry_area_perimeter(current_poly)[0]), 2)
                except Exception as e:
                    print("fix_col_info_with_aoi出错shadow_p_wkt:", shadow_p_wkt)

                return col_info

        def get_process_data(row):  # code 	col_list   cnt
            print("当前code：", row.code)
            print("当前dt：", ver_dt)

            b_aoi_change_list = brd_aoi_change_list.value
            rtree_index_aoi, rtree_dict_aoi = get_rtree_of_aoi(b_aoi_change_list, row.code)

            print("aoi数量", len(rtree_dict_aoi))

            col_dict = {}
            rtree_index = index.Index()
            for col in row.col_list:
                shadow_id = int(col[0])
                shadow_p_wkt = wkt.loads(col[2])
                if shadow_p_wkt.is_empty:
                    continue

                if not shadow_p_wkt.is_valid:
                    shadow_p_wkt = shadow_p_wkt.buffer(0)
                col_dict[shadow_id] = [shadow_id, col[1], str(shadow_p_wkt), float(col[3]), int(col[4]), col[5], col[6],
                                       int(col[7]), col[8], int(col[9]), int(col[10]), col[11]]

                if col[11] != ver_dt and shadow_p_wkt.is_valid:
                    try:
                        rtree_index.insert(shadow_id, shadow_p_wkt.bounds)
                    except Exception as e:
                        print("get_process_data出错shadow_p_wkt:", shadow_p_wkt)

            for shadow_id, info in col_dict.items():
                shadow_p_wkt = wkt.loads(info[2]).buffer(0)
                if info[11] == ver_dt and shadow_p_wkt.is_valid:
                    try:
                        match_shadows = set(rtree_index.intersection(shadow_p_wkt.bounds))
                        max_inter_raito = 0
                        for match_shadow_id in match_shadows:
                            match_shadow_wkt = wkt.loads(col_dict[match_shadow_id][2]).buffer(0)
                            if shadow_p_wkt.intersects(match_shadow_wkt):
                                inter_raito = shadow_p_wkt.intersection(match_shadow_wkt).area / shadow_p_wkt.area

                                if inter_raito > max_inter_raito:
                                    max_inter_raito = inter_raito
                    except Exception as e:
                        print("max_inter_raito出错shadow_p_wkt:", shadow_p_wkt)
                    if max_inter_raito < 0.1:  # 新增
                        col_info = col_dict[shadow_id]
                        fix_col_info_with_aoi(col_info, rtree_index_aoi, rtree_dict_aoi)

                        current_poly = fix_simple_polygon(wkt.loads(col_info[2]))

                        if current_poly.is_valid and not current_poly.is_empty:
                            col_info[2] = str(current_poly)
                            col_info[3] = round(abs(geod.geometry_area_perimeter(current_poly)[0]), 2)
                            yield col_info[:-1]
                else:  # 原始
                    col_info = col_dict[shadow_id]
                    fix_col_info_with_aoi(col_info, rtree_index_aoi, rtree_dict_aoi)

                    current_poly = fix_simple_polygon(wkt.loads(col_info[2]))

                    if current_poly.is_valid and not current_poly.is_empty:
                        col_info[2] = str(current_poly)
                        col_info[3] = round(abs(geod.geometry_area_perimeter(current_poly)[0]), 2)
                        yield col_info[:-1]

        aoi_change_list = self.sqlContext.sql(aoi_change_sql_query).collect()
        brd_aoi_change_list = self.sc.broadcast(aoi_change_list)

        shadow_h3_rdd = self.sqlContext.sql(raw_shadow_sql_query).repartition(3500, "code").rdd.flatMap(get_process_data)

        #shadow_h3_rdd.take(2)

        self.data_to_hive(shadow_h3_rdd, output_table, ver_dt)

        end = time.time()
        print("end - start", end - start)

    def run_city_county(self, spark_job, dt):

        raw_shadow_sql_query = f"""SELECT code, collect_list(col) as col_list, count(*) as cnt
                                    from 
                                    (
                                    SELECT COALESCE(a.code, b.code) as code, 
                                            array(
                                            COALESCE(a.shadow_id, b.shadow_id), 
                                            COALESCE(a.h3_code, b.h3_code),
                                            COALESCE(a.shadow_p_wkt, b.shadow_p_wkt),
                                            COALESCE(a.area, b.area), 
                                            COALESCE(a.heat, b.heat),
                                            COALESCE(a.blockid, b.blockid), 
                                            COALESCE(a.detailed_category, b.detailed_category), 
                                            COALESCE(a.code, b.code), 
                                            COALESCE(a.name, b.name), 
                                            COALESCE(a.level, b.level), 
                                            COALESCE(a.parentcode, b.parentcode),
                                            COALESCE(a.dt, b.dt)
                                            ) as col
                                    from (
                                      select * from 
                                      (select *, ROW_NUMBER() OVER (PARTITION BY shadow_id ORDER BY area DESC) AS nums
                                      from mart_peisongpa.shadow_h3_release_data
                                      where dt in ( select max(dt) from mart_peisongpa.shadow_h3_release_data where dt < '{dt}')
                                      ) t  where nums = 1                                 
                                    ) a
                                    FULL OUTER JOIN 
                                    (                                    
                                      select * from 
                                      (select *, ROW_NUMBER() OVER (PARTITION BY shadow_id ORDER BY area DESC) AS nums
                                      from mart_peisongpa.shadow_h3_heat_of_block_aoi_daily
                                      where dt='{dt}' and heat >=3
                                      ) t  where nums = 1                                
                                    ) b
                                    on a.shadow_id = b.shadow_id 
                                    ) t
                                    group by 1"""

        aoi_change_sql_query = f"""SELECT 
                                    COALESCE(a_code, b_code) as code,
                                    COALESCE(a_aoi_id, b_aoi_id) as aoi_id,
                                    COALESCE(a_polygon, a_polygon) as polygon,
                                    if(b_polygon is null, 1,  if(a_polygon is null, 2, 3) ) as change_type
                                    from 
                                    (SELECT a.code as a_code, a.aoi_id as a_aoi_id, a.polygon as a_polygon,
                                    b.code as b_code, b.aoi_id as b_aoi_id, b.polygon as b_polygon
                                    from 
                                    (SELECT code, aoi_id, aoi_polygon as polygon
                                    from mart_peisongpa.aoi_of_city_code_data_daily
                                    where dt={dt}
                                    ) a
                                    FULL OUTER JOIN
                                    (SELECT code, aoi_id, aoi_polygon as polygon
                                    from mart_peisongpa.aoi_of_city_code_data_daily
                                    where dt in (select max(dt) from mart_peisongpa.shadow_h3_release_data where dt < {dt})
                                    ) b
                                    on a.aoi_id = b.aoi_id
                                    ) t
                                    where a_polygon != b_polygon or b_polygon is null or a_polygon is null"""


        print(raw_shadow_sql_query)
        print(aoi_change_sql_query)

        output_table = "mart_peisongpa.shadow_h3_release_data"
        #spark_job.drop_table(output_table)

        self.process(raw_shadow_sql_query, aoi_change_sql_query, output_table, dt)


def process(args):
    """Spark任务接口 """
    parser = argparse.ArgumentParser()
    parser.add_argument('--dt', type=str, default='20250707', required=True)
    args, unknown = parser.parse_known_args()

    spark_job = SparkJob()
    job = DataProcess(spark_job.sc, spark_job.sqlContext)

    dt = args.dt

    job.run_city_county(spark_job, dt)
