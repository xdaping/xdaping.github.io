# encoding=utf-8
import time
import shapely.wkt as wkt
from rtree import index
from pyspark.sql.types import *
from jobs.spark_job import SparkJob
from utils.base_util import BaseUtil
from shapely.geometry import Polygon, LineString, Point
from shapely.ops import transform
import pyproj

class AOIRelation(object):

    def __init__(self, sc, sql_context):
        self.sc = sc
        self.sqlContext = sql_context

    def data_to_hive(self, results, output_table, ver):
        print('************data_to_hive **************', output_table, ver)
        print('type bill_time:', type(results))

        fields = [
            StructField("bm_aoi_id", IntegerType(), True),
            StructField("near_aoi_id", IntegerType(), True),
            StructField("dist_degree", DoubleType(), True),
            StructField("dist_meter", DoubleType(), True),
            StructField("inter_ratio", DoubleType(), True),
            StructField("near_inter_ratio", DoubleType(), True),
            StructField("pass_aoi_cnt", IntegerType(), True),
            StructField("pass_aoi_list", ArrayType(IntegerType()), True)
        ]

        dataf = self.sqlContext.createDataFrame(results, schema=StructType(fields))
        print(dataf.take(1))
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
                        `bm_aoi_id`             bigint         COMMENT '',
                        `near_aoi_id`           bigint         COMMENT '',
                        `dist_degree`           double         COMMENT '',
                        `dist_meter`            double         COMMENT '',
                        `inter_ratio`           double         COMMENT '',  
                        `near_inter_ratio`      double         COMMENT '',
                        `pass_aoi_cnt`          bigint         COMMENT '',
                        `pass_aoi_list`         array<bigint>     COMMENT ''      
                    ) COMMENT ''
                    PARTITIONED BY (dt string)
                    ROW FORMAT DELIMITED FIELDS TERMINATED BY '\t'
                    STORED AS ORC
                    """)

        self.sqlContext.sql(
            f"""insert  overwrite table {output_table} partition(dt='{ver}')
                    select * from tmp""")
        return

    def process(self, aoi_sql_query, output_table, dt):
        print(f"*****任务开始*****")
        start = time.time()

        def get_rtree_of_aoi(b_aoi_list):
            #bm_aoi_id, polygon
            rtree_index = index.Index()
            rtree_dict = {}
            for row in b_aoi_list:
                aoi_id = int(row[0])
                polygon = wkt.loads(row[1])
                rtree_dict[aoi_id] = polygon
                rtree_index.insert(aoi_id, polygon.bounds)
            return rtree_index, rtree_dict

        def get_aoi_relation(itr):
            project = pyproj.Transformer.from_crs(pyproj.CRS('EPSG:4326'), pyproj.CRS('EPSG:32618'),
                                                  always_xy=True).transform
            b_aoi_list = brd_aoi_list.value
            rtree_index, polygon_dict = get_rtree_of_aoi(b_aoi_list)

            link_to_city_list = []
            for row in itr:
                aoi_id, polygon = row[0], wkt.loads(row[1])

                center_point = polygon.centroid

                #使用中心点外扩buffer召回
                center_b = BaseUtil.buffer_m(polygon.centroid, 2000)

                # 使用AOI外扩buffer召回
                #polygon_b = BaseUtil.buffer_m(polygon, 2000)

                match_aois = set(rtree_index.intersection(center_b.bounds))

                for m_aoi_id in match_aois:
                    # print(polygon_dict[aoi])
                    m_polygon = polygon_dict[m_aoi_id]
                    m_center_point = m_polygon.centroid

                    if aoi_id == m_aoi_id or not center_b.intersects(m_polygon):
                        continue

                    dist = polygon.distance(m_polygon)

                    inter_ratio = polygon.intersection(m_polygon).area / polygon.area
                    m_inter_ratio = polygon.intersection(m_polygon).area / m_polygon.area

                    dist_t = transform(project, polygon).distance(transform(project, m_polygon))

                    if dist_t <= 1000:
                        c_point_line = LineString([polygon.centroid, m_polygon.centroid])
                        match_aois_l = set(rtree_index.intersection(c_point_line.bounds))
                        l_cnt = 0
                        l_list = []
                        for m_aoi_id_l in match_aois_l:
                            if m_aoi_id_l == aoi_id or m_aoi_id_l == m_aoi_id:
                                continue
                            m_polygon_l = polygon_dict[m_aoi_id_l]
                            if c_point_line.intersects(m_polygon_l) and not m_polygon_l.contains(center_point)\
                                    and not m_polygon_l.contains(m_center_point):
                                l_cnt += 1
                                l_list.append(m_aoi_id_l)
                        link_to_city_list.append([aoi_id, m_aoi_id, dist, dist_t, inter_ratio, m_inter_ratio, l_cnt, l_list])

            return link_to_city_list


        #
        aoi_rdd = self.sqlContext.sql(aoi_sql_query).rdd.map(lambda row: [row.bm_aoi_id, row.polygon]).cache()
        aoi_list = aoi_rdd.collect()
        brd_aoi_list = self.sc.broadcast(aoi_list)

        aoi_relation_rdd = aoi_rdd.repartition(500).mapPartitions(get_aoi_relation)

        self.data_to_hive(aoi_relation_rdd, output_table, dt)

        end = time.time()
        print("end - start", end - start)

    def run_city_county(self, spark_job, dt):

        aoi_sql_query = f"""SELECT bm_aoi_id, polygon
                              FROM mart_peisonglbs.bm_aoi_release_data
                             WHERE dt = '{dt}'"""

        #spark_job.drop_table("mart_peisongpa.bm_aoi_relation")
        output_table = "mart_peisongpa.bm_aoi_topology_relation"

        self.process(aoi_sql_query, output_table, dt)




def process(args):
    """Spark任务接口"""

    spark_job = SparkJob()
    job = AOIRelation(spark_job.sc, spark_job.sqlContext)

    dt = '20221010'
    job.run_city_county(spark_job, dt)








