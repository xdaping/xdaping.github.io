#!/user/bin/env python
# encoding=utf-8
# @Author: daping
# @Date: 2026/6/26 16:43

# -*- coding: utf-8 -*-

"""
@software: PyCharm
@file: area_optimize_job_navi.py
@time: 2023/7/18 10:06
"""
from pyspark.sql import SparkSession
from pyspark.sql.types import *

from jobs.test_yd.phf_navi_dcm_or.area_optimize_alg_navi_3circle import OptimizeMapPartitionCircle
from jobs.test_yd.phf_navi_dcm_or.area_optimize_alg_navi_distance import OptimizeMapPartitionDistance
from utils import logging_util

logger = logging_util.Logger()


class OptimizeSparkJob(object):
    def __init__(self, spark_session, output_table, dt, version):
        self.output_table = output_table
        self.spark_session = spark_session
        self.dt = dt
        self.version = version
        self.spark_session.sql("SET hive.exec.dynamic.partition.mode=nonstrict")
        self.spark_session.sql("SET spark.sql.sources.partitionOverwriteMode=dynamic")

        self.spark_session.sql("set hive.exec.dynamic.partition=true")
        self.spark_session.sql("set hive.exec.dynamic.partition.mode=nostrick")
        self.spark_session.sql("set spark.rdd.compress=true")
        self.spark_session.sql("set spark.serializer=org.apache.spark.serializer.KryoSerializer")
        self.spark_session.sql("set spark.sql.shuffle.partitions=2000")

        # shuffle时向上游拉取数据，如果遇到慢节点，可能导致任务一直hang住，时间甚至长达几个小时甚至更多，很多情况下kill任务重试后很快
        self.spark_session.sql("set spark.shuffle.maxFetchWaitTime=72000s")
        self.spark_session.sql("set spark.sql.broadcastTimeout=1000")
        self.spark_session.sql("set spark.network.timeout=1000")

    def write_hive(self, output_rdd):
        sql = """
            CREATE TABLE IF NOT EXISTS `{table_name}`
            (
                `da_id` bigint COMMENT '',
                `city_id` bigint COMMENT '',
                `poi_id` string COMMENT '',
                `aoi_id` bigint COMMENT '',
                `base_set` int COMMENT '',
                `candidate_set` int COMMENT '',
                `is_selected` int COMMENT '是否选中'
            )
            COMMENT 'PHF商家范围基于导航距离的OR求解结果'
            PARTITIONED BY (
                `dt` string
            )
            STORED AS ORC
        """.format(table_name=self.output_table)
        self.spark_session.sql(sql)

        schema = StructType([
            StructField("da_id", LongType(), False),
            StructField("city_id", LongType(), False),
            StructField("poi_id", StringType(), False),
            StructField("aoi_id", LongType(), False),
            StructField("base_set", IntegerType(), False),
            StructField("candidate_set", IntegerType(), False),
            StructField("is_selected", IntegerType(), False)])

        logger.info(f"final rdd snapshot:{output_rdd.take(1)}")
        self.spark_session.createDataFrame(output_rdd, schema).registerTempTable("warehouse")
        out_sql = """
                  insert overwrite table {table_name} partition(dt="{version}") 
                  select * from warehouse
                """.format(version=self.version, table_name=self.output_table)
        self.spark_session.sql(out_sql)

    def get_input_sql(self):
        sql = """
        select da_id,
               city_id,
               aoi_id,
               poi_id,
               prediction_map,
               base_set,
               candidate_set,
               navi_cls
          from mart_peisongpa.mnl_for_phf_navi_or
         where version='{dt}'
               -- and city_id in (420500,330400,130600,320509,120100,340100,320582,320583,530100,320900,360400,460100,450100,330600,220100,510700,130100,320100,230100,510100,370100,370800,370700,131000,210100,445100,350100,440500,370600,440400,350300,410100,210200,350500,340300)

        """.format(dt=self.dt)
        return sql

    def process(self, cls):
        input_df = self.spark_session.sql(self.get_input_sql()).persist()

        # 根据da_id，将同一区域数据分发到同一partition，然后对一个区域的数据进行求解
        all_da = input_df.select("city_id").distinct().rdd.flatMap(list).collect()
        da_cnt = len(all_da)
        logger.info(f"所有区域个数为：{da_cnt}个")
        if cls == '_circle':
            result_rdd = input_df \
                .rdd \
                .map(lambda row: (row.city_id, row)) \
                .partitionBy(da_cnt, lambda x: all_da.index(x)) \
                .map(lambda x: x[-1]) \
                .mapPartitions(OptimizeMapPartitionCircle.scip_optimize) \
                .repartition(100)
        else:
            result_rdd = input_df \
                .rdd \
                .map(lambda row: (row.city_id, row)) \
                .partitionBy(da_cnt, lambda x: all_da.index(x)) \
                .map(lambda x: x[-1]) \
                .mapPartitions(OptimizeMapPartitionDistance.scip_optimize) \
                .repartition(100)

        self.write_hive(result_rdd)


def process(args):
    """Spark任务接口 """
    # output_table = args.Doutput_table  # mart_peisongpa.phf_poi_scope_adjust
    output_table = 'mart_peisongpa.phf_poi_scope_navi_adjust'  # mart_peisongpa.phf_poi_scope_adjust
    # dt = args.dt  # "20221130"
    # dt = '20221130_20221031_20221129aoi'
    # dt = '20230331_20230302_20230331aoi'
    dt = '20230712_20230613_20230712_aoi_approx-1_1'
    cls = '_circle'  # circle or distance
    is_cross = True
    version = dt + cls+'_1.0' if is_cross else f"{dt}_without_cross"
    logger.info(f"table is :{output_table}, dt:{dt}, partition is :{version}")
    spark_session = SparkSession.builder.enableHiveSupport().getOrCreate()
    job = OptimizeSparkJob(spark_session, output_table, dt, version)
    job.process(cls)
