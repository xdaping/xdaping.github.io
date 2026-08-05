
# -*- coding: utf-8 -*-
from pyspark.sql import SparkSession


"""
环境 Spark 3.x

配置
spark.executor.instances: 4
spark.executor.memory: 8g
spark.driver.memory: 8g
spark.hadoop.odps.kube.mode: true
spark.hadoop.odps.sql.allow.namespace.schema: true
spark.hadoop.odps.namespace.schema: true
spark.hadoop.odps.project.name: maxcompute_default_project
spark.sql.catalog.odps: org.apache.spark.sql.execution.datasources.v2.odps.OdpsTableCatalog
spark.sql.extensions: org.apache.spark.sql.execution.datasources.v2.odps.extension.OdpsExtensions
spark.sql.catalog.odps.enableNamespaceSchema: true
"""



if __name__ == '__main__':
    spark = SparkSession.builder \
        .appName("hive_to_hive_demo") \
        .config("spark.sql.broadcastTimeout", 20 * 60) \
        .config("spark.sql.crossJoin.enabled", True) \
        .config("odps.exec.dynamic.partition.mode", "nonstrict") \
        .config("spark.hadoop.odps.kube.mode", "true") \
        .config("spark.sql.catalogImplementation", "odps") \
        .getOrCreate()

    # 1. 读取输入表（假设按天分区，pt 通过调度参数传入）
    src_df = spark.sql("""SELECT aoi_id, polygon
        from swiftx_mining.tmp_aoi_level_2""")

    # # 2. 做一些业务处理，例如按用户汇总金额
    # result_df = src_df.groupBy("user_id").sum("amount") \
    #     .withColumnRenamed("sum(amount)", "total_amount")

    src_df.createOrReplaceTempView("tmp_result")

    # 3. 写入输出表（覆盖对应分区）
    spark.sql("""CREATE TABLE  swiftx_mining.tmp_aoi_level_2_copy as
        SELECT aoi_id, polygon FROM tmp_result""")

    print("finish")

    spark.stop()