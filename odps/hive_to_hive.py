
# -*- coding: utf-8 -*-
from pyspark.sql import SparkSession

if __name__ == '__main__':
    spark = SparkSession.builder \
        .appName("hive_to_hive_demo") \
        .config("spark.sql.broadcastTimeout", 20 * 60) \
        .config("spark.sql.crossJoin.enabled", True) \
        .config("spark.hadoop.odps.task.wlm.quota", "os_8aaef08ca3e94f828dd099a4c7e1c57d") \
        .config("odps.exec.dynamic.partition.mode", "nonstrict") \
        .config("spark.sql.catalogImplementation", "odps") \
        .getOrCreate()

    # 1. 读取输入表（假设按天分区，pt 通过调度参数传入）
    src_df = spark.sql("""
        SELECT aoi_id, polygon
        from swiftx_mining.tmp_aoi_level_2
    """)

    # # 2. 做一些业务处理，例如按用户汇总金额
    # result_df = src_df.groupBy("user_id").sum("amount") \
    #     .withColumnRenamed("sum(amount)", "total_amount")

    src_df.createOrReplaceTempView("tmp_result")

    # 3. 写入输出表（覆盖对应分区）
    spark.sql("""
        CREATE TABLE  swiftx_mining.tmp_aoi_level_2_copy as
        SELECT aoi_id, polygon FROM tmp_result
    """)

    print("finish")

    spark.stop()