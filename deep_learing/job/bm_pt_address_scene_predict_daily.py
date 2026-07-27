# encoding=utf-8
from jobs.spark_job import SparkJob
import os
import tqdm
import argparse
from transformers import BertTokenizer, TFBertModel
import tensorflow as tf
import numpy as np
import tensorflow as tf
from pyspark.sql.types import *
from pyspark import SparkFiles
keras = tf.keras


class AddressScene(object):
    def __init__(self, sc, sql_context, model_file, resource_file):
        self.sc = sc
        self.sqlContext = sql_context
        self.sc.addFile(model_file, recursive=True)
        self.sc.addFile(resource_file, recursive=True)

    def data_to_hive(self, results, output_table, ver):
        print('************data_to_hive **************', output_table, ver)
        print('type:', type(results))

        fields = [
            StructField("user_id", LongType(), True),
            StructField("platform_order_id", StringType(), True),
            StructField("sender_address", StringType(), True),
            StructField("sender_scene_id", IntegerType(), True),
            StructField("sender_scene_name", StringType(), True),
            StructField("sender_scene_pro", DoubleType(), True),
            StructField("sender_scene_pro_list", ArrayType(DoubleType()), True),

            StructField("recipient_address", StringType(), True),
            StructField("recipient_scene_id", IntegerType(), True),
            StructField("recipient_scene_name", StringType(), True),
            StructField("recipient_scene_pro", DoubleType(), True),
            StructField("recipient_scene_pro_list", ArrayType(DoubleType()), True),
            StructField("order_identity", StringType(), True),

        ]

        dataf = self.sqlContext.createDataFrame(results, schema=StructType(fields))
        print(dataf.take(1))
        print()
        dataf.registerTempTable("tmp")
        self.sqlContext.sql("set hive.exec.dynamic.partition=true")
        self.sqlContext.sql("set hive.exec.parallel = true")
        self.sqlContext.sql("set hive.exec.dynamic.partition.mode = nonstrict")
        self.sqlContext.sql("set spark.sql.shuffle.partitions = 2000")
        # self.sqlContext.sql(f"""DROP TABLE IF EXISTS {output_table}""")

        self.sqlContext.sql(
            f"""
            CREATE TABLE IF NOT EXISTS {output_table}
            (
                `user_id`                  bigint        COMMENT '',
                `platform_order_id`        string        COMMENT '',
                `sender_address`           string        COMMENT '',
                `sender_scene_id`          bigint        COMMENT '',
                `sender_scene_name`        string        COMMENT '',
                `sender_scene_pro`         double        COMMENT '',
                `sender_scene_pro_list`    array<double> COMMENT '',
                `recipient_address`        string        COMMENT '',
                `recipient_scene_id`       bigint        COMMENT '',
                `recipient_scene_name`     string        COMMENT '',
                `recipient_scene_pro`      double        COMMENT '',
                `recipient_scene_pro_list` array<double> COMMENT '',
                `order_identity`           string        COMMENT ''       
            ) COMMENT ''
            PARTITIONED BY (dt string)
            ROW FORMAT DELIMITED FIELDS TERMINATED BY '\t'
            STORED AS ORC
            """)

        self.sqlContext.sql(
            f"""
            insert  overwrite table {output_table} partition(dt='{ver}')
            select *
            from tmp
            """)
        return

    def process_for_predict(self, output_table, dt, model_name, resource_name):

        def get_features(row):
            """
            user_id, platform_order_id, sender_address, recipient_address, order_identity
            """
            continuous_feature = [float(i) for i in row.con_feature.split(',')]
            label = 1 if int(row.label) > 0 else 0
            return [row.bm_waybill_id, label, row.nlp_feature, row.longitude, row.latitude, continuous_feature,
                    row.aoi_type, row.bm_city_id]

        def process_feature(one_batch, tokenizer, max_seq_len):

            #dt, user_id, platform_order_id, sender_address, recipient_address

            dt = one_batch['dt']
            user_id = one_batch['user_id']
            platform_order_id = one_batch['platform_order_id']

            sender_address = one_batch['sender_address']
            recipient_address = one_batch['recipient_address']
            order_identity = one_batch['order_identity']

            sender_features = tokenizer(sender_address, padding='max_length', max_length=max_seq_len, truncation=True, return_tensors="tf")
            recipient_features = tokenizer(recipient_address, padding='max_length', max_length=max_seq_len, truncation=True, return_tensors="tf")

            return dt, user_id, platform_order_id,sender_address,recipient_address,sender_features,recipient_features,order_identity

        def batcher(data, batch_size):
            batch_size = len(data) if batch_size > len(data) else batch_size
            if batch_size == 0:
                return -1
            for start_idx in range(0, len(data), batch_size):
                excerpt = slice(start_idx, start_idx + batch_size)
                batch = data[excerpt]

                dt = [str(item[0]) for item in batch]
                user_id = [int(item[1]) for item in batch]
                platform_order_id = [str(item[2]) for item in batch]
                sender_address = [str(item[3]) for item in batch]
                recipient_address = [str(item[4]) for item in batch]
                order_identity = [str(item[5]) for item in batch]

                batch_data = {'dt': dt, 'user_id': user_id, 'platform_order_id': platform_order_id,
                              'sender_address': sender_address, 'recipient_address': recipient_address,
                              'order_identity': order_identity}
                yield batch_data

        def predict_part_data(rows):

            setting_batch_size = 32
            max_seq_len = 50
            scene_name_dict = {0: '零售商铺', 1: '服务商铺', 2: '餐饮商铺', 3: '个人住宅', 4: '物品递送相关场所', 5: '医疗相关场所',
             6: '办公相关场所', 7: '娱乐运动场所', 8: '教育相关场所', 9: '公共服务相关场所', 10: '住宿相关场所',
             11: '无明确归类'}

            model_file = SparkFiles.get(model_name)
            model_loaded = tf.keras.models.load_model(model_file)

            resource_file = SparkFiles.get(resource_name)
            tokenizer = BertTokenizer.from_pretrained(resource_file)

            rows = [row for row in rows]
            for batch in batcher(rows, setting_batch_size):

                dt, user_id, platform_order_id,sender_address,recipient_address,sender_features,recipient_features,order_identity = process_feature(batch, tokenizer, max_seq_len)

                sender_feature_list = [sender_features["input_ids"], sender_features["token_type_ids"], sender_features["attention_mask"]]
                recipient_feature_list = [recipient_features["input_ids"], recipient_features["token_type_ids"], recipient_features["attention_mask"]]
                sender_pred = model_loaded(sender_feature_list, training=False)
                recipient_pred = model_loaded(recipient_feature_list, training=False)

                batch_size, _ = sender_pred.shape

                sender_pred_scene_id = tf.argmax(sender_pred, axis=1).numpy()
                sender_pred_scene_pro = tf.reduce_max(sender_pred, axis=1).numpy()
                sender_pred = sender_pred.numpy()

                recipient_pred_scene_id = tf.argmax(recipient_pred, axis=1).numpy()
                recipient_pred_scene_pro = tf.reduce_max(recipient_pred, axis=1).numpy()
                recipient_pred = recipient_pred.numpy()

                for i in range(batch_size):
                    yield [user_id[i], platform_order_id[i],
                           sender_address[i], int(sender_pred_scene_id[i]), scene_name_dict[int(sender_pred_scene_id[i])],
                           round(float(sender_pred_scene_pro[i]),4), [round(float(v), 4) for v in sender_pred[i]],
                           recipient_address[i], int(recipient_pred_scene_id[i]), scene_name_dict[int(recipient_pred_scene_id[i])],
                           round(float(recipient_pred_scene_pro[i]),4), [round(float(v), 4) for v in recipient_pred[i]],
                           order_identity[i]
                           ]

        sql_query = f"""SELECT dt, user_id, platform_order_id, sender_address, recipient_address,
                                Case
                                WHEN binded_phone!=sender_phone and binded_phone=recipient_phone then '收件人下单'
                                WHEN binded_phone=sender_phone and binded_phone!=recipient_phone then '发件人下单'
                                WHEN binded_phone=sender_phone and binded_phone=recipient_phone then '自发自收下单'
                                WHEN binded_phone!=sender_phone and binded_phone!=recipient_phone then '遥控下单'
                                else '其他'
                                end as order_identity 
                        from app_peisong.app_crowds_legwork_customer_waybill_day
                        where dt='{dt}'
                        and business_type =1
                        and waybill_status=50"""


        print(sql_query)

        process_data_train = self.sqlContext.sql(sql_query).rdd
        print('raw data:', process_data_train.take(3))
        #feature_data_train = process_data_train.repartition(1000).map(get_features)

        rdd_data = process_data_train.repartition(30).mapPartitions(predict_part_data)
        print(rdd_data.take(1))

        self.data_to_hive(rdd_data, output_table, dt)


def process(args):
    """Spark任务接口 """

    parser = argparse.ArgumentParser()
    parser.add_argument('--dt', type=str, default='20240101', required=True)
    parser.add_argument('--model_name', type=str, default='epoch_11_100train', required=True)
    #parser.add_argument('-Doutput_table', type=str, default='mart_peisongpa.bm_pt_address_scene_predict_data')
    args, unknown = parser.parse_known_args()

    spark_job = SparkJob()

    #output_table = args.Doutput_table
    output_table = "mart_peisongpa.bm_pt_address_scene_predict_daily"
    dt= args.dt
    model_name = args.model_name

    #spark_job.drop_table(output_table)
    #model_name = "epoch_11_100train"  # "epoch_3_70train"


    model_file = 'viewfs:///user/hadoop-peisongpa/xiongdaping/pt_model/'+model_name
    resource_name = "bert-base-chinese"
    resource_file = "viewfs:///user/hadoop-peisongpa/xiongdaping/pt_model/"+resource_name

    job = AddressScene(spark_job.sc, spark_job.sqlContext, model_file, resource_file)
    job.process_for_predict(output_table, dt, model_name, resource_name)



