#!/user/bin/env python
# encoding=utf-8
# @Author: daping
# @Date: 2024/8/4 18:47

import os
from abc import ABC

import tensorflow as tf
import numpy as np
import time
import logging
from tensorflow.keras import models, layers, regularizers
from tensorflow.keras.layers import Embedding, Dense, Flatten, Concatenate, Dropout
from tensorflow.keras import models, layers, regularizers
from tensorflow.keras.layers import Layer, Input, Dense, Embedding, Concatenate, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from sklearn.metrics import roc_auc_score
import random
import tqdm
from utils import multiplex


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

tf.compat.v1.flags.DEFINE_integer('epochs', 1, "训练轮数")
tf.compat.v1.flags.DEFINE_float('learning_rate', 0.0001, "模型初始学习率")
tf.compat.v1.flags.DEFINE_integer('con_feature_n', 6, "统计类特征数量")
tf.compat.v1.flags.DEFINE_integer('batch_size', 128, "")
tf.compat.v1.flags.DEFINE_float('lr_pow', 0.95, "")
FLAGS = tf.compat.v1.flags.FLAGS

logging.basicConfig(level=logging.INFO)


class FM_layer(layers.Layer):
    def __init__(self, k=10, w_reg=1e-4, v_reg=1e-4):
        super().__init__()
        self.k = k
        self.w_reg = w_reg
        self.v_reg = v_reg

    #build方法在第一次调用层的 call 方法时自动执行，这样可以根据输入数据的形状动态地创建权重。
    def build(self, input_shape):
        self.w0 = self.add_weight(name='w0', shape=(1,),
                                  initializer=tf.zeros_initializer(),
                                  trainable=True,)
        self.w = self.add_weight(name='w', shape=(input_shape[-1], 1),
                                 initializer=tf.random_normal_initializer(),
                                 trainable=True,
                                 regularizer=tf.keras.regularizers.l2(self.w_reg))
        self.v = self.add_weight(name='v', shape=(input_shape[-1], self.k),
                                 initializer=tf.random_normal_initializer(),
                                 trainable=True,
                                 regularizer=tf.keras.regularizers.l2(self.v_reg))

    def call(self, inputs, **kwargs):
        linear_part = tf.matmul(inputs, self.w) + self.w0   #shape:(batchsize, 1)

        inter_part1 = tf.pow(tf.matmul(inputs, self.v), 2)  #shape:(batchsize, self.k)
        inter_part2 = tf.matmul(tf.pow(inputs, 2), tf.pow(self.v, 2)) #shape:(batchsize, self.k)
        inter_part = 0.5*tf.reduce_sum(inter_part1 - inter_part2, axis=-1, keepdims=True) #shape:(batchsize, 1)

        output = linear_part + inter_part
        return output

class Dense_layer(layers.Layer):
    def __init__(self):
        super().__init__()

        self.hidden_layer = [Dense(i, activation='relu') for i in [256, 128, 64]]
        self.dense_0 = layers.Dense(8, kernel_regularizer=regularizers.l2(0.001), name='dense_0', activation="relu")
        self.dense_1 = layers.Dense(1, name='dense_1', activation='softmax')

    def call(self, inputs):
        x = inputs
        for layer in self.hidden_layer:
            x = layer(x)
        x = self.dense_0(x)
        output = self.dense_1(x)
        return output

class ClassifyModel(models.Model):
    tf.random.set_seed(1)  # 设置全局种子

    def __init__(self, feature_columns):
        super().__init__()
        self.dense_feature_columns, self.sparse_feature_columns = feature_columns
        self.embed_layers = {
            'embed_' + str(i): Embedding(feat['feat_num'], 4)
            for i, feat in enumerate(self.sparse_feature_columns)
        }

        self.FM = FM_layer()

        self.DNN = [layers.Dense(unit, activation='relu') for unit in [31,16,1]]
        self.dense = layers.Dense(2, name='dense', activation='softmax')


    @tf.function
    def call(self, inputs):
        sparse_inputs,dense_inputs = inputs
        # embedding

        # 稀疏特征的嵌入向量
        sparse_embeds = [self.embed_layers['embed_' + str(i)](sparse_inputs[:, i]) for i in range(sparse_inputs.shape[1])]
        sparse_embeds = Concatenate(axis=1)(sparse_embeds)

        x = tf.concat([dense_inputs, sparse_embeds], axis=-1)

        fm_output = self.FM(x)

        dnn_output = x
        for layer in self.DNN:
            dnn_output = layer(dnn_output)

        x = tf.nn.sigmoid(0.5 * (fm_output + dnn_output))

        output = self.dense(x)

        return output


def batcher(data, batch_size):
    batch_size = len(data) if batch_size > len(data) else batch_size
    if batch_size == 0:
        return None
    for start_idx in range(0, len(data), batch_size):
        excerpt = slice(start_idx, start_idx + batch_size)
        batch = data[excerpt]

        batch_lable = [int(item[0]) for item in batch]

        spare_data = [[0 if int(i)== -1 else int(i) for i in item[1]] for item in batch]
        dense_data = [[float(i) for i in item[2]] for item in batch]

        batch_data = {'label': batch_lable, 'spare_data': spare_data,'dense_data': dense_data}
        yield batch_data


def process_feature(one_batch):
    label = one_batch['label']
    spare_feature = one_batch['spare_data']
    dense_feature = one_batch['dense_data']
    label = tf.one_hot(tf.convert_to_tensor(np.array(label)), 2)

    spare_feature = tf.convert_to_tensor(spare_feature, dtype=tf.float32)
    dense_feature = tf.convert_to_tensor(dense_feature, dtype=tf.float32)

    return label, spare_feature, dense_feature


def run_model(model, features, label, training=False):
    pred = model(features, training=training)

    loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(label, pred))

    correct_prediction = tf.equal(tf.argmax(pred, axis=1), tf.argmax(label, axis=1))  # tf.argmax找出每一列最大值的索引
    acc = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32))  # tf.cast转化数据类型
    auc = tf.py_function(roc_auc_score, (label, pred), tf.float32)

    return loss, acc, auc


def get_train_vaild_data(train_filename, validation_filename):
    train_data = []
    with open(train_filename) as f:
        for line in f:
            user_id, is_new_user, is_dt_finish, is_own_coupon, user_tag, is_dt_flow_user, is_visit_helppost, is_visit_helppost_intent_order, is_ask_price_user_id, is_day_submit_user_id, gender_label, age, is_student, is_white_collar, edu_level_label, salary_level, waimai_level, daocan_level, daozong_level, maoyan_level, jiulv_level, sensitivity_score, sensitivity_level, first_visit_to_today_days, last_visit_to_today_days, pt_visit_customer_accum_7days, pt_visit_customer_accum_14days, pt_visit_customer_accum_30days, pt_visit_customer_accum_total, most_visit_app_name, most_visit_locate_city_id, pt_first_order_to_today_days, pt_first_arrived_order_to_today_days, pt_last_order_to_today_days, pt_last_arrived_order_to_today_days, pt_cdel_last_arrived_order_to_today_days, pt_cbuy_last_arrived_order_to_today_days, pt_cdel_last_order_to_today_days, pt_cbuy_last_order_to_today_days, pt_order_cnt_total, pt_last_7day_arrived_order_cnt, pt_last_14day_arrived_order_cnt, pt_last_30day_arrived_order_cnt, pt_arrived_order_cnt_total, pt_finish_ratio_total, pt_arrived_total_amt, pt_cbuy_arrived_order_cnt, pt_cdel_arrived_order_cnt, pt_cbuy_finish_ratio_total, pt_cdel_finish_ratio_total, is_high_customer, ds_arrived_order_cnt, ord_num_7days, ord_num_30days, ord_num_90days, ord_amt_7days, ord_amt_30days, ord_amt_90days, pay_amt_7days, pay_amt_30days, pay_amt_90days, act_fee_7days, act_fee_30days, act_fee_90days, mt_charge_fee_7days, mt_charge_fee_30days, mt_charge_fee_90days, peisong_order_7days, peisong_order_30days, peisong_order_90days, ka_order_7days, ka_order_30days, ka_order_90days, category_10_ord_num_90days, category_11_ord_num_90days, category_12_ord_num_90days, category_13_ord_num_90days, category_14_ord_num_90days, category_15_ord_num_90days, category_16_ord_num_90days, category_17_ord_num_90days, category_18_ord_num_90days, terminalc_act_nmd_disc_amt_7day, terminalc_act_nmd_disc_amt_30day, terminalc_act_nmd_disc_amt_90day, ord_num_60days, ord_num_14days, ord_num_21days, tot_user_avg_90, tot_user_avg_180, ord_num_180days, order_num_breakfast_7days, order_num_lunch_7days, order_num_afternoontea_7days, order_num_dinner_7days, order_num_supper_7days, order_num_breakfast_30days, order_num_lunch_30days, order_num_afternoontea_30days, order_num_dinner_30days, order_num_supper_30days, act_fee_180days, ord_amt_180days, peisong_order_180days, ka_order_180days, ord_num_weekdays_180days, count_nofee_order_num_180days, pay_amt_180days, ord_days_30days, wmbu_charge_fee_90days, ord_second_city_id_90days, ord_second_city_id_180days, milktea_ord_num_30days, roast_ord_num_30days, snack_ord_num_30days, ord_num_6days, order_num_breakfast_90days, order_num_lunch_90days, order_num_afternoontea_90days, order_num_dinner_90days, order_num_supper_90days, order_num_breakfast_90days_per, order_num_lunch_90days_per, order_num_afternoontea_90days_per, order_num_dinner_90days_per, order_num_supper_90days_per, ord_num_b30days, tot_user_avg_7, tot_user_avg_30, ord_num_90days_wm, pay_amt_90days_wm, ord_num_30days_wm, pay_amt_30days_wm, ord_num_180days_wm, pay_amt_180days_wm, pt_coupon_ct_30days, pt_use_coupon_ratio_30days, pt_use_coupon_cdel_ratio_30days, pt_use_coupon_cbuy_ratio_30days, pt_coupon_ct_15days, pt_use_coupon_ratio_15days, pt_use_coupon_cdel_ratio_15days, pt_use_coupon_cbuy_ratio_15days, pt_coupon_ct_7days, pt_use_coupon_ratio_7days, pt_use_coupon_cdel_ratio_7days, pt_use_coupon_cbuy_ratio_7days, pt_use_coupon_amount_avg_30days, pt_use_coupon_amount_avg_15days, pt_use_coupon_amount_avg_7days, dt = line.strip().split(
                '\t')
            sparse_feature = [waimai_level, age, most_visit_app_name, sensitivity_level]
            dense_feature = [pt_cdel_last_order_to_today_days, pt_visit_customer_accum_total,
                             terminalc_act_nmd_disc_amt_90day, pay_amt_90days, tot_user_avg_90, pay_amt_180days,
                             first_visit_to_today_days, tot_user_avg_180, order_num_dinner_90days_per,
                             sensitivity_score]

            if is_day_submit_user_id == 'is_day_submit_user_id':
                continue
            train_data.append([is_day_submit_user_id, sparse_feature, dense_feature])

    validation_data = []
    if validation_filename:
        with open(validation_filename) as f:
            for line in f:
                user_id, is_new_user, is_dt_finish, is_own_coupon, user_tag, is_dt_flow_user, is_visit_helppost, is_visit_helppost_intent_order, is_ask_price_user_id, is_day_submit_user_id, gender_label, age, is_student, is_white_collar, edu_level_label, salary_level, waimai_level, daocan_level, daozong_level, maoyan_level, jiulv_level, sensitivity_score, sensitivity_level, first_visit_to_today_days, last_visit_to_today_days, pt_visit_customer_accum_7days, pt_visit_customer_accum_14days, pt_visit_customer_accum_30days, pt_visit_customer_accum_total, most_visit_app_name, most_visit_locate_city_id, pt_first_order_to_today_days, pt_first_arrived_order_to_today_days, pt_last_order_to_today_days, pt_last_arrived_order_to_today_days, pt_cdel_last_arrived_order_to_today_days, pt_cbuy_last_arrived_order_to_today_days, pt_cdel_last_order_to_today_days, pt_cbuy_last_order_to_today_days, pt_order_cnt_total, pt_last_7day_arrived_order_cnt, pt_last_14day_arrived_order_cnt, pt_last_30day_arrived_order_cnt, pt_arrived_order_cnt_total, pt_finish_ratio_total, pt_arrived_total_amt, pt_cbuy_arrived_order_cnt, pt_cdel_arrived_order_cnt, pt_cbuy_finish_ratio_total, pt_cdel_finish_ratio_total, is_high_customer, ds_arrived_order_cnt, ord_num_7days, ord_num_30days, ord_num_90days, ord_amt_7days, ord_amt_30days, ord_amt_90days, pay_amt_7days, pay_amt_30days, pay_amt_90days, act_fee_7days, act_fee_30days, act_fee_90days, mt_charge_fee_7days, mt_charge_fee_30days, mt_charge_fee_90days, peisong_order_7days, peisong_order_30days, peisong_order_90days, ka_order_7days, ka_order_30days, ka_order_90days, category_10_ord_num_90days, category_11_ord_num_90days, category_12_ord_num_90days, category_13_ord_num_90days, category_14_ord_num_90days, category_15_ord_num_90days, category_16_ord_num_90days, category_17_ord_num_90days, category_18_ord_num_90days, terminalc_act_nmd_disc_amt_7day, terminalc_act_nmd_disc_amt_30day, terminalc_act_nmd_disc_amt_90day, ord_num_60days, ord_num_14days, ord_num_21days, tot_user_avg_90, tot_user_avg_180, ord_num_180days, order_num_breakfast_7days, order_num_lunch_7days, order_num_afternoontea_7days, order_num_dinner_7days, order_num_supper_7days, order_num_breakfast_30days, order_num_lunch_30days, order_num_afternoontea_30days, order_num_dinner_30days, order_num_supper_30days, act_fee_180days, ord_amt_180days, peisong_order_180days, ka_order_180days, ord_num_weekdays_180days, count_nofee_order_num_180days, pay_amt_180days, ord_days_30days, wmbu_charge_fee_90days, ord_second_city_id_90days, ord_second_city_id_180days, milktea_ord_num_30days, roast_ord_num_30days, snack_ord_num_30days, ord_num_6days, order_num_breakfast_90days, order_num_lunch_90days, order_num_afternoontea_90days, order_num_dinner_90days, order_num_supper_90days, order_num_breakfast_90days_per, order_num_lunch_90days_per, order_num_afternoontea_90days_per, order_num_dinner_90days_per, order_num_supper_90days_per, ord_num_b30days, tot_user_avg_7, tot_user_avg_30, ord_num_90days_wm, pay_amt_90days_wm, ord_num_30days_wm, pay_amt_30days_wm, ord_num_180days_wm, pay_amt_180days_wm, pt_coupon_ct_30days, pt_use_coupon_ratio_30days, pt_use_coupon_cdel_ratio_30days, pt_use_coupon_cbuy_ratio_30days, pt_coupon_ct_15days, pt_use_coupon_ratio_15days, pt_use_coupon_cdel_ratio_15days, pt_use_coupon_cbuy_ratio_15days, pt_coupon_ct_7days, pt_use_coupon_ratio_7days, pt_use_coupon_cdel_ratio_7days, pt_use_coupon_cbuy_ratio_7days, pt_use_coupon_amount_avg_30days, pt_use_coupon_amount_avg_15days, pt_use_coupon_amount_avg_7days, dt = line.strip().split(
                    '\t')

                sparse_feature = [waimai_level, age, most_visit_app_name, sensitivity_level]
                dense_feature = [pt_cdel_last_order_to_today_days,pt_visit_customer_accum_total,terminalc_act_nmd_disc_amt_90day,pay_amt_90days,tot_user_avg_90,pay_amt_180days,first_visit_to_today_days,tot_user_avg_180,order_num_dinner_90days_per,sensitivity_score]

                if is_day_submit_user_id == 'is_day_submit_user_id':
                    continue
                validation_data.append([is_day_submit_user_id, sparse_feature, dense_feature])

    random.shuffle(train_data)
    random.shuffle(validation_data)
    return train_data, validation_data


def train_valid(model, dataset, epoch, summary_writer, optimizer, training):
    global global_step
    tv_step = 0
    tv_loss = 0.0
    tv_acc = 0.0
    tv_auc = 0.0
    for tv_step, one_batch in enumerate(batcher(dataset, FLAGS.batch_size)):

        label, sparse_features, dense_features = process_feature(one_batch)

        #print(con_features)

        if training:
            global_step += 1
            with tf.GradientTape() as tape:
                loss, acc, auc = run_model(model, [sparse_features, dense_features], label, training=training)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        else:
            loss, acc, auc = run_model(model, [sparse_features, dense_features], label)

        tv_loss += loss.numpy()
        tv_acc += acc.numpy()
        tv_auc += auc.numpy()

        # if tv_step % 5 == 0:
        #    print(tv_step, tv_loss / (tv_step + 1), tv_acc / (tv_step + 1))

        # 训练过程中每个n个step输出并保存结果到tensorboard中
        if training:
            if tv_step % 1 == 0:
                logging.info("step:{}, train_loss:{},train_acc:{}, train_auc:{}".format(tv_step,
                                                                                        tv_loss / (tv_step + 1),
                                                                                        tv_acc / (tv_step + 1),
                                                                                        tv_auc / (tv_step + 1)))

                with summary_writer.as_default():
                    tf.summary.scalar('train/train_loss', loss, step=global_step)
                    tf.summary.scalar('train/train_acc', acc, step=global_step)
                    tf.summary.scalar('train/train_auc', auc, step=global_step)
                    tf.summary.scalar('train/learning_rate', optimizer.learning_rate, step=global_step)

    # 训练时学习率随epoch数量动态减小
    if training:
        optimizer.learning_rate.assign(optimizer.learning_rate * FLAGS.lr_pow)

    # 输出每个epoch的训练或验证效果
    avg_loss = tv_loss / (tv_step + 1)
    avg_acc = tv_acc / (tv_step + 1)
    avg_auc = tv_auc / (tv_step + 1)

    flag = 'train' if training else 'vaild'
    with summary_writer.as_default():
        tf.summary.scalar(flag + '/avg_loss', avg_loss, step=epoch)
        tf.summary.scalar(flag + '/avg_acc', avg_acc, step=epoch)
        tf.summary.scalar(flag + '/avg_auc', avg_auc, step=epoch)

    logging.info("【{}】 {}, epoch={}, learning_rate={}, avg_loss={}, avg_acc={}, avg_auc={}".format(flag,
                                                                                                   time.strftime(
                                                                                                       '%Y-%m-%d %H:%M:%S',
                                                                                                       time.localtime(
                                                                                                           time.time())),
                                                                                                   epoch.numpy(),
                                                                                                   optimizer.lr.numpy(),
                                                                                                   avg_loss, avg_acc,
                                                                                                   avg_auc))


def train_model(model, train_data, validation_data, epochs, output_folder):

    result_folder = os.path.join(os.getcwd(), output_folder)
    train_log_dir = os.path.join(result_folder, 'logs')
    if not os.path.exists(train_log_dir):
        os.makedirs(train_log_dir)
    summary_writer = tf.summary.create_file_writer(train_log_dir)

    # model.l_bert.apply_adapter_freeze()

    optimizer = tf.keras.optimizers.Adam(learning_rate=FLAGS.learning_rate)

    for epoch in tf.range(1, epochs + 1, dtype=tf.int64):
        train_valid(model, train_data, epoch, summary_writer, optimizer, training=True)
        train_valid(model, validation_data, epoch, summary_writer, optimizer, training=False)

        model_dir = result_folder + f'/epoch_{epoch}'
        model.save(model_dir, save_format="tf")
        logging.info('export saved model: {}'.format(model_dir))


def model_train(train_filename, validation_filename, output_folder):

    tf.keras.backend.clear_session()

    dense_feature_columns = []
    sparse_feature_columns = [{'feat_num': 5},
                              {'feat_num': 100},
                              {'feat_num': 7},
                              {'feat_num': 5}]

    feature_columns = (dense_feature_columns, sparse_feature_columns)
    hidden_units = [64, 32]  # 深度网络的隐藏单元

    # 创建 DCN 模型实例
    model = ClassifyModel(feature_columns)

    train_data, validation_data = get_train_vaild_data(train_filename, validation_filename)
    train_model(model, train_data, validation_data, FLAGS.epochs, output_folder)


def model_test(predict_filename: object, model_file: object) -> object:
    batch_size = 128

    model_loaded = tf.keras.models.load_model(model_file)
    print(model_loaded.summary())

    test_dataset, _ = get_train_vaild_data(predict_filename, predict_filename)

    # 打印模型参数变量值
    for layer in model_loaded.layers:
        for weight in layer.weights:
            print(weight.name, weight.numpy())

    for t_step, one_batch in enumerate(batcher(test_dataset, batch_size)):
        if t_step >= 3:
            break
        label, con_features = process_feature(one_batch)

        pred = model_loaded([con_features, con_features], training=False)

        print(pred)


global_step = 0  # 全局变量，训练step总数量
if __name__ == "__main__":
    train_filename = multiplex.data_path + 'test.txt'
    validation_filename = multiplex.data_path + 'test.txt'
    output_folder = 'model_v1'

    model_train(train_filename, validation_filename, output_folder)

    #model_test(validation_filename, output_folder+"/epoch_1")