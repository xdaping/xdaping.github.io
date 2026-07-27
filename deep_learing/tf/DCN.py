#!/user/bin/env python
# encoding=utf-8
# @Author: daping
# @Date: 2024/8/4 18:46

import os
from abc import ABC

import tensorflow as tf
import numpy as np
import time
import logging
from tensorflow.keras import models, layers, regularizers
from tensorflow.keras.layers import Layer, Input, Dense, Embedding, Concatenate, Flatten
from sklearn.metrics import roc_auc_score
import random
from utils import multiplex

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

tf.compat.v1.flags.DEFINE_integer('epochs', 1, "训练轮数")
tf.compat.v1.flags.DEFINE_float('learning_rate', 0.0001, "模型初始学习率")
tf.compat.v1.flags.DEFINE_integer('con_feature_n', 6, "统计类特征数量")
tf.compat.v1.flags.DEFINE_integer('batch_size', 128, "")
tf.compat.v1.flags.DEFINE_float('lr_pow', 0.95, "")
FLAGS = tf.compat.v1.flags.FLAGS

logging.basicConfig(level=logging.INFO)


class CrossNetwork(Layer):
    def __init__(self, layer_num, **kwargs):
        super(CrossNetwork, self).__init__(**kwargs)
        self.layer_num = layer_num

    def build(self, input_shape):
        self.cross_weights = [
            self.add_weight(name='w_cross_{}'.format(i),
                            shape=(input_shape[-1], 1),
                            initializer='random_normal',
                            trainable=True)
            for i in range(self.layer_num)
        ]
        self.cross_biases = [
            self.add_weight(name='b_cross_{}'.format(i),
                            shape=(input_shape[-1], 1),
                            initializer='zeros',
                            trainable=True)
            for i in range(self.layer_num)
        ]
        super(CrossNetwork, self).build(input_shape)

    def call(self, inputs, **kwargs):
        x_0 = tf.expand_dims(inputs, axis=2)  # [batch_size, input_dim, 1]
        x_l = x_0
        for i in range(self.layer_num):
            xl_w = tf.tensordot(x_l, self.cross_weights[i], axes=[1, 0])  # [batch_size, 1, 1]
            dot_ = tf.matmul(x_0, xl_w)  # [batch_size, input_dim, 1]
            x_l = dot_ + self.cross_biases[i] + x_l  # [batch_size, input_dim, 1]
        x_l = tf.squeeze(x_l, axis=2)  # [batch_size, input_dim]
        return x_l

    def get_config(self):
        config = super(CrossNetwork, self).get_config()
        config.update({'layer_num': self.layer_num})
        return config


class ClassifyModel(models.Model):
    def __init__(self, feature_columns, cross_num, hidden_units):
        super(ClassifyModel, self).__init__()
        self.dense_feature_columns, self.sparse_feature_columns = feature_columns

        # 嵌入层
        self.embedding_layers = {
            'embed_' + str(i): Embedding(input_dim=feat['feat_num'],output_dim=4)
             for i, feat in enumerate(self.sparse_feature_columns)
        }

        # 交叉网络
        self.cross_network = CrossNetwork(cross_num)

        # 深度网络
        self.deep_network = [Dense(units=unit, activation='relu') for unit in hidden_units]

        # 输出层
        #self.output_layer = Dense(units=output_dim, activation='sigmoid')
        self.output_layer = Dense(2, name='dense_3', activation='softmax')

    def call(self, inputs, training=None):
        sparse_inputs, dense_inputs = inputs
        embeds = [self.embedding_layers['embed_' + str(i)](sparse_inputs[:, i]) for i in range(sparse_inputs.shape[1])]
        embeds = Concatenate(axis=1)(embeds)

        x = Concatenate(axis=1)([dense_inputs, embeds])
        #print(embeds.shape, x.shape)  # (64, 16) (64, 26)

        # 交叉网络
        x_cross = self.cross_network(x)

        # 深度网络
        x_deep = x
        for layer in self.deep_network:
            x_deep = layer(x_deep)

        # 合并两个网络的输出
        x = Concatenate(axis=1)([x_cross, x_deep])
        output = self.output_layer(x)
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

    # 创建 DCN 模型实例
    cross_num = 3  # 交叉网络的层数
    hidden_units = [64, 32]  # 深度网络的隐藏单元
    model = ClassifyModel(feature_columns, cross_num, hidden_units)

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