#!/user/bin/env python
# encoding=utf-8
# @Author: daping
# @Date: 2024/8/4 18:48

import logging
import os
import random
import time

import tensorflow as tf
from causalml.inference.tf.utils import EpsilonLayer
from tensorflow.keras import models
from tensorflow.keras.layers import Dense, Concatenate
from tensorflow.keras.regularizers import l2
from utils import multiplex

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

tf.compat.v1.flags.DEFINE_integer('epochs', 1, "训练轮数")
tf.compat.v1.flags.DEFINE_float('learning_rate', 0.0001, "模型初始学习率")
tf.compat.v1.flags.DEFINE_integer('con_feature_n', 23, "统计类特征数量")
tf.compat.v1.flags.DEFINE_integer('batch_size', 128, "")
tf.compat.v1.flags.DEFINE_float('lr_pow', 0.95, "")
FLAGS = tf.compat.v1.flags.FLAGS

logging.basicConfig(level=logging.INFO)


class ClassifyModel(models.Model):
    tf.random.set_seed(1)  # 设置全局种子

    def __init__(self):
        super(ClassifyModel, self).__init__()

        self.neurons_per_layer = 128
        self.reg_l2 = 0.01

        self.t_dense1 = Dense(units=self.neurons_per_layer,activation="elu",kernel_initializer="RandomNormal",)
        self.t_dense2 = Dense(units=self.neurons_per_layer,activation="elu",kernel_initializer="RandomNormal",)
        self.t_dense3 = Dense(units=self.neurons_per_layer,activation="elu", kernel_initializer="RandomNormal",)
        self.t_dense4 = Dense(units=1, activation="sigmoid")
        self.dl = EpsilonLayer()

        self.y0_dense1 = Dense(units=int(self.neurons_per_layer / 2),activation="elu",kernel_regularizer=l2(self.reg_l2),)
        self.y0_dense2 = Dense(units=int(self.neurons_per_layer / 2), activation="elu",kernel_regularizer=l2(self.reg_l2), )
        self.y0_dense3 = Dense(units=1,activation=None,kernel_regularizer=l2(self.reg_l2),name="y0_predictions",)

        self.y1_dense1 = Dense(units=int(self.neurons_per_layer / 2), activation="elu",kernel_regularizer=l2(self.reg_l2),)
        self.y1_dense2 = Dense(units=int(self.neurons_per_layer / 2), activation="elu", kernel_regularizer=l2(self.reg_l2), )
        self.y1_dense3 = Dense(units=1, activation=None, kernel_regularizer=l2(self.reg_l2), name="y0_predictions", )

    def call(self, inputs):
        # representation
        x = self.t_dense1(inputs)
        x = self.t_dense2(x)
        x = self.t_dense3(x)
        t_predictions = self.t_dense4(x)

        # HYPOTHESIS
        y0_hidden = self.y0_dense1(x)
        y1_hidden = self.y1_dense1(x)

        # second layer
        y0_hidden = self.y0_dense2(y0_hidden)
        y1_hidden = self.y1_dense2(y1_hidden)

        # third
        y0_predictions = self.y0_dense3(y0_hidden)
        y1_predictions = self.y1_dense3(y1_hidden)

        epsilons = self.dl(t_predictions, name="epsilon")

        concat_predict = Concatenate(1)([y0_predictions, y1_predictions, t_predictions, epsilons])

        return concat_predict


def batcher(data, batch_size):
    batch_size = len(data) if batch_size > len(data) else batch_size
    if batch_size == 0:
        return None
    for start_idx in range(0, len(data), batch_size):
        excerpt = slice(start_idx, start_idx + batch_size)
        batch = data[excerpt]

        y_true = [[float(i) for i in item[0]] for item in batch]

        continue_data = [[float(i) for i in item[1]] for item in batch]

        batch_data = {'y_true': y_true, 'continue_data': continue_data}
        yield batch_data


def process_feature(one_batch):
    y_true = one_batch['y_true']
    con_feature = one_batch['continue_data']
    y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)
    con_feature = tf.convert_to_tensor(con_feature, dtype=tf.float32)

    return y_true, con_feature


def run_model(model, features, y_true, training=False):
    y_pred = model(features, training=training)

    treatment = y_true[:, 0]
    y_factual = y_true[:, 1]

    # y_pred 包含 [y0_hat, y1_hat, t_hat, epsilons]

    y0_hat = y_pred[:, 0]
    y1_hat = y_pred[:, 1]
    t_hat = y_pred[:, 2]
    epsilons = y_pred[:, 3]

    # 计算二元分类交叉熵损失
    treatment_loss = tf.keras.losses.binary_crossentropy(treatment, t_hat)
    # 计算回归损失（均方误差）
    y_hat = treatment * y1_hat + (1 - treatment) * y0_hat
    outcome_loss = tf.keras.losses.mean_squared_error(y_factual, y_hat)
    # 主损失

    # 计算 epsilon 的正则化项（例如 L2 正则化）
    epsilon_regularization = tf.reduce_mean(tf.square(epsilons))

    total_loss = treatment_loss + outcome_loss + 0.1*epsilon_regularization

    # 计算准确率
    correct_prediction = tf.equal(tf.round(t_hat), treatment)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32))

    #print('t_hat',t_hat)
    #print('treatment', treatment)

    auc_metric = tf.keras.metrics.AUC()
    auc_metric.update_state(y_factual, y_hat)
    auc = auc_metric.result()

    return total_loss, accuracy, auc


def get_train_vaild_data(train_filename, validation_filename):
    train_data = []
    with open(train_filename) as f:
        for line in f:
            user_id, treatment, label, first_visit_to_today_days, mt_charge_fee_90days, terminalc_act_nmd_disc_amt_90day, tot_user_avg_180, age, tot_user_avg_30, tot_user_avg_90, sensitivity_score, pay_amt_180days, order_num_lunch_90days_per, pt_visit_customer_accum_total, order_num_dinner_90days, act_fee_90days, ord_amt_180days, act_fee_180days, pay_amt_90days, act_fee_30days, order_num_afternoontea_90days_per, pay_amt_30days_wm, order_num_breakfast_90days_per, ord_num_90days, ord_amt_30days, pay_amt_180days_wm \
                  = line.strip().split('\t')
            if label == 'label':
                continue
            continuous_feature = [first_visit_to_today_days,mt_charge_fee_90days,terminalc_act_nmd_disc_amt_90day,tot_user_avg_180,age,tot_user_avg_30,tot_user_avg_90,sensitivity_score,pay_amt_180days,order_num_lunch_90days_per,pt_visit_customer_accum_total,order_num_dinner_90days,act_fee_90days,ord_amt_180days,act_fee_180days,pay_amt_90days,act_fee_30days,order_num_afternoontea_90days_per,pay_amt_30days_wm,order_num_breakfast_90days_per,ord_num_90days,ord_amt_30days,pay_amt_180days_wm]



            y_true = [treatment, label]

            train_data.append([y_true, continuous_feature])

    validation_data = []
    if validation_filename:
        with open(validation_filename) as f:
            for line in f:
                user_id, treatment, label, first_visit_to_today_days, mt_charge_fee_90days, terminalc_act_nmd_disc_amt_90day, tot_user_avg_180, age, tot_user_avg_30, tot_user_avg_90, sensitivity_score, pay_amt_180days, order_num_lunch_90days_per, pt_visit_customer_accum_total, order_num_dinner_90days, act_fee_90days, ord_amt_180days, act_fee_180days, pay_amt_90days, act_fee_30days, order_num_afternoontea_90days_per, pay_amt_30days_wm, order_num_breakfast_90days_per, ord_num_90days, ord_amt_30days, pay_amt_180days_wm \
                    = line.strip().split('\t')
                if label == 'label':
                    continue
                continuous_feature = [first_visit_to_today_days, mt_charge_fee_90days, terminalc_act_nmd_disc_amt_90day,tot_user_avg_180, age, tot_user_avg_30, tot_user_avg_90, sensitivity_score,pay_amt_180days, order_num_lunch_90days_per, pt_visit_customer_accum_total,order_num_dinner_90days, act_fee_90days, ord_amt_180days, act_fee_180days,pay_amt_90days, act_fee_30days, order_num_afternoontea_90days_per,pay_amt_30days_wm, order_num_breakfast_90days_per, ord_num_90days, ord_amt_30days,pay_amt_180days_wm]

                y_true = [treatment, label]
                validation_data.append([y_true, continuous_feature])

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

        label, con_features = process_feature(one_batch)

        if training:
            global_step += 1
            with tf.GradientTape() as tape:
                loss, acc, auc = run_model(model, con_features, label, training=training)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        else:
            loss, acc, auc = run_model(model, con_features, label)

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
    model = ClassifyModel()

    train_data, validation_data = get_train_vaild_data(train_filename, validation_filename)
    train_model(model, train_data, validation_data, FLAGS.epochs, output_folder)


def model_test(predict_filename: object, model_file: object) -> object:
    batch_size = 128

    model_loaded = tf.keras.models.load_model(model_file)
    print(model_loaded.summary())

    test_dataset, _ = get_train_vaild_data(predict_filename, predict_filename)

    #for t_step, one_batch in enumerate(test_dataset):
    for t_step, one_batch in enumerate(batcher(test_dataset, batch_size)):
        if t_step >= 3:
            break
        label, con_features = process_feature(one_batch)

        pred = model_loaded([con_features, con_features], training=False)

        #print(pred)


global_step = 0  # 全局变量，训练step总数量
if __name__ == "__main__":
    path = '/Users/daping/xiongdaping/MyFiles/play_data/'
    train_filename = path + 'paoti_0417_30_2yuan_sample-964441499-1721728475475.txt'
    validation_filename = path + 'paoti_0417_30_2yuan_sample-964441499-1721728475475.txt'
    output_folder = 'model_v1'

    model_train(train_filename, validation_filename, output_folder)

    #model_test(validation_filename, output_folder+"/epoch_1")