#!/user/bin/env python
# encoding=utf-8
# @Author: daping
# @Date: 2024/7/31 19:13

from utils import multiplex
import tensorflow as tf
import numpy as np
import time
import logging
from tensorflow.keras import models, layers, regularizers
from sklearn.metrics import roc_auc_score
import random
import tqdm
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

tf.compat.v1.flags.DEFINE_integer('epochs', 1, "训练轮数")
tf.compat.v1.flags.DEFINE_float('learning_rate', 0.0001, "模型初始学习率")
tf.compat.v1.flags.DEFINE_integer('max_seq_len', 10, "")
tf.compat.v1.flags.DEFINE_integer('con_feature_n', 10, "统计类特征数量")
tf.compat.v1.flags.DEFINE_integer('batch_size', 64, "")
tf.compat.v1.flags.DEFINE_string('model_dir', '../data/chinese_rbt3_L-3_H-768_A-12', "")
tf.compat.v1.flags.DEFINE_float('lr_pow', 0.95, "")
FLAGS = tf.compat.v1.flags.FLAGS

logging.basicConfig(level=logging.INFO)


class DNNModel(models.Model):
    tf.random.set_seed(1)  # 设置全局种子

    def __init__(self, model_dir):
        super(DNNModel, self).__init__()
        self.bn_layer = layers.BatchNormalization(name='bn_layer')
        self.bn_layer0 = layers.BatchNormalization(name='bn_layer0')
        self.bn_layer1 = layers.BatchNormalization(name='bn_layer1')
        self.bn_layer2 = layers.BatchNormalization(name='bn_layer2')
        self.bn_layer3 = layers.BatchNormalization(name='bn_layer3')
        #
        self.dense = layers.Dense(128, name='dense', activation="relu")
        self.dense_0 = layers.Dense(128, kernel_regularizer=regularizers.l2(0.001), name='dense_0', activation="relu")
        self.dense_1 = layers.Dense(64, kernel_regularizer=regularizers.l2(0.001), name='dense_1', activation="relu")
        self.dense_2 = layers.Dense(32, kernel_regularizer=regularizers.l2(0.001), name='dense_2', activation="relu")
        self.dense_3 = layers.Dense(2, name='dense_3', activation='softmax')

    def call(self, x):
        nlp_x, con_x = x

        con_x = self.dense(self.bn_layer(con_x))  #
        x = tf.concat([nlp_x, con_x], 1)  #
        #
        x = self.dense_0(self.bn_layer0(x))
        x = self.dense_1(self.bn_layer1(x))
        x = self.dense_2(self.bn_layer2(x))
        x = self.dense_3(self.bn_layer3(x))

        return x


def batcher(data, batch_size):
    batch_size = len(data) if batch_size > len(data) else batch_size
    if batch_size == 0:
        return None
    for start_idx in range(0, len(data), batch_size):
        excerpt = slice(start_idx, start_idx + batch_size)
        batch = data[excerpt]

        batch_lable = [int(item[0]) for item in batch]

        continue_data = [[float(i) for i in item[1]] for item in batch]

        batch_data = {'label': batch_lable, 'continue_data': continue_data}
        yield batch_data



def process_feature(one_batch):
    label = one_batch['label']
    con_feature = one_batch['continue_data']
    label = tf.one_hot(tf.convert_to_tensor(np.array(label)), 2)

    con_feature = tf.convert_to_tensor(con_feature, dtype=tf.float32)

    return label, con_feature


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
            nlp_f,spare_f_1,spare_f_2,spare_f_3,dense_f_1,dense_f_2,dense_f_3,dense_f_4,dense_f_5,dense_f_6,dense_f_7,dense_f_8,dense_f_9,dense_f_10,class_label,regress_label = line.strip().split(',')
            continuous_feature = [dense_f_1,dense_f_2,dense_f_3,dense_f_4,dense_f_5,dense_f_6,dense_f_7,dense_f_8,dense_f_9,dense_f_10]
            if class_label == 'class_label':
                continue
            train_data.append([class_label, continuous_feature])

    validation_data = []
    if validation_filename:
        with open(validation_filename) as f:
            for line in f:
                nlp_f,spare_f_1, spare_f_2, spare_f_3, dense_f_1, dense_f_2, dense_f_3, dense_f_4, dense_f_5, dense_f_6, dense_f_7, dense_f_8, dense_f_9, dense_f_10, class_label, regress_label = line.strip().split(',')
                continuous_feature = [dense_f_1, dense_f_2, dense_f_3, dense_f_4, dense_f_5, dense_f_6, dense_f_7, dense_f_8, dense_f_9, dense_f_10]
                if class_label == 'class_label':
                    continue
                train_data.append([class_label, continuous_feature])

    random.shuffle(train_data)
    random.shuffle(validation_data)
    return train_data, validation_data


def train_valid(model, dataset, epoch, summary_writer, optimizer, training):
    """
    训练或验证运行流程
    @param model: enhanced mnl模型
    @param dataset: 训练或验证数据集
    @param epoch: 第i个epoch
    @param summary_writer: tensorboard事件写入
    @param optimizer: 优化器
    @param training: 是否训练
    @return:
    """
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
                loss, acc, auc = run_model(model, [con_features, con_features], label, training=training)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        else:
            loss, acc, auc = run_model(model, [con_features, con_features], label)

        tv_loss += loss.numpy()
        tv_acc += acc.numpy()
        tv_auc += auc.numpy()

        # if tv_step % 5 == 0:
        #    print(tv_step, tv_loss / (tv_step + 1), tv_acc / (tv_step + 1))

        # 训练过程中每个n个step输出并保存结果到tensorboard中
        if training:
            if tv_step % 5 == 0:
                logging.info("step:{}, train_loss:{},train_acc:{}, train_auc:{}".format(tv_step,
                            tv_loss / (tv_step + 1), tv_acc / (tv_step + 1), tv_auc / (tv_step + 1)))

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
         time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())),epoch.numpy(),
                         optimizer.lr.numpy(), avg_loss, avg_acc, avg_auc))


def train_model(model, train_data, validation_data, epochs, output_folder):
    """
    跑训练
    @param model: enhanced mnl模型
    @param train_dataset: 训练数据集
    @param validation_dataset: 验证数据集
    @param epochs: 训练的epoch数量
    @param output_folder: 结果输出文件目录
    @return:
    """

    result_folder = os.path.join(os.getcwd(), output_folder)
    train_log_dir = os.path.join(result_folder, 'logs')
    if not os.path.exists(train_log_dir):
        os.makedirs(train_log_dir)
    summary_writer = tf.summary.create_file_writer(train_log_dir)

    # model.l_bert.apply_adapter_freeze()


    optimizer = tf.keras.optimizers.Adam(learning_rate=FLAGS.learning_rate)

    for epoch in tf.range(1, epochs + 1, dtype=tf.int64):
        train_valid(model, train_data, epoch, summary_writer, optimizer,  training=True)
        train_valid(model, validation_data, epoch, summary_writer, optimizer,  training=False)


        model_dir = result_folder + f'/epoch_{epoch}'
        model.save(model_dir, save_format="tf")
        logging.info('export saved model: {}'.format(model_dir))


def model_train(train_filename, validation_filename, output_folder):

    tf.keras.backend.clear_session()
    model = DNNModel(FLAGS.model_dir)

    train_data, validation_data = get_train_vaild_data(train_filename, validation_filename)
    train_model(model, train_data, validation_data, FLAGS.epochs, output_folder)


def model_test(predcit_file, model_file):
    batch_size=100

    model_loaded = tf.keras.models.load_model(model_file)
    print(model_loaded.summary())

    test_data, _ = get_train_vaild_data(train_filename, validation_filename)

    for t_step, one_batch in enumerate(batcher(test_data, batch_size)):
        if t_step >=3:
            break

        label, con_features = process_feature(one_batch)

        pred = model_loaded([con_features, con_features], training =False)

        print(pred)


global_step = 0  # 全局变量，训练step总数量
if __name__ == "__main__":

    train_filename = multiplex.data_path+'test.txt'
    validation_filename = multiplex.data_path+'test.txt'

    output_folder = 'model_v1'

    model_train(train_filename, validation_filename, output_folder)

    #model_test(validation_filename, output_folder+"/epoch_1")