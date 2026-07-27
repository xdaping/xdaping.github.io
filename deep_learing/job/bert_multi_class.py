#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: daping
# @Date:    2024/6/12 21:19

import os
from transformers import BertTokenizer, TFBertModel
import tensorflow as tf
from tensorflow import keras
import numpy as np
import time
import logging
from sklearn.metrics import roc_auc_score
import random
import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

tf.compat.v1.flags.DEFINE_integer('epochs', 10, "训练轮数")
tf.compat.v1.flags.DEFINE_float('learning_rate', 0.0001, "模型初始学习率")
tf.compat.v1.flags.DEFINE_integer('max_seq_len', 30, "")
tf.compat.v1.flags.DEFINE_integer('con_feature_n', 5, "统计类特征数量")
tf.compat.v1.flags.DEFINE_integer('batch_size', 64, "")
tf.compat.v1.flags.DEFINE_float('lr_pow', 0.95, "")
FLAGS = tf.compat.v1.flags.FLAGS

logging.basicConfig(level=logging.INFO)


class DeliveryClassifyModel(keras.models.Model):
    tf.random.set_seed(1)  # 设置全局种子

    def __init__(self):
        super(DeliveryClassifyModel, self).__init__()


        self.l_bert = TFBertModel.from_pretrained(
            '/Users/daping/.cache/huggingface/hub/models--bert-base-chinese/snapshots/c30a6ed22ab4564dc1e3b2ecbf6e766b0611a33f')

        #self.l_bert = TFBertModel.from_pretrained('bert-base-chinese')

        # # 冻结池化层
        # for layer in  self.l_bert.layers:
        #     if 'pooler' in layer.name:
        #         layer.trainable = False



        # DNN部分
        self.bn_layer0 = keras.layers.BatchNormalization(name='bn_layer0')
        self.bn_layer1 = keras.layers.BatchNormalization(name='bn_layer1')
        self.bn_layer2 = keras.layers.BatchNormalization(name='bn_layer2')
        self.bn_layer3 = keras.layers.BatchNormalization(name='bn_layer3')

        #
        #self.dense = tf.keras.layers.Dense(128, name='dense', activation="relu")

        self.dense_0 = tf.keras.layers.Dense(256, kernel_regularizer=tf.keras.regularizers.l2(0.001), name='dense_0',
                                             activation="relu")
        self.dense_1 = tf.keras.layers.Dense(128, kernel_regularizer=tf.keras.regularizers.l2(0.001), name='dense_1',
                                             activation="relu")
        self.dense_2 = tf.keras.layers.Dense(64, kernel_regularizer=tf.keras.regularizers.l2(0.001), name='dense_2',
                                             activation="relu")
        self.dense_3 = tf.keras.layers.Dense(12, name='dense_3', activation='softmax')

    # @tf.function(input_signature=[
    #     {
    #         "input_ids": tf.TensorSpec(shape=[None, 128], dtype=tf.int32),
    #         "token_type_ids": tf.TensorSpec(shape=[None, 128], dtype=tf.int32),
    #         "attention_mask": tf.TensorSpec(shape=[None, 128], dtype=tf.int32),
    #         "con_feature": tf.TensorSpec(shape=[None, 10], dtype=tf.float32)
    #     }
    # ])
    @tf.function
    def call(self, x):
        input_ids, token_type_ids, attention_mask = x
        nlp_x = {}
        nlp_x["input_ids"] = input_ids
        nlp_x["token_type_ids"] = token_type_ids
        nlp_x["attention_mask"] = attention_mask

        nlp_x = self.l_bert(nlp_x)

        #①使用方式一
        nlp_x = nlp_x.pooler_output  # 分类
        # ①使用方式二
        #nlp_x = keras.layers.Flatten()(nlp_x.last_hidden_state)

        # nlp_x = nlp_x.last_hidden_state  # 标记级别任务（如命名实体识别）
        # nlp_x = tf.reduce_mean(nlp_x, axis=1)

        # con_x = x["con_feature"]
        #
        # con_x = self.dense(self.bn_layer(con_x))  #
        # fea = tf.concat([nlp_x, con_x], 1)  #

        #print(fea.shape)
        #
        fea =  self.dense_0(self.bn_layer0(nlp_x))
        fea =  self.dense_1(self.bn_layer1(fea))
        fea =  self.dense_2(self.bn_layer2(fea))
        fea = self.dense_3(self.bn_layer3(fea))

        return fea


def batcher(data, batch_size):
    batch_size = len(data) if batch_size > len(data) else batch_size
    if batch_size == 0:
        return None
    for start_idx in range(0, len(data), batch_size):
        excerpt = slice(start_idx, start_idx + batch_size)
        batch = data[excerpt]

        batch_lable = [int(item[0]) for item in batch]
        batch_data = [str(item[1]) for item in batch]

        batch_data = {'label': batch_lable, 'nlp_data': batch_data}
        yield batch_data


def process_feature(one_batch, tokenizer):
    label = one_batch['label']
    nlp_feature = one_batch['nlp_data']
    #con_feature = one_batch['continue_data']
    label = tf.one_hot(tf.convert_to_tensor(np.array(label)), 12)

    features = tokenizer(nlp_feature, padding='max_length', max_length=FLAGS.max_seq_len, truncation=True, return_tensors="tf")


    return label, features


def run_model(model, features, label, training=False):
    pred = model(features, training=training)

    loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(label, pred))

    correct_prediction = tf.equal(tf.argmax(pred, axis=1), tf.argmax(label, axis=1))  # tf.argmax找出每一列最大值的索引
    acc = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32))  # tf.cast转化数据类型


    return loss, acc, acc


def get_train_vaild_data(train_filename, validation_filename):
    train_data = []
    with open(train_filename) as f:
        for line in f:
            label, _, address = line.strip().split('\t')

            if label == 'label':
                continue
            train_data.append([label, address])

    validation_data = []
    if validation_filename:
        with open(validation_filename) as f:
            for line in f:
                label, _, address = line.strip().split('\t')

                if label == 'label':
                    continue
                validation_data.append([label, address])

    random.shuffle(train_data)
    random.shuffle(validation_data)
    return train_data, validation_data


def train_valid(model, dataset, epoch, summary_writer, optimizer, tokenizer, training):

    global global_step
    tv_step = 0
    tv_loss = 0.0
    tv_acc = 0.0
    tv_auc = 0.0
    for tv_step, one_batch in enumerate(batcher(dataset, FLAGS.batch_size)):

        label, features = process_feature(one_batch, tokenizer)


        feature_list = [features["input_ids"], features["token_type_ids"], features["attention_mask"]]

        # if tv_step == 2:
        #    break

        if training:
            global_step += 1
            with tf.GradientTape() as tape:
                loss, acc, auc = run_model(model, feature_list, label, training=training)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        else:
            loss, acc, auc = run_model(model, feature_list, label)

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
    tf.keras.backend.clear_session()

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

    #tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    tokenizer = BertTokenizer.from_pretrained(
        '/Users/daping/.cache/huggingface/hub/models--bert-base-chinese/snapshots/c30a6ed22ab4564dc1e3b2ecbf6e766b0611a33f')

    optimizer = tf.keras.optimizers.Adam(learning_rate=FLAGS.learning_rate)

    for epoch in tf.range(1, epochs + 1, dtype=tf.int64):
        train_valid(model, train_data, epoch, summary_writer, optimizer, tokenizer, training=True)
        train_valid(model, validation_data, epoch, summary_writer, optimizer, tokenizer, training=False)

        model_dir = result_folder + f'/epoch_{epoch}'
        # model._set_inputs([{
        #         #     "con_x1": tf.TensorSpec(shape=(None, FLAGS.con_feature_n), dtype=tf.float32, name="1"),
        #         #     "con_x2": tf.TensorSpec(shape=(None, FLAGS.con_feature_n), dtype=tf.float32, name="1")
        #         # }])
        model.save(model_dir, save_format="tf")


        logging.info('export saved model: {}'.format(model_dir))


def main(train_filename, validation_filename, output_folder):
    """
    跑训练
    @param train_filename: 训练文件
    @param validation_filename: 验证文件
    @param output_folder: 模型结果输出目录
    @return:
    """

    tf.keras.backend.clear_session()
    model = DeliveryClassifyModel()


    train_data, validation_data = get_train_vaild_data(train_filename, validation_filename)
    train_model(model, train_data, validation_data, FLAGS.epochs, output_folder)


def summary():
    tf.keras.backend.clear_session()
    model = DeliveryClassifyModel()

    # 创建实际的输入张量
    input_ids = tf.random.uniform((1, 30), dtype=tf.int32, minval=0, maxval=100)
    token_type_ids = tf.random.uniform((1, 30), dtype=tf.int32, minval=0, maxval=2)
    attention_mask = tf.random.uniform((1, 30), dtype=tf.int32, minval=0, maxval=2)

    inputs = {
        "input_ids": input_ids,
        "token_type_ids": token_type_ids,
        "attention_mask": attention_mask
    }

    # 调用模型
    output = model(inputs)

    # 打印模型摘要
    model.summary()

global_step = 0  # 全局变量，训练step总数量
if __name__ == "__main__":
    train_filename = 'address_scene_v2.txt'
    validation_filename = 'address_scene_v2.txt'
    output_folder = './address_scene_model_v2'

    #summary()
    main(train_filename, validation_filename, output_folder)
