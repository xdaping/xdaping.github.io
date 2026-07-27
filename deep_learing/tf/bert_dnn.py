#!/user/bin/env python
# encoding=utf-8
# @Author: daping
# @Date: 2024/8/4 18:34

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
from utils import multiplex


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

tf.compat.v1.flags.DEFINE_integer('epochs', 100, "训练轮数")
tf.compat.v1.flags.DEFINE_float('learning_rate', 0.0001, "模型初始学习率")
tf.compat.v1.flags.DEFINE_integer('max_seq_len', 10, "")
tf.compat.v1.flags.DEFINE_integer('con_feature_n', 10, "统计类特征数量")
tf.compat.v1.flags.DEFINE_integer('batch_size', 64, "")
tf.compat.v1.flags.DEFINE_float('lr_pow', 0.95, "")
FLAGS = tf.compat.v1.flags.FLAGS

logging.basicConfig(level=logging.INFO)


class DeliveryClassifyModel(keras.models.Model):
    tf.random.set_seed(1)  # 设置全局种子

    def __init__(self):
        super(DeliveryClassifyModel, self).__init__()

        self.l_bert = TFBertModel.from_pretrained('bert-base-chinese')
        # DNN部分
        self.bn_layer = keras.layers.BatchNormalization(name='bn_layer')
        self.bn_layer0 = keras.layers.BatchNormalization(name='bn_layer0')
        self.bn_layer1 = keras.layers.BatchNormalization(name='bn_layer1')
        self.bn_layer2 = keras.layers.BatchNormalization(name='bn_layer2')
        self.bn_layer3 = keras.layers.BatchNormalization(name='bn_layer3')
        #
        self.dense = tf.keras.layers.Dense(128, name='dense', activation="relu")
        self.dense_0 = tf.keras.layers.Dense(128, kernel_regularizer=tf.keras.regularizers.l2(0.001), name='dense_0',
                                             activation="relu")
        self.dense_1 = tf.keras.layers.Dense(64, kernel_regularizer=tf.keras.regularizers.l2(0.001), name='dense_1',
                                             activation="relu")
        self.dense_2 = tf.keras.layers.Dense(32, kernel_regularizer=tf.keras.regularizers.l2(0.001), name='dense_2',
                                             activation="relu")
        self.dense_3 = tf.keras.layers.Dense(2, name='dense_3', activation='softmax')
    def call(self, x):
        nlp_x={}
        nlp_x["input_ids"] = x["input_ids"]
        nlp_x["token_type_ids"] = x["token_type_ids"]
        nlp_x["attention_mask"] = x["attention_mask"]

        nlp_x = self.l_bert(nlp_x)

        #①使用方式一
        #nlp_x = nlp_x.pooler_output
        # ①使用方式二
        nlp_x = keras.layers.Flatten()(nlp_x.last_hidden_state)

        con_x = x["con_feature"]

        con_x = self.dense(self.bn_layer(con_x))  #
        fea = tf.concat([nlp_x, con_x], 1)  #

        #print(fea.shape)
        #
        fea = self.dense_0(self.bn_layer0(fea))
        fea = self.dense_1(self.bn_layer1(fea))
        fea = self.dense_2(self.bn_layer2(fea))
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

        continue_data = [[float(i) for i in item[2]] for item in batch]

        batch_data = {'label': batch_lable, 'nlp_data': batch_data, 'continue_data': continue_data}
        yield batch_data



def process_feature(one_batch, tokenizer):
    label = one_batch['label']
    nlp_feature = one_batch['nlp_data']
    con_feature = one_batch['continue_data']
    label = tf.one_hot(tf.convert_to_tensor(np.array(label)), 2)

    features = tokenizer(nlp_feature, padding=True, max_length=FLAGS.max_seq_len, truncation=True, return_tensors="tf")

    #print(len(nlp_feature[0]), features['input_ids'].shape)

    con_feature = tf.convert_to_tensor(con_feature, dtype=tf.float32)

    features["con_feature"] = con_feature

    return label, features


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
            nlp_f, spare_f_1, spare_f_2, spare_f_3, dense_f_1, dense_f_2, dense_f_3, dense_f_4, dense_f_5, dense_f_6, dense_f_7, dense_f_8, dense_f_9, dense_f_10, class_label, regress_label = line.strip().split(
                ',')
            continuous_feature = [dense_f_1, dense_f_2, dense_f_3, dense_f_4, dense_f_5, dense_f_6, dense_f_7,
                                  dense_f_8, dense_f_9, dense_f_10]

            if class_label == 'class_label':
                continue
            train_data.append([class_label, nlp_f, continuous_feature])

    validation_data = []
    if validation_filename:
        with open(validation_filename) as f:
            for line in f:
                nlp_f, spare_f_1, spare_f_2, spare_f_3, dense_f_1, dense_f_2, dense_f_3, dense_f_4, dense_f_5, dense_f_6, dense_f_7, dense_f_8, dense_f_9, dense_f_10, class_label, regress_label = line.strip().split(
                    ',')
                continuous_feature = [dense_f_1, dense_f_2, dense_f_3, dense_f_4, dense_f_5, dense_f_6, dense_f_7,
                                      dense_f_8, dense_f_9, dense_f_10]

                if class_label == 'class_label':
                    continue
                validation_data.append([class_label, nlp_f, continuous_feature])

    random.shuffle(train_data)
    random.shuffle(validation_data)
    return train_data, validation_data


def train_valid(model, dataset, epoch, summary_writer, optimizer, tokenizer,  training):
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

        label, features = process_feature(one_batch, tokenizer)

        #print(features)

        if training:
            global_step += 1
            with tf.GradientTape() as tape:
                loss, acc, auc = run_model(model, features, label, training=training)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        else:
            loss, acc, auc = run_model(model, features, label)

        tv_loss += loss.numpy()
        tv_acc += acc.numpy()
        tv_auc += auc.numpy()

        # if tv_step % 5 == 0:
        #    print(tv_step, tv_loss / (tv_step + 1), tv_acc / (tv_step + 1))

        # 训练过程中每个n个step输出并保存结果到tensorboard中
        if training:
            if tv_step % 1 == 0:
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

    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

    optimizer = tf.keras.optimizers.Adam(learning_rate=FLAGS.learning_rate)

    for epoch in tf.range(1, epochs + 1, dtype=tf.int64):
        train_valid(model, train_data, epoch, summary_writer, optimizer, tokenizer,  training=True)
        train_valid(model, validation_data, epoch, summary_writer, optimizer, tokenizer,  training=False)

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
    #model.build(input_shape=[(None, FLAGS.max_seq_len), (None, FLAGS.con_feature_n)])
    #bert.load_bert_weights(model.l_bert, model_ckpt)
    #model.summary()

    train_data, validation_data = get_train_vaild_data(train_filename, validation_filename)
    train_model(model, train_data, validation_data, FLAGS.epochs, output_folder)


global_step = 0  # 全局变量，训练step总数量
if __name__ == "__main__":

    train_filename = multiplex.data_path + 'test.txt'
    validation_filename = multiplex.data_path + 'test.txt'

    output_folder = 'delivery_model'

    main(train_filename, validation_filename, output_folder)
