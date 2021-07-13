# -*- coding: utf-8 -*-
# @Time    : 2021-07-13 21:17
# @Author  : 吴佳杨

from tensorflow import keras
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.preprocessing.text import *
import pandas as pd
import numpy as np

BATCH_SIZE = 8          # 每个batch的数据量，如果代码无法运行可适当降低
MAXLEN = 256            # 限制句子的最大长度

# 如没有GPU或者代码报错可删除这一块
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


class DataGenerator(keras.utils.Sequence):
    """
    数据生成器，每次仅生成一个batch的数据，对于cv领域能有效缓解内存不够用的问题，对于nlp领域也能有效减少训练时间

    注意：这里与TextCNN的数据生成器有些不一样
    """
    def __init__(self, data: pd.DataFrame, tokenizer: Tokenizer, batch_size=BATCH_SIZE):
        """
        :param data: 总数据集
        :param tokenizer: 分词器
        :param batch_size: 每个batch的数据量
        """
        self.data = data
        self.tokenizer = tokenizer
        self.batch_size = batch_size

    def __len__(self):
        """
        :return: 共有多少个batch
        """
        return int(np.ceil(len(self.data) / float(self.batch_size)))    # np.ceil() 上取整

    def __getitem__(self, item):
        """
        :param item: 数组下标
        :return: 通过下标生成数据（可参考list）
        """
        batch_data = self.data[item * self.batch_size: (item + 1) * self.batch_size]
        batch_x = self.tokenizer.texts_to_sequences(batch_data['text'])
        maxlen = max([len(l) for l in batch_x])                         # 计算该batch中句子的最大长度
        maxlen = maxlen if maxlen < MAXLEN else MAXLEN
        batch_x = keras.preprocessing.sequence.pad_sequences(batch_x, maxlen)   # 将长句子截断，短句子补0
        return batch_x


class F1_score(keras.metrics.Metric):
    """
    keras二分类f1_score标准版（能力不足可先跳过这部分内容）

    注意：这里与TextCNN的F1_score是一样的
    """
    def __init__(self, name='f1_score', **kwargs):
        super(F1_score, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name='tp', initializer='zeros')       # <tf.Variable 'tp:0' shape=() dtype=float32, numpy=0.0>
        self.y_true_positives = self.add_weight(name='tp+fn', initializer='zeros')  # tp + fn
        self.y_pred_positives = self.add_weight(name='tp+fp', initializer='zeros')  # tp + fp

    def update_state(self, y_true, y_pred):
        """
        更新状态

        :param y_true: Tensor(shape=(None, 1))
        :param y_pred: Tensor(shape=(None, 2))
        """
        y_true = K.cast(y_true[:, 0], dtype='float32')                  # shape = (None), dtype='float32'
        y_pred = K.cast(K.argmax(y_pred, axis=-1), dtype='float32')     # shape = (None), dtype='float32'
        self.true_positives.assign_add(K.sum(y_true * y_pred))          # 只能通过'assign()'和相关方法给tf.Variable赋值
        self.y_true_positives.assign_add(K.sum(y_true))
        self.y_pred_positives.assign_add(K.sum(y_pred))

    def result(self):
        """
        :return: 计算f1_score
        """
        precision = self.true_positives / (self.y_pred_positives + K.epsilon())
        recall = self.true_positives / (self.y_true_positives + K.epsilon())
        f1 = 2 * precision * recall / (precision + recall + K.epsilon())
        return f1


# 加载测试集和分词器（字典）
test = pd.read_csv('../data/test.csv', index_col=0)
with open('vocab.json', 'r') as f:
    tokenizer = tokenizer_from_json(f.read())
test_generator = DataGenerator(test, tokenizer)

# 加载模型并进行预测，预测结果保存为y_pred.csv
model = keras.models.load_model('cnn_model.h5', custom_objects={"F1_score": F1_score})
y = model.predict(test_generator, verbose=1)
y = np.argmax(y, axis=-1)
y = pd.DataFrame({'label': y})
y['id'] = y.index
y.to_csv('y_pred.csv', index=False)
