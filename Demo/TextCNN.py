# -*- coding: utf-8 -*-
# @Time    : 2021-07-13 16:59
# @Author  : 吴佳杨

from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras.preprocessing.text import *
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

BATCH_SIZE = 8          # 每个batch的数据量，如果代码无法运行可适当降低
MAXLEN = 256            # 限制句子的最大长度


class DataGenerator(keras.utils.Sequence):
    """
    数据生成器，每次仅生成一个batch的数据，对于cv领域能有效缓解内存不够用的问题，对于nlp领域也能有效减少训练时间
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
        batch_y = batch_data['label'].values
        return batch_x, batch_y


class F1_score(keras.metrics.Metric):
    """
    keras二分类f1_score标准版（能力不足可先跳过这部分内容）
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


class AutoSave(keras.callbacks.Callback):
    """
    在训练过程中自动保存val_f1最高的模型并记录训练日志，能有效避免过拟合
    """
    def __init__(self):
        super().__init__()
        self.best_f1 = 0

    def on_epoch_end(self, epoch, logs=None):
        """
        每个epoch训练结束后会自动调用该函数
        """
        f = open('log.txt', 'a')
        f.write(str(epoch) + '\t' + str(logs))
        if logs['val_f1'] >= self.best_f1:
            self.best_f1 = logs['val_f1']
            self.model.save('cnn_model.h5')
            f.write('\tSaved')
        f.write('\n')
        f.close()


# 读取数据集并构建分词器（字典）
train = pd.read_csv('../data/train.csv', index_col=0)
val = pd.read_csv('../data/val.csv', index_col=0)
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train['text'])
tokenizer.fit_on_texts(val['text'])
with open('vocab.json', 'w') as f:
    f.write(tokenizer.to_json())        # 保存分词器（字典）

# 构建TextCNN模型
input_layer = keras.Input((None,))
emb = keras.layers.Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=128)(input_layer)
c1 = keras.layers.Conv1D(filters=64, kernel_size=2)(emb)
c2 = keras.layers.Conv1D(filters=64, kernel_size=3)(emb)
c3 = keras.layers.Conv1D(filters=64, kernel_size=4)(emb)
pool1 = keras.layers.GlobalMaxPool1D()(c1)
pool2 = keras.layers.GlobalMaxPool1D()(c2)
pool3 = keras.layers.GlobalMaxPool1D()(c3)
pool_output = keras.layers.concatenate([pool1, pool2, pool3])
d = keras.layers.Dropout(0.2)(pool_output)
y = keras.layers.Dense(2, activation='softmax')(d)
textcnn_model = keras.Model(input_layer, y)

# 训练并保存模型
train_generator = DataGenerator(train, tokenizer)
val_generator = DataGenerator(val, tokenizer)
textcnn_model.compile(optimizer='adam', loss=keras.losses.SparseCategoricalCrossentropy(), metrics=F1_score(name='f1'))
history = textcnn_model.fit(train_generator, epochs=10, validation_data=val_generator, callbacks=[AutoSave()])

# 可视化训练结果
figure, ax = plt.subplots(1, 2, figsize=(16, 9))
ax[1].plot(history.history['loss'])
ax[1].plot(history.history['val_loss'])
ax[1].set_title('Model loss')
ax[1].set_ylabel('Loss')
ax[1].set_xlabel('Epoch')
ax[1].legend(['Train', 'Test'], loc='upper left')
ax[0].plot(history.history['f1'])
ax[0].plot(history.history['val_f1'])
ax[0].set_title('Model F1_score')
ax[0].set_ylabel('F1_score')
ax[0].set_xlabel('Epoch')
ax[0].legend(['Train', 'Test'], loc='upper left')
plt.show()


