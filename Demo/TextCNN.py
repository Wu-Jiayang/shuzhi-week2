# -*- coding: utf-8 -*-
# @Time    : 2021-07-13 16:59
# @Author  : 吴佳杨

from tensorflow import keras
from tensorflow.keras.preprocessing.text import *
import pandas as pd

train = pd.read_csv('../data/train.csv', index_col=0)
val = pd.read_csv('../data/val.csv', index_col=0)
print(train)