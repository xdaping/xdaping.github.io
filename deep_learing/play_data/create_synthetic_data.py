#!/user/bin/env python
# encoding=utf-8
# @Author: daping
# @Date: 2024/7/31 13:04
from utils import multiplex
import numpy as np
import pandas as pd
import random

np.random.seed(1)

num_rows =10000

file_path = multiplex.data_path + 'test.txt'


nlp_features = np.array([[''.join(chr(random.randint(0x4E00, 0x9FA5)) for _ in range(10))] for _ in range(num_rows)])

spare_features = np.random.randint(0,5, size=(num_rows, 3))
dense_features = np.random.rand(num_rows, 10) * 10

class_label = np.random.randint(0,2, size=(num_rows,1))
regress_label = np.random.rand(num_rows, 1) *100

#print(nlp_features.shape, spare_features.shape)

data = np.hstack((nlp_features, spare_features, dense_features))

labels = np.hstack((class_label, regress_label))

columns =['npl_f'] + [f'spare_f_{i+1}' for i in range(3)] + [f'dense_f_{i+1}' for i in range(10)] + ['class_label', 'regress_label']

df = pd.DataFrame(np.hstack((data, labels)), columns = columns)

df['class_label'] = df['class_label'].astype(float).astype(int)

print(df.head())

df.to_csv(file_path, index = False)

