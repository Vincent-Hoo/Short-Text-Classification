#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
这个代码是用来生成文本-词的二部图，导出图的edgelist，二部图中word只能和text相连
图节点的编号是先text再word
"""
from scipy.io import mmwrite, mmread
from scipy.sparse import csr_matrix
import numpy as np
import pandas as pd

data = pd.read_csv('../data/news_data/clean_data.csv')
vocab = pd.read_csv('../data/news_data/vocab.txt')
data['words'] = data.apply((lambda x: x.title + " " +  x.content), axis = 1)


A = np.zeros((data.shape[0], vocab.shape[0]))

hash_map = dict()
for index, row in vocab.iterrows():
    hash_map[row['word']] = index
vocab = vocab['word'].values


# 统计word co-occurrence信息
for index, row in data.iterrows():
    word_list = row['words'].split(' ')
    for word in word_list:
        pos = hash_map[word]
        A[index][pos] = 1

# 生成二部图
a = np.where(A != 0)
start = a[0]
end = a[1] + A.shape[0]
df = pd.DataFrame(columns = ['start', 'end'])

df['start'] = start
df['end'] = end
df.to_csv('../data/news_data/document_level/document_word_biparite.edgelist', index = None, header = None, sep = ' ')

