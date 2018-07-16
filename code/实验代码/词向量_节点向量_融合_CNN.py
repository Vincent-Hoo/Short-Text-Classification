#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 11:13:00 2018

@author: hzb
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Embedding, Conv2D, GlobalMaxPooling2D, Activation, Dense, concatenate, Merge
from keras.layers.core import Reshape
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K
from keras.layers import Lambda
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import OneHotEncoder
from gensim.models import Word2Vec, KeyedVectors
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

def manual_tokenizer(X, maxlen):
    new_X = np.zeros((X.shape[0], maxlen))
    for i in range(X.shape[0]):
        word_list = X[i].split(" ")
        temp = np.zeros((maxlen,))
        if len(word_list) <= 30:
            for j in range(len(word_list)):
                temp[j] = vocab[word_list[j]]
        else:
            for j in range(30):
                temp[j] = vocab[word_list[j]]
        new_X[i] = temp
    new_X = new_X.astype(int)
    return new_X

MAX_WORD_NUM = 30
d = 100

train_data = pd.read_csv('../../data/news_data/clean_train_data.csv')
test_data = pd.read_csv('../../data/news_data/clean_test_data.csv')
data = pd.read_csv('../../data/news_data/clean_data.csv')
vocab = pd.read_csv('../../data/news_data/vocab.txt')
word2vec_model = KeyedVectors.load_word2vec_format('../../data/news_data/model/word2vec_dim100_format')
node2vec_model = KeyedVectors.load_word2vec_format('../../data/news_data/word_level/no_early_stopwords/context_2/word_cont2_emb100_len5_num15.emb')

hash_map = dict()
for index, row in vocab.iterrows():
    hash_map[row['word']] = index
vocab = hash_map


train_data['aggregation'] = train_data['title'] +  ' ' + train_data['content']
test_data['aggregation'] = test_data['title'] +  ' ' + test_data['content']


enc = OneHotEncoder()
X_train = train_data.aggregation.values
X_val = test_data.aggregation.values
Y_train = train_data.label.values
Y_val = test_data.label.values
Y_train = enc.fit_transform(np.array([Y_train]).T).toarray()
Y_val = enc.fit_transform(np.array([Y_val]).T).toarray()



X_train = manual_tokenizer(X_train, MAX_WORD_NUM)
X_val = manual_tokenizer(X_val, MAX_WORD_NUM)

word_embedding_matrix = np.zeros((len(vocab), d))
node_embedding_matrix = np.zeros((len(vocab), d))
for word in vocab.keys():
    pos = vocab[word]
    word_embedding_matrix[pos] = word2vec_model.wv[word]

for i in range(node_embedding_matrix.shape[0]):
    node_embedding_matrix[i] = node2vec_model.wv[str(i)]
    
inputs = Input(shape = (MAX_WORD_NUM,))
word_embedding = Embedding(input_dim = len(vocab), output_dim = d, weights = [word_embedding_matrix], trainable = True, input_length = MAX_WORD_NUM)(inputs)
node_embedding = Embedding(input_dim = len(vocab), output_dim = d, weights = [node_embedding_matrix], trainable = True, input_length = MAX_WORD_NUM)(inputs)


my_stack = Lambda(lambda x: K.stack(x, axis = 3))
merged = my_stack([word_embedding, node_embedding])


conv_output = []
for filter_ in [2,3,4]:
    conv = Conv2D(256, (filter_, d), padding = 'same')(merged)
    conv = Activation('relu')(conv)
    conv = GlobalMaxPooling2D()(conv)
    conv_output.append(conv)

x = concatenate(conv_output)
x = Dense(400)(x)
x = Activation('relu')(x)
x = Dense(4)(x)
outputs = Activation('softmax')(x)

model = Model(inputs, outputs)

model.compile(optimizer=Adam(), metrics=['accuracy'], loss='categorical_crossentropy')


print('Start Training')
model.fit(x = X_train, y = Y_train, epochs=1)

score = model.evaluate(X_val, Y_val, verbose = 1)
print("accuracy: " + str(score[1]))
