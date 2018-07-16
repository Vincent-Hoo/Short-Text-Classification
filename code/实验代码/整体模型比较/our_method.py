#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 16:42:39 2018

@author: hzb
"""


import pandas as pd
import numpy as np
from keras.layers import Input, Embedding, Conv2D, GlobalMaxPooling2D, Activation, Dense, concatenate, Lambda
from keras import backend as K
from keras.models import Model
from keras.optimizers import Adam
from gensim.models import Word2Vec, KeyedVectors
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import OneHotEncoder
import xgboost as xgb
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


'''
parser = ArgumentParser("nodevec_classifier", formatter_class = ArgumentDefaultsHelpFormatter, conflict_handler = "resolve")
parser.add_argument("--input", required = True)
parser.add_argument("--filename", required = True)
args = parser.parse_args()
'''
#filename = args.filename
filename = 'word_node2vec_merged.log'

d = 100
MAX_WORD_NUM = 30

train_data = pd.read_csv('../../../data/news_data/clean_train_data.csv')
test_data = pd.read_csv('../../../data/news_data/clean_test_data.csv')
vocab = pd.read_csv('../../../data/news_data/vocab.txt')
doc2vec_model = KeyedVectors.load_word2vec_format('../../../data/news_data/document_level/document_dim50_len100_num20.emb')
node2vec_model = KeyedVectors.load_word2vec_format('../../../data/news_data/word_level/no_early_stopwords/context_3/word_cont3_emb100_len15_num15.emb')
word2vec_model = KeyedVectors.load_word2vec_format('../../../data/news_data/model/word2vec_dim100_format')

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

doc_train = np.zeros((X_train.shape[0], 50))
doc_val = np.zeros((X_val.shape[0], 50))
for i in range(doc_train.shape[0]):
    doc_train[i] = doc2vec_model.wv[str(i)]
for i in range(doc_val.shape[0]):
    doc_val[i] = doc2vec_model.wv[str(i + doc_train.shape[0])]
print(doc_train.shape, doc_val.shape)
	
inputs = Input(shape = (MAX_WORD_NUM,))
word_embedding = Embedding(input_dim = len(vocab), output_dim = d, weights = [word_embedding_matrix], trainable = True, input_length = MAX_WORD_NUM)(inputs)
node_embedding = Embedding(input_dim = len(vocab), output_dim = d, weights = [node_embedding_matrix], trainable = True, input_length = MAX_WORD_NUM)(inputs)


my_stack = Lambda(lambda x: K.stack(x, axis = 3))
merged = my_stack([word_embedding, node_embedding])

conv_output = []
for i in range(100):
    height = np.random.randint(10) + 1
    conv = Conv2D(1, (height, 100), padding = 'same')(merged)
    conv = Activation('relu')(conv)
    conv = GlobalMaxPooling2D()(conv)
    conv_output.append(conv)

doc_input = Input(shape = (50, ))	

x = concatenate(conv_output, name = 'a' )
x = concatenate([x, doc_input])
print(x.shape)
x = Dense(100)(x)
x = Activation('relu')(x)
x = Dense(60)(x)
x = Activation('relu')(x)
x = Dense(20)(x)
x = Activation('relu')(x)
x = Dense(4)(x)
outputs = Activation('softmax')(x)


model = Model([inputs, doc_input], outputs)

model.compile(optimizer=Adam(), metrics=['accuracy'], loss='categorical_crossentropy')

model.fit(x = [X_train, doc_train], y = Y_train, epochs=1)

score = model.evaluate([X_train, doc_train], Y_train, verbose = 1)
print(score)
score = model.evaluate([X_val, doc_val], Y_val, verbose = 1)
print(score)


