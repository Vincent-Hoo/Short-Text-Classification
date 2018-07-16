#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 16:42:39 2018

@author: hzb
"""


import pandas as pd
import numpy as np
from keras.layers import Input, Embedding, Conv2D, GlobalMaxPooling2D, Activation, Dense, concatenate, merge, Lambda
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
from LOG import LOG
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import tensorflow as tf
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

parser = ArgumentParser("nodevec_classifier", formatter_class = ArgumentDefaultsHelpFormatter, conflict_handler = "resolve")
parser.add_argument("--input", required = True)
parser.add_argument("--input_wordvec", required = True)
parser.add_argument("--d", required = True)
args = parser.parse_args()

config  = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)


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

d = int(args.d)
MAX_WORD_NUM = 30

train_data = pd.read_csv('../../data/news_data/clean_train_data.csv')
test_data = pd.read_csv('../../data/news_data/clean_test_data.csv')
vocab = pd.read_csv('../../data/news_data/vocab.txt')
#node2vec_model = KeyedVectors.load_word2vec_format(args.input)
node2vec_model = KeyedVectors.load_word2vec_format(args.input)
word2vec_model = KeyedVectors.load_word2vec_format(args.input_wordvec)

LOG(filename, "\n\n\n")
LOG(filename, "configuration: " + args.input)
LOG(filename, "configuration: " + args.input_wordvec)

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

print("preparation finished")

inputs = Input(shape = (MAX_WORD_NUM,))
word_embedding = Embedding(input_dim = len(vocab), output_dim = d, weights = [word_embedding_matrix], trainable = False, input_length = MAX_WORD_NUM)(inputs)
node_embedding = Embedding(input_dim = len(vocab), output_dim = d, weights = [node_embedding_matrix], trainable = False, input_length = MAX_WORD_NUM)(inputs)


my_stack = Lambda(lambda x: K.stack(x, axis = 3))
merged = my_stack([word_embedding, node_embedding])

#merged = tf.stack([word_embedding, node_embedding], axis = 3)


conv_output = []
for i in range(100):
    height = np.random.randint(10) + 1
    conv = Conv2D(1, (height, d), padding = 'same')(merged)
    conv = Activation('relu')(conv)
    conv = GlobalMaxPooling2D()(conv)
    conv_output.append(conv)

x = concatenate(conv_output, name = 'a' )
x = Dense(4)(x)
outputs = Activation('softmax')(x)


model = Model(inputs, outputs)

model.compile(optimizer=Adam(), metrics=['accuracy'], loss='categorical_crossentropy')

model.fit(x = X_train, y = Y_train, epochs=1)

dense1_layer_model = Model(inputs=model.input, outputs=model.get_layer('a').output)  


A = np.row_stack([X_train, X_val])
dense1_output = dense1_layer_model.predict(A)  
print(dense1_output.shape)












X_train = dense1_output[:120000]
Y_train = train_data['label'].values
print(X_train.shape)

X_val = dense1_output[120000:]
Y_val = test_data['label'].values

# RandomForest
rf_classifier = RandomForestClassifier(n_estimators = 100, max_depth = 6, min_samples_split = 5, max_features = 'sqrt')
lr_classifier = LogisticRegression()



LOG(filename, 'start training Random Forest Model')
rf_classifier.fit(X_train,  Y_train)
pre_label_valid = rf_classifier.predict(X_val)
pre_label_train = rf_classifier.predict(X_train)
LOG(filename, 'train accuracy: ' + str(accuracy_score(Y_train, pre_label_train)))
LOG(filename, 'valid accuracy: ' + str(accuracy_score(Y_val, pre_label_valid)))


LOG(filename, '\n')    




LOG(filename, 'start training Logistic Regression Model')
lr_classifier.fit(X_train,  Y_train)
pre_label_valid = lr_classifier.predict(X_val)
pre_label_train = lr_classifier.predict(X_train)
LOG(filename, 'train accuracy: ' + str(accuracy_score(Y_train, pre_label_train)))
LOG(filename, 'valid accuracy: ' + str(accuracy_score(Y_val, pre_label_valid)))


LOG(filename, '\n')    


xgb_param = {'max_depth':7,
             'subsample': 0.6,
             'colsample_bytree':0.6,
             'colsample_bylevel':0.6,
             'eta':0.02,
             'alpha': 1.5,
             'lambda': 0.8,
             'objective':'multi:softmax',
             'eval_metric': 'logloss',
             'num_class': 5,
	     'silent': 1
             }


LOG(filename, 'start training Xgboost Model')
dtrain = xgb.DMatrix(X_train, label = Y_train)
dvalid = xgb.DMatrix(X_val, label = Y_val)
bst = xgb.train(xgb_param, dtrain, 200)
valid_pre = bst.predict(dvalid)
train_pre = bst.predict(dtrain)
train_accuracy = accuracy_score(Y_train, train_pre)
valid_accuracy = accuracy_score(Y_val, valid_pre)
LOG(filename, 'train accuracy: ' + str(train_accuracy))
LOG(filename, 'valid accuracy: ' + str(valid_accuracy))

''''''