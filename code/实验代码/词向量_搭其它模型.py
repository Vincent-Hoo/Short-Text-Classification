#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 18 15:24:02 2018

@author: hezb
"""

import pandas as pd
import numpy as np
from gensim.models import Word2Vec, KeyedVectors
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import OneHotEncoder
import xgboost as xgb
from LOG import LOG
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


filename = 'wordvec_only.log'
d = 50

train_data = pd.read_csv('../../data/news_data/clean_train_data.csv')
test_data = pd.read_csv('../../data/news_data/clean_test_data.csv')
vocab = pd.read_csv('../../data/news_data/vocab.txt')
word2vec_model = KeyedVectors.load_word2vec_format('../../data/news_data/model/word2vec_dim50_format')

hash_map = dict()
for index, row in vocab.iterrows():
    hash_map[row['word']] = index


train_data['aggregation'] = train_data['title'] +  ' ' + train_data['content']
test_data['aggregation'] = test_data['title'] +  ' ' + test_data['content']

def get_sentence_vector(word_list):
    sentence_vector = np.zeros(d)
    for i in range(len(word_list)):
        word = word_list[i]
        sentence_vector += word2vec_model.wv[word]
    sentence_vector /= len(word_list)    
    return sentence_vector



X_train = np.zeros((train_data.shape[0], d))
for i in range(train_data.shape[0]):
    X_train[i] = get_sentence_vector(train_data.loc[i, 'aggregation'].split(' '))
Y_train = train_data['label'].values


X_val = np.zeros((test_data.shape[0], d))
for i in range(test_data.shape[0]):
    X_val[i] = get_sentence_vector(test_data.loc[i, 'aggregation'].split(' '))
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
