#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 18 15:24:02 2018

@author: hezb
"""

import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import OneHotEncoder
import xgboost as xgb
from LOG import LOG

filename = 'LAD_doc2vec.log'
d = 100

data = pd.read_csv('../../data/news_data/clean_data.csv')
X_total = pd.read_csv('../../data/news_data/model/LDA_embedding.txt', header = None, sep = ' ').values

Y_total = data['label'].values

# RandomForest
rf_classifier = RandomForestClassifier(n_estimators = 100, max_depth = 6, min_samples_split = 5, max_features = 'sqrt', verbose = 1)

sss = StratifiedShuffleSplit(n_splits = 4, test_size = 0.1)

LOG(filename, 'start training Random Forest Model')
cnt = 0
for train_index, valid_index in sss.split(X_total, Y_total):
    X_train, X_val = X_total[train_index], X_total[valid_index]
    Y_train, Y_val = Y_total[train_index], Y_total[valid_index]
    
    LOG(filename, 'ROUND: ' + str(cnt))
    cnt += 1
    rf_classifier.fit(X_train,  Y_train)

    pre_label_valid = rf_classifier.predict(X_val)
    pre_label_train = rf_classifier.predict(X_train)
    LOG(filename, 'train accuracy: ' + str(accuracy_score(Y_train, pre_label_train)))
    LOG(filename, 'valid accuracy: ' + str(accuracy_score(Y_val, pre_label_valid)))

LOG(filename, '\n')    



lr_classifier = LogisticRegression()
LOG(filename, 'start training Logistic Regression Model')
cnt = 0
for train_index, valid_index in sss.split(X_total, Y_total):
    X_train, X_val = X_total[train_index], X_total[valid_index]
    Y_train, Y_val = Y_total[train_index], Y_total[valid_index]
    
    LOG(filename, 'ROUND: ' + str(cnt))
    cnt += 1
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
cnt = 0
for train_index, valid_index in sss.split(X_total, Y_total):
    X_train, X_val = X_total[train_index], X_total[valid_index]
    Y_train, Y_val = Y_total[train_index], Y_total[valid_index]

    LOG(filename, 'ROUND: ' + str(cnt))
    cnt += 1
    
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