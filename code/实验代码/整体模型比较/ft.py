#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 23:40:07 2018

@author: hzb
"""

import fasttext
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

train_data = pd.read_csv('../../../data/news_data/clean_train_data.csv')
test_data = pd.read_csv('../../../data/news_data/clean_test_data.csv')

train_data['aggregation'] = train_data['title'] +  ' ' + train_data['content']
test_data['aggregation'] = test_data['title'] +  ' ' + test_data['content']

def fun(x):
    return x['aggregation'] + " __label__" + str(x['label'])

train_data['aggregation'] = train_data.apply(fun, axis = 1)

train = train_data.loc[:, ['aggregation']]
train.to_csv('fasttext_trainset.txt', index = None, header = None)

classifier = fasttext.supervised(
        'fasttext_trainset.txt',
        "fasttext_train_model",
        label_prefix = '__label__'
        )

test = test_data.aggregation.values.tolist()

res = classifier.predict(test)

y_pred = []
for i in res:
    y_pred.append(int(i[0]))

y_val = test_data['label']
acc = accuracy_score(y_val, y_pred)
print(acc)