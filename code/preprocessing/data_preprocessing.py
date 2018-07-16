# -*- coding: utf-8 -*-
"""
Created on Thu May 17 19:19:49 2018

@author: Vincent Ho
"""

import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords


train_data = pd.read_csv('../../data/news_data/useless/train.csv', header = None)
test_data = pd.read_csv('../../data/news_data/useless/test.csv', header = None)
train_data.columns = ['label', 'title', 'content']
test_data.columns = ['label', 'title', 'content']
stopwords = stopwords.words("english")
stopwords.append('')
stopwords.append('null')
stopwords.append('nan')
stopwords.append('void')


def clean_data(x):
    # 将数字换成NUM

    nums = re.findall(r'\d+', x)
    for num in nums:
        x = x.replace(num, 'NUM')

    # 去掉标点符号
    x = re.sub('[^a-zA-Z]', ' ', x)

    # 转成小写字母
    words = x.lower().split(' ')

    # 去掉停用词
    words = [w for w in words if not w in stopwords]
    
    
    return " ".join(words)


data = train_data

data['clean_content'] = data['content'].apply(clean_data)
data['clean_title'] = data['title'].apply(clean_data)


clean_train_data = data.loc[:, ['label', 'clean_title', 'clean_content']]
clean_train_data.columns = ['label', 'title', 'content']
clean_train_data['title'] = clean_train_data['title'].replace({'': '-'})
clean_train_data['content'] = clean_train_data['content'].replace({'': '-'})
clean_train_data.to_csv('../../../data/news_data/clean_train_data.csv', index = None)




data = test_data
data['clean_content'] = data['content'].apply(clean_data)
data['clean_title'] = data['title'].apply(clean_data)


clean_test_data = data.loc[:, ['label', 'clean_title', 'clean_content']]
clean_test_data.columns = ['label', 'title', 'content']
clean_test_data['title'] = clean_test_data['title'].replace({'': '-'})
clean_test_data['content'] = clean_test_data['content'].replace({'': '-'})
clean_test_data.to_csv('../../data/news_data/clean_test_data.csv', index = None)



word_count = dict()
for index, row in clean_train_data.iterrows():
    for word in row['content'].split(' ') + row['title'].split(' '):
        if word not in word_count:
            word_count[word] = 1
        else:
            word_count[word] += 1
            
for index, row in clean_test_data.iterrows():
    for word in row['content'].split(' ') + row['title'].split(' '):
        if word not in word_count:
            word_count[word] = 1
        else:
            word_count[word] += 1
        
vocab = pd.DataFrame({'word': list(word_count.keys()), 'count': list(word_count.values())})

vocab = vocab.sort_values(['count'], ascending = False)

vocab.to_csv('../../data/news_data/vocab.txt', index = None)


clean_data = clean_train_data.append(clean_test_data)
clean_data = clean_data.reset_index()
del clean_data['index']
clean_data.to_csv('../../data/news_data/clean_data.csv', index = None)
