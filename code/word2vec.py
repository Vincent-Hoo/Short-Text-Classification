#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
该代码直接用skip-gram模型生成词向量
"""

from gensim.models import Word2Vec
import pandas as pd
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

parser = ArgumentParser("nodevec_classifier", formatter_class = ArgumentDefaultsHelpFormatter, conflict_handler = "resolve")
parser.add_argument("--dimension", required = True)
args = parser.parse_args()

train_data = pd.read_csv('../data/news_data/clean_train_data.csv')
test_data = pd.read_csv('../data/news_data/clean_test_data.csv')

data = train_data.append(test_data)
data = data.reset_index()

def fun(x):
    return x.title + " " + x.content

data['aggregation'] = data.apply(fun, axis = 1)

corpus = data['aggregation'].values

for i in range(corpus.shape[0]):
    corpus[i] = corpus[i].split(" ")
    
model = Word2Vec(corpus, size=int(args.dimension), window=10, min_count=0, sg=1, hs=1)
model.wv.save_word2vec_format('../data/news_data/model/word2vec_dim' + str(args.dimension) + '_format')
model.save('../data/news_data/model/word2vec_dim' + str(args.dimension))