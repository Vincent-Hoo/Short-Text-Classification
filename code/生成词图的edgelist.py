# -*- coding: utf-8 -*-
"""
这个代码是用来生成词图的，首先统计word co-occurrence信息，再生成图，导出edgelist
"""

import pandas as pd
import numpy as np
import networkx as nx

# parameters
early_stop_threshold = 0.28
context_size = 3

data = pd.read_csv('../data/news_data/clean_data.csv')
vocab = pd.read_csv('../data/news_data/vocab.txt')
data['words'] = data.apply((lambda x: x.title + " " +  x.content), axis = 1)


hash_map = dict()
for index, row in vocab.iterrows():
    hash_map[str(row['word'])] = index
vocab = vocab['word'].values

# function: construct early stopwords list
def construct_early_stopwords():
    mid_map = np.zeros((vocab.shape[0], 5))
    for index, row in data.iterrows():
        if isinstance(row['words'], str):
            word_list = row['words'].split(' ')
            score = row['label']
            for i in range(len(word_list)):
                word = hash_map[word_list[i]]
                mid_map[word][score-1] += 1
    early_stopwords = [vocab[index] for index in np.where((mid_map.var(axis = 1) < early_stop_threshold) == True)[0].tolist()]
    return early_stopwords, mid_map

def construct_adajcent_matrix(context):
    A = np.zeros((vocab.shape[0], vocab.shape[0]))
    for index, row in data.iterrows():
        word_list = row['words'].split(' ')


        for c in range(1, context+1):
            for i in range(len(word_list) - c):
                word1 = hash_map[word_list[i]]
                word2 = hash_map[word_list[i+c]]
                if word1 != word2:
                    A[word1][word2] += 1
                    A[word2][word1] += 1

    return A

'''
# 生成早停词
early_stop_threshold = 0.48
early_stopwords, mid_map = construct_early_stopwords()
early_stopwords.remove(np.nan)
'''

graph = construct_adajcent_matrix(context_size)
G = nx.from_numpy_matrix(graph)

edges = list(G.edges(data=True))

edge_df = pd.DataFrame(edges, columns = ['start', 'end', 'weight'])
edge_df['weight'] = edge_df['weight'].apply(lambda x: int(x['weight']))
edge_df.to_csv('../data/news_data/word_level/no_early_stopwords/context_' + str(context_size)+ '/word_cooccurence_context_3.edgelist', index = None, header = None, sep = ' ')
          