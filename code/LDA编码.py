# -*- coding: utf-8 -*-
"""
Created on Fri May 18 11:53:29 2018

@author: Vincent Ho
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

data = pd.read_csv('../../data/news_data/clean_data.csv')
data['aggregation'] = data['title'] +  ' '+ data['content']

countVectorizer = CountVectorizer()
textvector = countVectorizer.fit_transform(data['aggregation'].values)

tfidfVectorizer = TfidfTransformer()
tfidf = tfidfVectorizer.fit_transform(textvector.toarray())

print('start training')
lda = LatentDirichletAllocation(n_components=100, verbose = 1)
lda_embedding = lda.fit_transform(tfidf)
print('training finished')

df = pd.DataFrame(lda_embedding)
df.to_csv('../../data/news_data/model/LDA_embedding.txt', header = None, index = None, sep = ' ')
