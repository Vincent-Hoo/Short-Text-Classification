import pandas as pd
import numpy as np
from gensim.models import Word2Vec, KeyedVectors
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import OneHotEncoder
import xgboost as xgb

d = 150

data = pd.read_csv('../../../data/news_data/clean_data.csv')
vocab = pd.read_csv('../../../data/news_data/vocab.txt')
doc2vec_model = KeyedVectors.load_word2vec_format('../../../data/news_data/document_level/document_dim50_len100_num20.emb')
node2vec_model = KeyedVectors.load_word2vec_format('../../../data/news_data/word_level/no_early_stopwords/context_3/word_cont3_emb100_len15_num15.emb')
word2vec_model = KeyedVectors.load_word2vec_format('../../../data/news_data/model/word2vec_dim100_format')



hash_map = dict()
for index, row in vocab.iterrows():
    hash_map[row['word']] = index


data['aggregation'] = data['title'] +  ' ' + data['content']


def get_sentence_vector(word_list):
    sentence_vector = np.zeros(100)
    for i in range(len(word_list)):
        word = word_list[i]
        pos = str(hash_map[word])
        sentence_vector += node2vec_model.wv[pos]
        sentence_vector += word2vec_model.wv[word]
    sentence_vector /= (2*len(word_list))    
    return sentence_vector

X_total = np.zeros((data.shape[0], d))
for i in range(data.shape[0]):
    X_total[i, :100] = get_sentence_vector(data.loc[i, 'aggregation'].split(' '))
    X_total[i, 100:] = doc2vec_model[str(i)]
print(X_total.shape)

Y_total = data['label'].values

X_train = X_total[:120000]
X_val = X_total[120000:]
Y_train = Y_total[:120000]
Y_val = Y_total[120000:]
print(X_train.shape, Y_train.shape)
print(X_val.shape, Y_val.shape)


# RandomForest
rf_classifier = RandomForestClassifier(n_estimators = 100, max_depth = 6, min_samples_split = 5, max_features = 'sqrt')
lr_classifier = LogisticRegression()



rf_classifier.fit(X_train,  Y_train)
pre_label_valid = rf_classifier.predict(X_val)
pre_label_train = rf_classifier.predict(X_train)
print('train accuracy: ' + str(accuracy_score(Y_train, pre_label_train)))
print('valid accuracy: ' + str(accuracy_score(Y_val, pre_label_valid)))





lr_classifier.fit(X_train,  Y_train)
pre_label_valid = lr_classifier.predict(X_val)
pre_label_train = lr_classifier.predict(X_train)
print('train accuracy: ' + str(accuracy_score(Y_train, pre_label_train)))
print('valid accuracy: ' + str(accuracy_score(Y_val, pre_label_valid)))




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


dtrain = xgb.DMatrix(X_train, label = Y_train)
dvalid = xgb.DMatrix(X_val, label = Y_val)
bst = xgb.train(xgb_param, dtrain, 200)
valid_pre = bst.predict(dvalid)
train_pre = bst.predict(dtrain)
train_accuracy = accuracy_score(Y_train, train_pre)
valid_accuracy = accuracy_score(Y_val, valid_pre)
print('train accuracy: ' + str(train_accuracy))
print('valid accuracy: ' + str(valid_accuracy))