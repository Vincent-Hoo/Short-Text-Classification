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

parser = ArgumentParser("nodevec_classifier", formatter_class = ArgumentDefaultsHelpFormatter, conflict_handler = "resolve")
parser.add_argument("--input", required = True)
args = parser.parse_args()

filename = 'node_docvec_only.log'
d = 100

data = pd.read_csv('../../data/news_data/clean_data.csv')
doc2vec_model = KeyedVectors.load_word2vec_format(args.input)
LOG(filename, "\n\n\n")
LOG(filename, "configuration: " + args.input)


X_total = np.zeros((data.shape[0], d))
for i in range(data.shape[0]):
    X_total[i] = doc2vec_model[str(i)]
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