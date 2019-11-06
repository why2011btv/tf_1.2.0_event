from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import sys
if '../utils' not in sys.path:
    sys.path.append('../utils')



from data import Data
from bilm import BidirectionalLanguageModel, Batcher, weight_layers, TokenBatcher
from utils import elmo_encoder
import numpy as np

from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression as LR
from sklearn.neural_network import MLPClassifier as MLP
from sklearn.metrics import precision_recall_fscore_support

remove_stop = False
rst_file = './results/context.txt'
if len(sys.argv) > 1:
    remove_stop, length, rst_file = bool(int(sys.argv[1])), int(sys.argv[2]), sys.argv[3]

this_data = Data()
this_data.load_hieve(file_direct='../datasets/hieve_processed/', remove_stop=remove_stop)

# for elmo = begin
options_file = "../elmo/elmo_2x1024_128_2048cnn_1xhighway_options.json"
weight_file = "../elmo/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5"
vocab_file = "../elmo/vocab-2016-09-10.txt"
token_embedding_file = "../elmo/emb.hdf5"

elmo = elmo_encoder()
elmo.build(options_file, weight_file, vocab_file, token_embedding_file)

text_embed = elmo.embed_sent_batch(this_data.text, this_data.max_length)

print (text_embed.shape)

label_map = np.array([[0,1,0],[1,0,0],[0,0,1]])

X1 = np.array([text_embed[x[0]][x[3]] for x in this_data.data_cases])
X2 = np.array([text_embed[x[0]][x[4]] for x in this_data.data_cases])
Y = np.array([int(x[5]) for x in this_data.data_cases])
Yl = np.array([label_map[int(x[5])] for x in this_data.data_cases])

kf = KFold(n_splits=5, shuffle=True)

folds = [(train_index, test_index) for train_index, test_index in kf.split(X1) if np.sum(Yl[test_index][:,0]) * 1. / len(test_index) > 0.01 and np.sum(Yl[train_index][:,0])  * 1. / len(train_index) > 0.01]
print (len(folds))

fp = open(rst_file, 'w')

#classifiers = [(LR(), 'LR'), (SVM(kernel='linear'), 'SVM'), (MLP(), 'MLP')]


results = []
for train_index, test_index in folds:
    m = LR(class_weight = 'balanced')
    m.fit(np.concatenate((X1[train_index], X2[train_index]), axis=-1), Y[train_index])
    yp = m.predict(np.concatenate((X1[test_index], X2[test_index]), axis=-1))
    rst = precision_recall_fscore_support(Y[test_index], yp)
    print (rst)
    results.append(rst)

print('LR\n', file = fp)
print(np.mean(results, axis=0), file = fp)
print('\n\n', file=fp)

results = []
for train_index, test_index in folds:
    m = SVC(kernel='linear', class_weight = 'balanced')
    m.fit(np.concatenate((X1[train_index], X2[train_index]), axis=-1), Y[train_index])
    yp = m.predict(np.concatenate((X1[test_index], X2[test_index]), axis=-1))
    rst = precision_recall_fscore_support(Y[test_index], yp)
    print (rst)
    results.append(rst)

print('Linear SVM\n', file = fp)
print(np.mean(results, axis=0), file = fp)
print('\n\n', file=fp)

results = []
for train_index, test_index in folds:
    m = MLP(class_weight = 'balanced')
    m.fit(np.concatenate((X1[train_index], X2[train_index]), axis=-1), Yl[train_index])
    yp = m.predict(np.concatenate((X1[test_index], X2[test_index]), axis=-1))
    rst = precision_recall_fscore_support(Y[test_index], yp)
    print (rst)
    results.append(rst)

print('MLP\n', file = fp)
print(np.mean(results, axis=0), file = fp)
print('\n\n', file=fp)

results = []
for train_index, test_index in folds:
    m = LR(class_weight = 'balanced')
    m.fit(X1[train_index] - X2[train_index], Y[train_index])
    yp = m.predict(X1[test_index] - X2[test_index])
    rst = precision_recall_fscore_support(Y[test_index], yp)
    print (rst)
    results.append(rst)

print('LR sub\n', file = fp)
print(np.mean(results, axis=0), file = fp)
print('\n\n', file=fp)

results = []
for train_index, test_index in folds:
    m = SVC(kernel='linear',class_weight = 'balanced')
    m.fit(X1[train_index] - X2[train_index], Y[train_index])
    yp = m.predict(X1[test_index] - X2[test_index])
    rst = precision_recall_fscore_support(Y[test_index], yp)
    print (rst)
    results.append(rst)

print('Linear SVM sub\n', file = fp)
print(np.mean(results, axis=0), file = fp)
print('\n\n', file=fp)

results = []
for train_index, test_index in folds:
    m = MLP(class_weight = 'balanced')
    m.fit(X1[train_index] - X2[train_index], Yl[train_index])
    yp = m.predict(X1[test_index] - X2[test_index])
    rst = precision_recall_fscore_support(Y[test_index], yp)
    print (rst)
    results.append(rst)

print('MLP sub\n', file = fp)
print(np.mean(results, axis=0), file = fp)
print('\n\n', file=fp)

fp.close()