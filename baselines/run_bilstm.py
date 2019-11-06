from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import sys
if '../utils' not in sys.path:
    sys.path.append('../utils')

if '../src' not in sys.path:
    sys.path.append('../src')



from data import Data
from bilm import BidirectionalLanguageModel, Batcher, weight_layers, TokenBatcher
from utils import elmo_encoder
import numpy as np

from sklearn.model_selection import KFold
from sklearn.metrics import precision_recall_fscore_support
from bilstm import TFParts
from container_bilstm import Container

exp_fold_limit = 3

remove_stop = False
rst_file = './results/context.txt'
this_dim = 64
this_combine = 'concat'
this_use_type = False
if len(sys.argv) > 1:
    remove_stop, this_use_type, this_dim, this_layer, this_combine, rst_file = bool(int(sys.argv[1])), bool(int(sys.argv[2])), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5], sys.argv[6]

this_data = Data()
this_data.load_stop_words('../elmo/stop_words_en.txt')
this_data.load_hieve(file_direct='../datasets/hieve_processed/', remove_stop=remove_stop, num_case_limit=46072)


# for elmo = begin
options_file = "../elmo/elmo_2x1024_128_2048cnn_1xhighway_options.json"
weight_file = "../elmo/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5"
vocab_file = "../elmo/vocab-2016-09-10.txt"
token_embedding_file = "../elmo/elmo_token_embeddings-2.hdf5"

elmo = elmo_encoder()
elmo.build(options_file, weight_file, vocab_file, token_embedding_file)

text_embed = elmo.embed_sent_batch(this_data.text, this_data.max_length)

print (text_embed.shape)

text_index = np.array([int(x[0]) for x in this_data.data_cases])
index1 = np.array([int(x[3]) for x in this_data.data_cases])
index2 = np.array([int(x[4]) for x in this_data.data_cases])
Y = np.array([this_data.label_map[int(x[5])] for x in this_data.data_cases])
t1, t2 = None, None
this_num_type = -1
if this_use_type:
    this_num_type = this_data.n_event_type
    t1, t2 = np.array([this_data.session_event_types[int(x[0])][x[1]] for x in this_data.data_cases]), np.array([this_data.session_event_types[int(x[0])][x[2]] for x in this_data.data_cases])

kf = KFold(n_splits=5, shuffle=True)

folds = [(train_index, test_index) for train_index, test_index in kf.split(text_index) if np.sum(Y[test_index][:,0]) * 1. / len(test_index) > 0.01 and np.sum(Y[train_index][:,0])  * 1. / len(train_index) > 0.02]
print (len(folds))

folds = folds[:exp_fold_limit]

fp = open(rst_file, 'w')

results = []
for train_index, test_index in folds:
    model = None
    model = TFParts(text_embed, batch_size=64, dim=this_dim, label_weights=this_data.label_weights(), layers=this_layer, encoder='lstm', residual=True, combine=this_combine, use_type=this_use_type, num_type=this_num_type)
    m = Container()
    m.load(model)
    m.fit(text_index, index1, index2, Y, train_index, t1, t2, epochs=80, lr=0.0002)
    yp = []
    for i in range(len(test_index)):
        if this_use_type:
            yp.append(m.predict_one(text_index[test_index[i]], index1[test_index[i]], index2[test_index[i]], t1[test_index[i]], t2[test_index[i]]))
        else:
            yp.append(m.predict_one(text_index[test_index[i]], index1[test_index[i]], index2[test_index[i]]))
    rst = precision_recall_fscore_support(np.argmax(Y[test_index], axis=-1), np.argmax(yp, axis=-1))
    print (rst)
    results.append(rst)
    m.close()

print('LR\n', file = fp)
print(np.mean(results, axis=0), file = fp)
print('\n\n', file=fp)
