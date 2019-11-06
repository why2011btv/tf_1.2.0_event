"""Processing of data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import pickle
import time
import csv
from sklearn.svm import SVR
from os import listdir
from os.path import isfile, join    

from nltk.tokenize import sent_tokenize, word_tokenize

class Data(object):
    '''The abustrct class that defines interfaces for holding all data.
    '''
    def __init__(self):
        self.text = []
        self.text_verbs = []
        self.session_events = {} #dict of dict map to pos
        self.session_covered_events = []
        self.session_event_types = {}
        self.event_types = {} #surface to type_id
        self.event_vocab = set([])
        self.session_subrels = [] # list of sets of subrelations, triples (session_id, p_id, c_id)
        self.data_cases = [] # subrelations, triples (session_id, p_id, c_id, p_pos, c_pos, label)
        self.stop_words = set([])
        self.n_text = 0
        self.n_event_type = 0
        self.max_length = -1
        self.label_map = np.array([0., 1.])
        # recorded for tf_parts
        self.dim = 64
        self.wv_dim = 128
        self.sent_length = 30
        self.batch_size = 256
        self.L1=False
        self.num_pos = 5.
    
    def label_weights(self): #True, False
        return [len(self.data_cases) * 0.5 / self.num_pos, len(self.data_cases) * 0.5 / (len(self.data_cases) - self.num_pos)]

    def load_stop_words(self, filename):
        for line in open(filename):
            line = line.strip()
            self.stop_words.add(line)
        print ("Added", len(self.stop_words), "stop words.")

    def load_hieve(self, file_direct='../datasets/hieve_processed/', event_indices = [1, 5, 3], relation_indices = [1, 2, 3], key_rel = 'SuperSub', remove_stop = True, lower=True, wildcard='<UNK>', label_map = np.array([[0.,1.],[1.,0.]]), num_case_limit=-1): # event indices: [<event_id>, <seq_index>, <event_type>]   relation indices: [<event_id1>, <event_id2>, <rel_name>]
        self.label_map = label_map
        onlyfiles = [f for f in listdir(file_direct) if isfile(join(file_direct, f))]
        print ('#Files:', len(onlyfiles))
        sid, tid = -1, -1
        for fname in onlyfiles:
            sid += 1
            self.session_events[sid] = {}
            self.session_subrels.append( set([]) )
            self.session_covered_events.append( set([]) )
            for line in open(file_direct + fname):
                line = line.strip().split('\t')
                if line[0] == 'Text':
                    tokens = [wildcard if (remove_stop) and (x in self.stop_words) else x.lower() if lower else x for x in line[1].split(' ') ]
                    self.text.append(tokens)
                    if len(tokens) > self.max_length:
                        self.max_length = len(tokens)
                elif line[0] == 'Event':
                    self.session_events[sid][line[event_indices[0]]] = int(line[event_indices[1]])
                    self.event_vocab.add(self.text[-1][int(line[event_indices[1]])])
                    self.session_covered_events[sid].add(line[event_indices[0]])
                    this_typename = line[event_indices[2]]
                    this_tid = self.event_types.get(this_typename.lower())
                    if this_tid is None:
                        tid += 1
                        self.event_types[this_typename] = this_tid = tid
                    if self.session_event_types.get(sid) is None:
                        self.session_event_types[sid] = {line[event_indices[0]]: this_tid}
                    else:
                        self.session_event_types[sid][line[event_indices[0]]] = this_tid
                elif line[0] == 'Relation':
                    if line[relation_indices[2]] == key_rel:
                        self.session_subrels[sid].add((line[relation_indices[0]], line[relation_indices[1]]))
        self.n_text = sid + 1
        self.n_event_type = tid + 1
        self.num_pos = 0.
        no_rel_cases = []
        for i in range(sid):
            for x in self.session_covered_events[i]:
                for y in self.session_covered_events[i]:
                    if x != y:
                        label = 0
                        if (x, y) in self.session_subrels[i]:
                            label = 1
                            self.num_pos += 1
                        #elif (y, x) in self.session_subrels[i]:
                            #label = 2
                        #triples (session_id, p_id, c_id, p_pos, c_pos, label)
                            self.data_cases.append((i, x, y, self.session_events[i][x], self.session_events[i][y], label))
                        else:
                            no_rel_cases.append((i, x, y, self.session_events[i][x], self.session_events[i][y], label))
        self.data_cases += no_rel_cases
        if num_case_limit > 0:
            self.data_cases = self.data_cases[:num_case_limit]
        print (len(self.data_cases), 'rel cases out of', sid, 'articles.')
        print ('#event_types', self.n_event_type)
        print ('label_map',self.label_map,'  label_weights',self.label_weights())
    
    def hieve_nltk_verbs(self, wildcard='<UNK>'):
        import nltk
        nltk.download('propbank')
        nltk.download('framenet_v17')
        from nltk.corpus import propbank
        from nltk.corpus import framenet as fn
        verbs = [x.lower() for x in propbank.verbs()]
        for i in range(len(fn.lus())):
            try:
                x = fn.lu(i).name[:-2].lower()
                x = x[:x.rindex('.')]
                verbs.append(x)
            except:
                pass
        verbs = set(verbs)
        verbs |= self.event_vocab
        for tt in self.text:
            rb = []
            for x in tt:
                if x not in verbs:
                    rb.append(wildcard)
                else:
                    rb.append(x)
            self.text_verbs.append(rb)
        print ("Tokenized hieve text with FrameNet and PropBank verbs.")


    def save(self, filename):
        f = open(filename,'wb')
        pickle.dump(self.__dict__, f, pickle.HIGHEST_PROTOCOL)
        f.close()
        print("Save data object as", filename)
    def load(self, filename):
        f = open(filename,'rb')
        tmp_dict = pickle.load(f)
        self.__dict__.update(tmp_dict)
        print("Loaded data object from", filename)