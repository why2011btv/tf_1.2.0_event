import tensorflow as tf
import numpy as np
from numpy.fft import fft, ifft
import sys
from bilm import weight_layers
from bilm import BidirectionalLanguageModel, weight_layers, TokenBatcher

#from allennlp.modules.elmo import Elmo, batch_to_ids (comment temporally)

def circular_correlation(h, t):
    return tf.real(tf.spectral.ifft(tf.multiply(tf.conj(tf.spectral.fft(tf.complex(h, 0.))), tf.spectral.fft(tf.complex(t, 0.)))))

def np_ccorr(h, t):
    return ifft(np.conj(fft(h)) * fft(t)).real


class elmo_encoder(object):
    def __init__(self):
        self.max_batch = 120000
        print ("WARNING: Currently max_batch_size of elmo encoder is set to", self.max_batch)
        pass
    
    def build(self, options_file, weight_file, vocab_file, token_embedding_file):
        self._bilm = BidirectionalLanguageModel(
            options_file,
            weight_file,
            use_character_inputs=False,
            embedding_weight_file=token_embedding_file,
            max_batch_size = self.max_batch)
        self._token_batcher = TokenBatcher(vocab_file)
        #self.length = length
    
    # sentences has to list of word lists. [['You', 'see', '?'], ['That', 'is', 'very', 'interesting', '.']]
    def embed_sent_batch(self, sentences, length):
        sentences_tokenid = self._token_batcher.batch_sentences(sentences)
        # s_tokenid = s_tokenid[1:][:-1]
        tf.reset_default_graph()
        processed_sentences_tokenid = []
        length += 2 # Take into account <s> and </s>
        for s_tokenid in sentences_tokenid:
            if (len(s_tokenid) >= length):
                s_tokenid = s_tokenid[:length]
            else:
                s_tokenid = np.pad(s_tokenid, (0, length - len(s_tokenid)), 'constant', constant_values=(0))
            #s_tokenid = np.expand_dims(s_tokenid, axis=0)
            processed_sentences_tokenid.append(s_tokenid)
        batch_size = len(processed_sentences_tokenid)
        processed_sentences_tokenid = np.array(processed_sentences_tokenid)
        # tf
        with tf.device("/cpu:0"):
            context_token_ids = tf.placeholder('int32', shape=(batch_size, length))
            context_embeddings_op = self._bilm(context_token_ids)
            elmo_context_output = weight_layers('output', context_embeddings_op, l2_coef=0.0)['weighted_op']
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            print ('++++++Check_point_1\n')
            with tf.Session(config=config) as sess:
                sess.run([tf.global_variables_initializer()])
                elmo_context_output_ = sess.run([elmo_context_output],feed_dict={context_token_ids: processed_sentences_tokenid})[0]
        #print (elmo_context_output_.shape)
        return elmo_context_output_



"""
class elmo_encoder(object):
    def __init__(self):
        self.elmo = None
        pass
    
    def build(self, options_file, weight_file, length):
        self.elmo = Elmo(options_file, weight_file, 2, dropout=0)
        self.length = length
    
    def embed_sent(self, sent):
        #print (sent)
        s_tokenid = batch_to_ids([sent])[0]
        #print (s_tokenid)
        if (len(s_tokenid) >= self.length):
            s_tokenid = s_tokenid[:self.length]
        else:
            s_tokenid = np.pad(s_tokenid, (0, self.length - len(s_tokenid)), 'constant', constant_values=(0))
        s_tokenid = np.expand_dims(s_tokenid, axis=0)
        #print (s_tokenid)
        print (s_tokenid.shape)
        embeddings = self.elmo(s_tokenid)['elmo_representations']
        embeddings = np.mean(embeddings, axis=0)
        embeddings = embeddings[-2]
        assert (len(embeddings) == self.length)
        return embeddings
"""