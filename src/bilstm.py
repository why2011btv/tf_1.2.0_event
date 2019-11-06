'''Model for holding TF parts. etc.'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

import sys
if '../utils' not in sys.path:
    sys.path.append('../utils')

import data as pymod_data
from data import Data
import pickle

# Orthogonal Initializer from
# https://github.com/OlavHN/bnlstm
def orthogonal(shape):
  flat_shape = (shape[0], np.prod(shape[1:]))
  a = np.random.normal(0.0, 1.0, flat_shape)
  u, _, v = np.linalg.svd(a, full_matrices=False)
  q = u if u.shape == flat_shape else v
  return q.reshape(shape)

def orthogonal_initializer(scale=1.0, dtype=tf.float32):
  def _initializer(shape, dtype=tf.float32, partition_info=None):
    return tf.constant(orthogonal(shape) * scale, dtype)
  return _initializer

class TFParts(object):
    '''TensorFlow-related things. 
    
    This is to keep TensorFlow-related components in a neat shell.
    '''

    def __init__(self, pre_text, dim, batch_size, label_weights, layers=2, encoder='lstm', combine='concat', residual=True, use_type=False, num_type=-1): #label weights = [1., 1.]
        self._pre_text = pre_text
        self._text_length = pre_text.shape[1]
        self._num_text = pre_text.shape[0]
        self._dim = dim  # dimension of states. 
        self._wv_dim = pre_text.shape[2] # dimension of word embeddings
        self._batch_size = batch_size
        self._epoch_loss = 0
        self._encoder_type = encoder
        self._layers = layers
        self._combine = combine
        self._label_weights = label_weights
        self._residual = residual
        self._num_type = 0
        self._use_type = use_type
        if use_type:
            assert (num_type > 0)
            self._num_type = num_type
        #self._num_text = len(pre_embed)
        # margins
        self.build()

    @property
    def dim(self):
        return self._dim

    @property
    def batch_size(self):
        return self._batch_size

    def build(self):
        tf.reset_default_graph()

        with tf.variable_scope("graph", initializer=orthogonal_initializer()):
            # Variables (matrix of embeddings/transformations)
            
            self._text_emb_const = tf.constant(self._pre_text)
            
            # input a batch of utt embeddings
            self._text_index = text_index = tf.placeholder(
                dtype=tf.int64,
                shape=[self.batch_size],
                name='text_index')
            
            # one text, for testing
            self._text_index_e = text_index_e = tf.placeholder(
                dtype=tf.int64,
                shape=[1],
                name='text_index_e')
                
            
            
            self._text = text = tf.nn.embedding_lookup(self._text_emb_const, text_index)
            
            self._text_e = text_e = tf.nn.embedding_lookup(self._text_emb_const, text_index_e)
            
            
            self._index1 = index1 = tf.placeholder(
                dtype=tf.int64,
                shape=[self.batch_size],
                name='index1')
            
            self._index2 = index2 = tf.placeholder(
                dtype=tf.int64,
                shape=[self.batch_size],
                name='index2')
            
            self._index1_e = index1_e = tf.placeholder(
                dtype=tf.int64,
                shape=[1],
                name='index1_e')
            
            self._index2_e = index2_e = tf.placeholder(
                dtype=tf.int64,
                shape=[1],
                name='index2_e')
            
            self._label = label = tf.placeholder(
                dtype=tf.float32,
                shape=[self.batch_size, 2],
                name='label') # pos, nul, neg
            
            if self._use_type:
                self._type1 = type1 = tf.placeholder(
                dtype=tf.int64,
                shape=[self.batch_size],
                name='type1')
                
                self._type2 = type2 = tf.placeholder(
                dtype=tf.int64,
                shape=[self.batch_size],
                name='type2')
                
                self._type1_e = type1_e = tf.placeholder(
                dtype=tf.int64,
                shape=[1],
                name='type1_e')
                
                self._type2_e = type2_e = tf.placeholder(
                dtype=tf.int64,
                shape=[1],
                name='type2_e')
                
                type1, type2, type1_e, type2_e = tf.one_hot(type1, self._num_type), tf.one_hot(type2, self._num_type), tf.one_hot(type1_e, self._num_type), tf.one_hot(type2_e, self._num_type)
            
            nindex = tf.range(self.batch_size, dtype=tf.int64)
            zindex = tf.constant([0], dtype=tf.int64)
            lweights = tf.reshape(tf.tile(self._label_weights, multiples = [self.batch_size]), (self.batch_size, 2))
            
            # ========== allowed encoders =============
            assert (self._encoder_type in ['lstm','tcn','bow', 'att'])
            print ("Creating encoder type as " + self._encoder_type + "...")
            # gru
            if self._encoder_type == 'lstm':
                text_embed = self._text
                text_embed_e = self._text_e
                for i in range(self._layers):
                    res_input = text_embed
                    res_input_e = text_embed_e
                    text_embed = tf.keras.layers.Dropout(0.2)(tf.keras.layers.Bidirectional(tf.keras.layers.CuDNNGRU(self.dim, return_sequences=True), merge_mode='concat')(text_embed))
                    text_embed_e = tf.keras.layers.Dropout(0.2)(tf.keras.layers.Bidirectional(tf.keras.layers.CuDNNGRU(self.dim, return_sequences=True), merge_mode='concat')(text_embed_e))
                    if self._residual:
                        text_embed = tf.keras.layers.Concatenate()([text_embed, res_input])
                        text_embed_e = tf.keras.layers.Concatenate()([text_embed_e, res_input_e])
            elif self._encoder_type == 'tcn':
                text_embed = self._text
                text_embed_e = self._text_e
                dilation = 1
                for i in range(self._layers):
                    res_input = text_embed
                    res_input_e = text_embed_e
                    fwd, bck = text_embed, text_embed
                    fwd_e, bck_e = text_embed_e, text_embed_e
                    fwd, fwd_e = tf.keras.layers.Conv1D(self.dim, 3, dilation_rate=dilation, padding='causal', use_bias=False)(fwd), tf.keras.layers.Conv1D(self.dim, 3, dilation_rate=dilation, padding='causal', use_bias=False)(fwd_e)
                    bck, bck_e = tf.reverse(tf.keras.layers.Conv1D(self.dim, 3, dilation_rate=dilation, padding='causal', use_bias=False)(tf.reverse(bck, [-2])), [-2]), tf.reverse(tf.keras.layers.Conv1D(self.dim, 3, dilation_rate=dilation, padding='causal', use_bias=False)(tf.reverse(bck_e, [-2])), [-2])
                    dilation *= 2
                    text_embed = tf.keras.layers.Concatenate()([fwd, bck])
                    text_embed_e = tf.keras.layers.Concatenate()([fwd_e, bck_e])
                    if self._residual:
                        text_embed = tf.keras.layers.Concatenate()([text_embed, res_input])
                        text_embed_e = tf.keras.layers.Concatenate()([text_embed_e, res_input_e])
            # =========================================
            
            d1 = tf.keras.layers.Dense(self.dim, activation='linear')
            d2 = tf.keras.layers.Dense(2, activation='softmax')
            lrelu = tf.keras.layers.LeakyReLU()
            
            index1 = tf.stack([nindex, index1], axis=1)
            index2 = tf.stack([nindex, index2], axis=1)
            index1_e = tf.stack([zindex, index1_e], axis=1)
            index2_e = tf.stack([zindex, index2_e], axis=1)
            
            token1 = tf.gather_nd(text_embed, index1)
            token2 = tf.gather_nd(text_embed, index2)
            token1_e = tf.gather_nd(text_embed_e, index1_e)
            token2_e = tf.gather_nd(text_embed_e, index2_e)
            
            if self._use_type:
                token1, token2, token1_e, token2_e = tf.concat([token1, type1], axis=-1), tf.concat([token2, type2], axis=-1), tf.concat([token1_e, type1_e], axis=-1), tf.concat([token2_e, type2_e], axis=-1)
            
            print ('token1.shape:',token1.shape)

            # ========== output and loss ==============
            print ("Combine mode:", self._combine)
            assert (self._combine in ['concat', 'subtract'])
            if self._combine == 'concat':
                token_emb = tf.keras.layers.Concatenate()([token1, token2])
                token_emb_e = tf.keras.layers.Concatenate()([token1_e, token2_e])
            elif self._combine == 'subtract':
                token_emb = tf.keras.layers.Subtract()([token1, token2])
                token_emb_e = tf.keras.layers.Subtract()([token1_e, token2_e])
            self._output = output = d2(lrelu(d1(token_emb)))
            self._output_e = output_e = d2(lrelu(d1(token_emb_e)))
            
            lweights = tf.reduce_sum(tf.multiply(lweights, label), -1)
            
            self._A_loss = A_loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(label, output, lweights))
            # =========================================

            # Force normalize pre-projected vecs
            
            # Optimizer
            self._lr = lr = tf.placeholder(tf.float32)
            self._opt = opt = tf.train.AdamOptimizer(lr)#AdagradOptimizer(lr)#tf.train.AdamOptimizer(lr)#GradientDescentOptimizer(lr)
            self._train_op_A = train_op_A = opt.minimize(A_loss)

            # Saver
            self._saver = tf.train.Saver()