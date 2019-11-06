''' Module for training TF parts.'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import time

import sys

if '../utils' not in sys.path:
    sys.path.append('../utils')

import data  
import bilstm

from utils import elmo_encoder


class Container(object):
    def __init__(self):
        self.text = None
        self.index1 = None
        self.index2 = None
        self.type1 = None
        self.type2 = None
        self.Y = None
        self.text_indices = None
        self.tf_parts = None
        self.batch_size = 64
        self.sess = None

    def load(self, tf_parts):
        self.tf_parts = tf_parts
        self.batch_size = tf_parts.batch_size
        self.use_type = tf_parts._use_type

    def gen_A_batch(self, indices=None, forever=False, shuffle=True):
        l = self.index1.shape[0]
        if indices is None:
            indices = np.arange(l)
        else:
            indices = np.array(indices)
        while True:
            i1, i2, Y, text_indices, t1, t2 = self.index1, self.index2, self.Y, self.text_indices, self.type1, self.type2
            if shuffle:
                np.random.shuffle(indices)
            for i in range(0, l, self.batch_size):
                batchX = text_indices[indices[i: i+self.batch_size]]
                if batchX.shape[0] < self.batch_size:
                    batchX = np.concatenate((batchX, text_indices[indices[:self.batch_size - batchX.shape[0]]]), axis=0)
                    if batchX.shape[0] != self.batch_size:
                        print (batchX.shape)
                    assert (batchX.shape[0] == self.batch_size)
                batch_i1 = i1[indices[i: i+self.batch_size]]
                if batch_i1.shape[0] < self.batch_size:
                    batch_i1 = np.concatenate((batch_i1, i1[indices[:self.batch_size - batch_i1.shape[0]]]), axis=0)
                    assert (batch_i1.shape[0] == self.batch_size)
                batch_i2 = i2[indices[i: i+self.batch_size]]
                if batch_i2.shape[0] < self.batch_size:
                    batch_i2 = np.concatenate((batch_i2, i2[indices[:self.batch_size - batch_i2.shape[0]]]), axis=0)
                    assert (batch_i2.shape[0] == self.batch_size)
                batchY = Y[indices[i: i+self.batch_size]]
                if batchY.shape[0] < self.batch_size:
                    batchY = np.concatenate((batchY, Y[indices[:self.batch_size - batchY.shape[0]]]), axis=0)
                    assert (batchY.shape[0] == self.batch_size)
                if self.use_type:
                    batch_t1 = t1[indices[i: i+self.batch_size]]
                    if batch_t1.shape[0] < self.batch_size:
                        batch_t1 =  np.concatenate((batch_t1, t1[indices[:self.batch_size - batch_t1.shape[0]]]), axis=0)
                    batch_t2 = t2[indices[i: i+self.batch_size]]
                    if batch_t2.shape[0] < self.batch_size:
                        batch_t2 =  np.concatenate((batch_t2, t2[indices[:self.batch_size - batch_t2.shape[0]]]), axis=0)
                    yield batchX, batch_i1.astype(np.int64), batch_i2.astype(np.int64), batchY, batch_t1.astype(np.int64), batch_t2.astype(np.int64)
                else:
                    yield batchX, batch_i1.astype(np.int64), batch_i2.astype(np.int64), batchY
            if not forever:
                break

    def fit(self, text_indices, i1, i2, Y, indices, t1=None, t2=None, epochs=20, lr=0.001):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.index1,self.index2,self.Y,self.text_indices,self.type1,self.type2 = i1, i2, Y, text_indices, t1, t2
        if self.use_type:
            assert (self.type1 is not None and self.type2 is not None)
        self.sess = sess = tf.Session(config=config) # allow memory auto growth
        sess.run(tf.initialize_all_variables())
        
        num_A_batch = len(list(self.gen_A_batch(indices)))
        print('num_A_batch =', num_A_batch)
        
        # margins
        t0 = time.time()
        for epoch in range(epochs):
            epoch_loss = self.train1epoch(sess, num_A_batch, indices, lr, epoch + 1)
            print("Time use: %d" % (time.time() - t0))
            if np.isnan(epoch_loss):
                print("Training collapsed.")
                return
            #if (epoch + 1) % save_every_epoch == 0:
            #    this_save_path = self.tf_parts._saver.save(sess, self.save_path)
            #    self.this_data.save(self.data_save_path)
            #    print("bilstm saved in file: %s. Data saved in file: %s" % (this_save_path, self.data_save_path))
        #this_save_path = self.tf_parts._saver.save(sess, self.save_path)
        #self.this_data.save(self.data_save_path)
        #print("bilstm saved in file: %s" % this_save_path)
        #sess.close()
        print("Done")
    
    def predict_one(self, text_id, i1, i2, t1=None, t2=None):
        if self.use_type:
            r = self.sess.run([self.tf_parts._output_e],
                                feed_dict={self.tf_parts._text_index_e: [text_id], 
                                           self.tf_parts._index1_e: [i1],
                                           self.tf_parts._index2_e: [i2],
                                           self.tf_parts._type1_e: [t1],
                                           self.tf_parts._type2_e: [t2]}
                        )[0]
        else:
            r = self.sess.run([self.tf_parts._output_e],
                                feed_dict={self.tf_parts._text_index_e: [text_id], 
                                           self.tf_parts._index1_e: [i1],
                                           self.tf_parts._index2_e: [i2]}
                        )[0]
        return r
    
    
    def close(self):
        self.sess.close()
        print ("Sess closed.")
    

    def train1epoch(self, sess, num_A_batch, indices, lr, epoch):
        '''build and train a model.

        Args:
            self.batch_size: size of batch
            num_epoch: number of epoch. A epoch means a turn when all A/B_t/B_h/C are passed at least once.
            dim: dimension of embedding
            lr: learning rate
            self.this_data: a Data object holding data.
            save_every_epoch: save every this number of epochs.
            save_path: filepath to save the tensorflow model.
        '''

        this_gen_A_batch = self.gen_A_batch(indices, forever=True)
        
        this_loss, loss_A = 0., 0.

        for batch_id in range(num_A_batch):
            # Optimize loss A
            if self.use_type:
                text, i1, i2, Y, t1, t2  = next(this_gen_A_batch)
                _, loss_A = sess.run([self.tf_parts._train_op_A, self.tf_parts._A_loss],
                            feed_dict={self.tf_parts._text_index: text, 
                                       self.tf_parts._index1: i1,
                                       self.tf_parts._index2: i2,
                                       self.tf_parts._label: Y,
                                       self.tf_parts._type1: t1,
                                       self.tf_parts._type2: t2,
                                       self.tf_parts._lr: lr})
            else:
                text, i1, i2, Y  = next(this_gen_A_batch)
                _, loss_A = sess.run([self.tf_parts._train_op_A, self.tf_parts._A_loss],
                            feed_dict={self.tf_parts._text_index: text, 
                                       self.tf_parts._index1: i1,
                                       self.tf_parts._index2: i2,
                                       self.tf_parts._label: Y,
                                       self.tf_parts._lr: lr})
            # Observe total loss
            this_loss += loss_A
            
            if ((batch_id + 1) % 50 == 0) or batch_id == num_A_batch - 1:
                print('process: %d / %d. Epoch %d' % (batch_id+1, num_A_batch, epoch))

        this_avg_loss = this_loss / num_A_batch
        print("Loss of epoch %d = %s, which is %s per case" % (epoch, this_loss, this_avg_loss))
        return this_avg_loss

# A safer loading is available in Tester, with parameters like batch_size and dim recorded in the corresponding Data component
def load_tfparts(batch_size = 128,
                dim = 64,
                this_data=None,
                save_path = 'this-bilstm.ckpt', L1=False):
    tf_parts = bilstm.TFParts(num_rels=this_data.num_rels(),
                             num_cons=this_data.num_cons(),
                             dim=dim,
                             batch_size=self.batch_size, L1=L1)
    with tf.Session() as sess:
        tf_parts._saver.restore(sess, save_path)