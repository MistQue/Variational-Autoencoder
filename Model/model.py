#! -*- coding:utf-8 -*-

import os
import sys

import numpy as np
import tensorflow as tf

from encoder import Encoder
from decoder import Decoder

class Model(object):
    def __init__(self, input_dim, z_dim):
        
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.lr = 0.001
        
        # -- encoder -------
        self.encoder = Encoder([input_dim, 600, 300, 100], z_dim)
        
        # -- decoder -------
        self.decoder = Decoder([z_dim, 100, 300, 600, input_dim])
        
        
    def set_model(self):

        self.x = tf.placeholder(tf.float32, [None, self.input_dim])
        self.batch_size = tf.shape(self.x)[0]
        
        # encode
        mu, log_sigma = self.encoder.set_model(self.x, is_training = True)


        loss_kl = tf.reduce_mean(-0.5 * tf.reduce_sum(1 + 2 * log_sigma - tf.square(mu) - tf.exp(2 * log_sigma), 1))
        
        eps = tf.random_normal([self.batch_size, self.z_dim])
        z = eps * tf.exp(log_sigma) + mu
        
        # decodem, clip the value to avoid log to nan
        gen_data = self.decoder.set_model(z, is_training = True)
        gen_data_clip = tf.clip_by_value(gen_data, 1e-7, 1 - 1e-7)                     
        reconstruct_error = tf.reduce_mean(-tf.reduce_sum(self.x * tf.log(gen_data_clip) + 
                                                        (1 - self.x) * tf.log(1 - gen_data_clip), 1))
        
        
        tf.get_variable_scope().reuse_variables()
        # -- train -----
        self.obj_vae = reconstruct_error + loss_kl
        train_vars = self.encoder.get_variables()
        train_vars.extend(self.decoder.get_variables())
        self.train_vae = tf.train.AdamOptimizer(self.lr).minimize(self.obj_vae, var_list = train_vars)
     
            
        # == for sharing variables ===
        tf.get_variable_scope().reuse_variables()
        # -- for using ---------------------
        self.z_input = tf.placeholder(tf.float32, [None, self.z_dim])
        self.mu, _  = self.encoder.set_model(self.x, is_training = False)
        self.generate_data = self.decoder.set_model(self.z_input, is_training = False)
        
   
    def training_vae(self, sess, data):
        _, obj_vae = sess.run([self.train_vae, self.obj_vae],
                                  feed_dict = {self.x: data})
        return obj_vae

    def encoding(self, sess, data):
        ret = sess.run(self.mu, feed_dict = {self.x: data})
        return ret
    
    def gen_data(self, sess, z):
        datas = sess.run(self.generate_data, feed_dict = {self.z_input: z})
        return datas
    
if __name__ == u'__main__':
    model = Model(28 * 28, 2)
    model.set_model()
    
