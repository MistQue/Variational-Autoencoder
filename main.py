#! -*- coding:utf-8 -*-

import os
import sys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from Model.model import Model
from tensorflow.examples.tutorials.mnist import input_data

if __name__ == u'__main__':

    # parameter
    batch_size = 100
    epoch_num = 300
    z_dim = 2
    training_num = 55000
    dataset_select = 0
    
    # get_data
    print('-- get data--')
    mnist = input_data.read_data_sets("MNIST_data/", one_hot = True) 
    data = mnist.train.images[0:training_num]
    input_dim = np.shape(data)[1]

    # make model
    print('-- make model --')
    model = Model(input_dim, z_dim)
    model.set_model()
    
    # training
    print('-- begin training --')
    num_one_epoch = len(data) // batch_size
    
    record_vae = np.zeros(epoch_num)
 
    with tf.Session() as sess:
        saver = tf.train.Saver()
        init = tf.initialize_all_variables()
        sess.run(init)
        
        for epoch in range(epoch_num):
            print('** epoch {} begin **'.format(epoch))
            obj_vae = 0.0
            np.random.shuffle(data)
            for step in range(num_one_epoch):
                
                # get batch data
                batch_data = data[step * batch_size: (step + 1) * batch_size]
                
                # train
                obj_vae += model.training_vae(sess, batch_data)
                
                if step%10 == 0:
                    print('   step {}/{} end'.format(step, num_one_epoch));sys.stdout.flush()
            
                        
            record_vae[epoch] = obj_vae / float(num_one_epoch)

            print('epoch:{}, VAE loss = {}'.format(epoch, record_vae[epoch]))
            saver.save(sess, './Save/model.ckpt')

            
    np.save('./Save/record_vae.npy', record_vae)


