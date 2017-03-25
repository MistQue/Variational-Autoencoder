#! -*- coding:utf-8 -*-


import os
import sys

import numpy as np
import math

import matplotlib.pyplot as plt

class Sampler(object):
    def __init__(self, batch_size, z_dim):
        
        self.batch_size = batch_size
        self.z_dim = z_dim

    def __call__(self):     
        return np.random.normal(size = [self.batch_size, self.z_dim])


if __name__ == u'__main__':
    
    s = Sampler(10, 2)
    p = s()
    print 'batch_size:10'
    print 'z_dim:2'
    print 'shape', np.shape(p)
   