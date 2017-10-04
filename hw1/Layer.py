#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 22:28:54 2017

@author: py
"""

import numpy as np
from configs import configs

config = configs()

class Layer():
    
    def __init__(self, layer_id, actFun):

        self.actFun = actFun
        self.layer_id = layer_id
        
    
        
    def feedforward(self, input_, W, b):
        
        if self.layer_id < config.nn_layers - 1:       
            self.z = np.matmul(input_, W) + b
            self.a = self.actFun(self.z)
        
        if self.layer_id == config.nn_layers - 1:
            self.z = np.matmul(input_, W) + b
            exp_scores = np.exp(self.z)
            self.a = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        return self.a
        
    def backprop(self, input_, num_examples, delta):
            
            dW = 1/num_examples * np.dot(np.transpose(input_),delta)
            db = 1/num_examples * np.sum(delta, axis = 0, keepdims = True)
            
            return dW, db



