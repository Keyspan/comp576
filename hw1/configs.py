#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 15:45:19 2017

@author: py
"""

class configs():
    
    def __init__(self):
        
        self.nn_input_dim = 2
        
        self.nn_hidden_dim = 3
        
        self.nn_output_dim = 2
        
        self.actFun_type = 'tanh'
        
        self.nn_layers = 50
        # the nn_layers here are number of hidden_layer plus output_layer
        
        self.reg_lambda = 0.01