#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 09:51:56 2017

@author: py
"""

__author__ = 'py'
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from Layer import Layer
from configs import configs

config = configs()
        
def generate_data():
    '''
    generate data
    :return: X: input data, y: given labels
    '''
    np.random.seed(0)
    X, y = datasets.make_moons(200, noise=0.20)
    return X, y

def plot_decision_boundary(pred_func, X, y):
    '''
    plot the decision boundary
    :param pred_func: function used to predict the label
    :param X: input data
    :param y: given labels
    :return:
    '''
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    plt.show()


class NeuralNetwork(object):
    """
    This class builds and trains a neural network
    """
    def __init__(self, X, y, seed=0):
    
        
        # initialize the weights and biases in the network
        np.random.seed(seed)
        self.layers = [Layer(i, 
                             lambda x: self.actFun(x, type=config.actFun_type)
                             ) for i in range(config.nn_layers)]
        self.W = [np.random.randn(config.nn_hidden_dim, config.nn_hidden_dim
                                  )/ np.sqrt(config.nn_hidden_dim)] * config.nn_layers
        self.W[0] = np.random.randn(config.nn_input_dim, config.nn_hidden_dim
                                  )/ np.sqrt(config.nn_input_dim)
        self.W[-1] =  np.random.randn(config.nn_hidden_dim, config.nn_output_dim
                                  )/ np.sqrt(config.nn_hidden_dim)
        self.b = [np.zeros((1, config.nn_hidden_dim))] * config.nn_layers
        self.b[-1] = np.zeros((1, config.nn_output_dim))
        self.X = X
        self.y = y
        self.neurons = [0] * (config.nn_layers + 1)
        self.neurons[0] = X
        self.delta = [0] * config.nn_layers       
        self.dW = [0] * config.nn_layers
        self.db = [0] * config.nn_layers

    def actFun(self, z, type):
        '''
        actFun computes the activation functions
        :param z: net input
        :param type: Tanh, Sigmoid, or ReLU
        :return: activations
        '''

        # YOU IMPLMENT YOUR actFun HERE
        if type == 'tanh':
            act = np.tanh(z)
        if type == 'sigmoid':
            act = 1 / (1 + np.exp(-z))
        if type == 'relu':
            act = z * (z > 0)

        return act

    def diff_actFun(self, z, type):
        '''
        diff_actFun computes the derivatives of the activation functions wrt the net input
        :param z: net input
        :param type: Tanh, Sigmoid, or ReLU
        :return: the derivatives of the activation functions wrt the net input
        '''

        # YOU IMPLEMENT YOUR diff_actFun HERE
        if type == 'tanh':
            diff = 1 - np.power(np.tanh(z),2)
        if type == 'sigmoid':
            sig = 1 / (1 + np.exp(-z))
            diff = sig * (1 - sig)
        if type == 'relu':
            diff = 1 * (z > 0) 

        return diff

    def feedforward(self):
        '''
        update the neurons in all layers
        '''
        
        for i, layer in enumerate(self.layers):
            self.neurons[i+1] = layer.feedforward(self.neurons[i], self.W[i], self.b[i])

        return None

    def calculate_loss(self):
        '''
        calculate_loss computes the loss for prediction
        :param X: input data
        :param y: given labels
        :return: the loss for prediction
        '''
        num_examples = len(self.X)
        self.feedforward()
        # Calculating the loss

        data_loss = -np.sum(np.log(self.neurons[-1])[np.arange(num_examples),
                            self.y])/num_examples

        # Add regulatization term to loss (optional)
        data_loss += config.reg_lambda / 2 * np.sum([np.sum(np.square(W)) for W 
                                                   in self.W] )
        return (1. / num_examples) * data_loss

    def predict(self, X, type_):
        '''
        predict infers the label of a given data point X
        :param X: input data
        :return: label inferred
        '''
        if type_ == 'train':
            
            self.feedforward()
            pred = np.argmax(self.neurons[-1], axis=1)
        else:
            
            neurons = [X]
            for i, layer in enumerate(self.layers):
                neurons.append(layer.feedforward(neurons[i], self.W[i], self.b[i]))   
            pred = np.argmax(neurons[-1], axis = 1)
            
        return pred

    def backprop(self):
        
        for i, layer in enumerate(self.layers[::-1]):
            
            index = config.nn_layers - 1 - i
            
            self.dW[index], self.db[index] = layer.backprop(self.neurons[index],
                   len(self.X), self.delta[index])
        
        return None

    def fit_model(self, epsilon=0.01, num_passes=20000, print_loss=True):
        '''
        fit_model uses backpropagation to train the network
        :param X: input data
        :param y: given labels
        :param num_passes: the number of times that the algorithm runs through the whole dataset
        :param print_loss: print the loss or not
        :return:
        '''
        # Gradient descent.
        for j in np.arange(0, num_passes):
            # Forward propagation
            self.feedforward()
            # Updata deltas
            self.neurons[-1][range(len(self.X)), self.y] -= 1
            self.delta[-1] = np.array(self.neurons[-1])
            for i in np.arange(config.nn_layers)[-2::-1]:
                self.delta[i] = np.multiply(np.dot(self.delta[i+1], self.W[i+1].T), 
                          self.diff_actFun(self.neurons[i+1], config.actFun_type))
            
            # Backpropagation
            self.backprop()

            # Add regularization terms (b1 and b2 don't have regularization terms)
            self.dW = [dw + config.reg_lambda * w for w, dw in zip(self.W, self.dW)]

            # Gradient descent parameter update
            self.W = [w - epsilon * dw for w, dw in zip(self.W, self.dW)]
            self.b = [b - epsilon * db for b, db in zip(self.b, self.db)]
            
            if print_loss and j % 1000 == 0:
                print("Loss after iteration {}:{}".format(j, self.calculate_loss()))

    def visualize_decision_boundary(self):
        '''
        visualize_decision_boundary plots the decision boundary created by the trained network
        :param X: input data
        :param y: given labels
        :return:
        '''
        plot_decision_boundary(lambda x: self.predict(x, 'test'), self.X, self.y)

def main():
     # generate and visualize Make-Moons dataset
     X, y = generate_data()
     plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)
     plt.show()

     model = NeuralNetwork(X,y)
     model.fit_model()
     model.visualize_decision_boundary()

if __name__ == "__main__":
    main()