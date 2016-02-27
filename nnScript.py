#!/usr/bin/python
from __future__ import print_function
import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
from sys import argv
import json
from scipy.special import expit
from itertools import product
from time import process_time as ptime
from time import time
import datetime

train_matrix_label = None

def initializeWeights(n_in,n_out):
    epsilon = sqrt(6) / sqrt(n_in + n_out + 1)
    W = (np.random.rand(n_out, n_in + 1)*2* epsilon) - epsilon
    return W
    
def sigmoid(z):
    return expit(z) #1.0 / (1.0 + np.exp(-1.0 * np.array(z)))

def featureSelection(train_data, validation_data, test_data):
    num_features = train_data.shape[1]
    assert(validation_data.shape[1] == num_features)
    assert(test_data.shape[1] == num_features)
    column_indexes = np.arange(num_features)
    same_bool = np.all(train_data == train_data[0,:], axis = 0)
    train_same = column_indexes[same_bool]
    same_bool = np.all(validation_data == validation_data[0,:], axis = 0)
    val_same = column_indexes[same_bool]
    same_bool = np.all(test_data == test_data[0,:], axis = 0)
    test_same = column_indexes[same_bool]
    common_columns = np.intersect1d(np.intersect1d(train_same, val_same,True), test_same, True)
    train_data = np.delete(train_data,common_columns,axis=1)
    validation_data = np.delete(validation_data,common_columns,axis=1)
    test_data = np.delete(test_data,common_columns,axis=1)
    num_features = train_data.shape[1]
    assert(validation_data.shape[1] == num_features)
    assert(test_data.shape[1] == num_features)
    return train_data, validation_data, test_data

def preprocess():
    global train_matrix_label
    mat = loadmat('mnist_all.mat') #loads the MAT object as a Dictionary
    # Data partition into the validation and training matrix - https://piazza.com/class/ii0wz7uvsf112m?cid=139
    #Pick a reasonable size for validation data
    validation_size = 500
    max_value = 255.0
    train_data = np.array([],dtype=np.float)
    train_label = np.array([],dtype=np.uint8)
    validation_data = np.array([],dtype=np.float)
    validation_label = np.array([],dtype=np.uint8)
    test_data = np.array([],dtype=np.float)
    test_label = np.array([],dtype=np.uint8)
    train_matrix_label = np.array([],dtype = np.uint8)
    for key, value in mat.items():
      if not key.startswith("__"):
        featureData = np.random.permutation(value.astype(np.float64)/max_value)
        numSamples = value.shape[0]
        result = np.uint8(int(key[-1]))
        labelData = np.empty(numSamples,dtype = np.uint8)
        labelData.fill(result)
        matrix_label = np.zeros((numSamples - validation_size,10),dtype = np.uint8)
        matrix_label[:,result] = 1
        if 'test' in key:
          test_data = featureData if test_data.size == 0 else np.vstack([test_data, featureData])
          test_label = np.append(test_label, labelData)
        elif 'train' in key:
          validation_data = featureData[:validation_size,:] if validation_data.size == 0 else np.vstack([validation_data,featureData[:validation_size]])
          train_data = featureData[validation_size:,:] if train_data.size == 0 else np.vstack([train_data, featureData[validation_size:]])
          validation_label = np.append(validation_label, labelData[:validation_size])
          train_label = np.append(train_label, labelData[validation_size:])
          train_matrix_label = matrix_label if train_matrix_label.size == 0 else np.vstack([train_matrix_label,matrix_label])
    train_data, validation_data, test_data = featureSelection(train_data, validation_data, test_data)
    return train_data, train_label, validation_data, validation_label, test_data, test_label

# Where do the weights get updated? https://piazza.com/class/ii0wz7uvsf112m?cid=117
# https://piazza.com/class/ii0wz7uvsf112m?cid=116
def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log 
    %   likelihood error function with regularization) given the parameters 
    %   of Neural Networks, thetraining data, their corresponding training 
    %   labels and lambda - regularization hyper-parameter.
    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.
    % Output: 
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input 
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden 
    %     layer to unit i in output layer."""

    global train_matrix_label
    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args
    w1 = params[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    
    train_data_ones = np.column_stack([training_data, np.ones(training_data.shape[0],dtype = np.float64)]) 
    n_samples = training_data.shape[0]   

    z1 =  sigmoid(np.dot(train_data_ones,w1.T))
    z1 = np.column_stack([z1, np.ones(z1.shape[0], dtype = np.float64)])
    o1 = sigmoid(np.dot(z1, w2.T))

    y_ol_diff = train_matrix_label - o1
    J = np.sum(np.sum(np.square(y_ol_diff),axis=1)*0.5) * (1.0/n_samples)
    lamb = lambdaval / (2.0 * n_samples)
    obj_val = J + lamb * (np.square(w1).sum() + np.square(w2).sum()) 
    
    # This is vector calculation
    deltaL = y_ol_diff * (1 - o1) * o1
    grad_w2 = np.add(np.dot(-deltaL.T, z1), lambdaval * w2) * (1.0/n_samples)
    temp_sum = np.dot(deltaL, w2)
    zmat = np.multiply(-(1-z1), z1)
    res_mat = np.multiply(zmat, temp_sum)
    res_mat = res_mat[:, :-1]

    p1 = np.dot(res_mat.T,  train_data_ones)
    p2 = (lambdaval * w1)
    grad_w1 = (p1 + p2)/float(n_samples)
    obj_grad = np.concatenate((np.array(grad_w1).flatten(), np.array(grad_w2).flatten()),0)
    return (obj_val,obj_grad)

def nnPredict(w1,w2,data):
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.
    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit i in hidden
    %     layer to unit j in output layer.
    % data: matrix of data. Each row of this matrix represents the feature 
    %       vector of a particular image
       
    % Output: 
    % label: a column vector of predicted labels"""
    ones_column = np.ones((np.array(data).shape[0], 1), dtype = np.uint8)
    data = np.column_stack([data, ones_column])
    z = sigmoid(np.dot(data, w1.transpose()))
    one_column2 = np.ones((np.array(z).shape[0],1), dtype = np.uint8)
    z = np.column_stack([z, one_column2])
    o = sigmoid(np.dot(z, w2.transpose()))
    #Calculate the max in every row which gives the actual digit recognized
    # Related Piazza posts: https://piazza.com/class/ii0wz7uvsf112m?cid=128
    return np.argmax(o, axis = 1)

"""**************Neural Network Script Starts here********************************"""
    
fileName = "_".join(["out",argv[1],argv[2],str(int(time())),".out"])
f = open(fileName,"w")
f.write("Start: "+str(datetime.datetime.now())+"\n")
start = ptime()
train_data, train_label, validation_data,validation_label, test_data, test_label = preprocess()
n_input = train_data.shape[1]
n_hidden = int(argv[1])
n_class = 10
lambdaval = float(argv[2])
f.write("Hidden count: "+argv[1]+" lambda: "+argv[2]+"\n")
print("Hidden count: "+argv[1]+" lambda: "+argv[2])
# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden)
initial_w2 = initializeWeights(n_hidden, n_class)

# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()),0)
args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)
opts = {'maxiter' : 50}    # Preferred value.
nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)
#Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

# Test the computed parameters
predicted_label = nnPredict(w1,w2,train_data)

#find the accuracy on Training Dataset
accuracy = str(100*np.mean((predicted_label == train_label).astype(float)))

f.write( "Train_label: "+','.join([str(x) for x in train_label.tolist()])+"\n")
f.write("predicted_label: "+','.join([str(x) for x in predicted_label.tolist()]) + "\n")
f.write("Accuracy: "+accuracy+"\n")
f.write("Train_label_count:"+','.join([str(x) for x in np.bincount(train_label).tolist()])+"\n")
f.write("predicted_label_count:"+','.join([str(x) for x in np.bincount(predicted_label).tolist()])+"\n")
print('\nTraining set Accuracy:' + accuracy + '%')

#find the accuracy on Validation Dataset
predicted_label = nnPredict(w1,w2,validation_data)
accuracy = str(100*np.mean((predicted_label == validation_label).astype(float)))
f.write( "Train_label: "+','.join([str(x) for x in train_label.tolist()])+"\n")
f.write("predicted_label: "+','.join([str(x) for x in predicted_label.tolist()]) + "\n")
f.write("Accuracy: "+accuracy+"\n")
f.write("Validation_label_count:"+','.join([str(x) for x in np.bincount(validation_label).tolist()])+"\n")
f.write("predicted_label_count:"+','.join([str(x) for x in np.bincount(predicted_label).tolist()])+"\n")

print('\nValidation set Accuracy:' + accuracy + '%')

predicted_label = nnPredict(w1,w2,test_data)
accuracy = str(100*np.mean((predicted_label == test_label).astype(float)))
f.write("Test_label: "+','.join([str(x) for x in train_label.tolist()])+"\n")
f.write("predicted_label: "+','.join([str(x) for x in predicted_label.tolist()]) + "\n")
f.write("Accuracy: "+accuracy+"\n")
f.write("Test_label_count:"+','.join([str(x) for x in np.bincount(test_label).tolist()])+"\n")
f.write("predicted_label_count:"+','.join([str(x) for x in np.bincount(predicted_label).tolist()])+"\n")
print('\nTest set Accuracy:' + accuracy + '%')
f.write("End: "+str(datetime.datetime.now())+"\n")
f.write("Time_cousumed: "+str(ptime()-start)+"\n")
f.close()
