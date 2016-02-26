#!/usr/bin/python
from __future__ import print_function
import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
from sys import argv
from scipy.special import expit

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
    mat = loadmat('mnist_all.mat') #loads the MAT object as a Dictionary
    # Data partition into the validation and training matrix - https://piazza.com/class/ii0wz7uvsf112m?cid=139
    #Pick a reasonable size for validation data
    validation_size = 100
    max_value = 255.0
    n_output = 10
    train_data = np.array([])
    train_label = np.array([])
    validation_data = np.array([])
    validation_label = np.array([])
    test_data = np.array([])
    test_label = np.array([])
    for key, value in mat.iteritems():
      if not key.startswith("__"):
        featureData = np.random.permutation(value.astype(np.float64)/max_value)
        num = int(key[-1])
        numSamples = value.shape[0]
        labelData = np.zeros((numSamples,n_output),dtype = np.uint8)
        trueLabel = np.ones(numSamples)
        labelData[:,num] = trueLabel
        if 'test' in key:
          test_data = featureData if test_data.size == 0 else np.vstack([test_data, featureData])
          test_label = labelData if test_label.size == 0 else np.vstack([test_label, labelData])
        elif 'train' in key:
          validation_data = featureData[:validation_size,:] if validation_data.size == 0 else np.vstack([validation_data,featureData[:validation_size]])
          validation_label = labelData[:validation_size,:] if validation_label.size == 0 else np.vstack([validation_label,labelData[:validation_size]])
          train_data = featureData[validation_size:,:] if train_data.size == 0 else np.vstack([train_data, featureData[validation_size:]])
          train_label = labelData[validation_size:,:] if train_label.size == 0 else np.vstack([train_label, labelData[validation_size:]])
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
    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args
    w1 = params[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    
    train_data_ones = np.column_stack([training_data, np.ones(training_data.shape[0],dtype = np.float64)]) 
    n_samples = training_data.shape[0]   

    z1 =  sigmoid(np.dot(train_data_ones,w1.T))
    z1 = np.column_stack([z1, np.ones(z1.shape[0], dtype = np.float64)])
    o1 = sigmoid(np.dot(z1, w2.T))

    y_ol_diff = training_label - o1
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
    ones_column = np.ones((np.array(data).shape[0], 1), dtype = int)
    data = np.column_stack([data, ones_column])
    z = sigmoid(np.dot(data, w1.transpose()))
    one_column2 = np.ones((np.array(z).shape[0],1), dtype = int)
    z = np.column_stack([z, one_column2])
    o = sigmoid(np.dot(z, w2.transpose()))
    #Calculate the max in every row which gives the actual digit recognized
    ind_matrix = np.argmax(o, axis = 1)
    res_matx = np.zeros((ind_matrix.shape[0],o.shape[1]))
    #Counter to keep track of index of column with max value in each row of indices matrix indMatx
    i = 0
    #Update the corresponding value in each row's index to '1' leaving the others as '0'
    #so that we label the output to one of the digits 0-9 for each row
    for row in range(res_matx.shape[0]):
        res_matx[row][ind_matrix[i]] = 1
        i += 1

    labels = np.array(res_matx)
    # Related Piazza posts: https://piazza.com/class/ii0wz7uvsf112m?cid=128
    return labels

def predictDiff(predicted, actual):
  predictions = {}
  for i in range(10):
    ac = np.count_nonzero(actual[:,i])
    pc = np.count_nonzero(predicted[:,i])
    predictions[str(i)] = { 'actual' : ac, 'predicted' : pc}
  return predictions


"""**************Neural Network Script Starts here********************************"""

train_data, train_label, validation_data,validation_label, test_data, test_label = preprocess()
#  Train Neural Network
# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1] 
# set the number of nodes in hidden unit (not including bias unit)
n_hidden = int(argv[1])
                   
# set the number of nodes in output unit
n_class = train_label.shape[1]                  

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden)
initial_w2 = initializeWeights(n_hidden, n_class)

# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()),0)

# set the regularization hyper-parameter
lambdaval = float(argv[2])
args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

#Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

opts = {'maxiter' : 50}    # Preferred value.

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)

#In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
#and nnObjGradient. Check documentation for this function before you proceed.
#nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)


#Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

gpredictions = {}
key = '_'.join(argv[1:])

temp = {}
# Test the computed parameters
predicted_label = nnPredict(w1,w2,train_data)

#find the accuracy on Training Dataset
accuracy = str(100*np.mean((predicted_label == train_label).astype(float)))
#print('\nTraining set Accuracy:' + accuracy + '%')
temp["Training"] = { 'predictions': predictDiff(predicted_label, train_label), 'accuracy' :accuracy }

predicted_label = nnPredict(w1,w2,validation_data)
accuracy = str(100*np.mean((predicted_label == validation_label).astype(float)))
#find the accuracy on Validation Dataset
#print('\nValidation set Accuracy:' + accuracy + '%')
temp["Validation"] ={ 'predictions' :predictDiff(predicted_label, validation_label), 'accuracy' :accuracy }


predicted_label = nnPredict(w1,w2,test_data)
accuracy = str(100*np.mean((predicted_label == test_label).astype(float)))
#find the accuracy on Validation Dataset
#print('\nTest set Accuracy:' + accuracy + '%')
temp["Testing"] ={ 'predictions' : predictDiff(predicted_label, test_label), 'accuracy' :accuracy }
gpredictions[key] = temp
print("\n",gpredictions,"\n")