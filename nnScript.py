#!/usr/bin/env python
from __future__ import print_function
import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
import sys
import pdb

# How to run jobs in Metallica https://piazza.com/class/ii0wz7uvsf112m?cid=115
def initializeWeights(n_in,n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
       
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""
    
    epsilon = sqrt(6) / sqrt(n_in + n_out + 1);
    W = (np.random.rand(n_out, n_in + 1)*2* epsilon) - epsilon;
    return W
    
def sigmoid(z):
    
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""
    
    return 1.0 / (1.0 + np.exp(-1.0 * np.array(z))) #Calculates sigmoid of each element in the scalar/vector/matrix

def preprocess():
    """ Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set

     Some suggestions for preprocessing step:
     - divide the original data set to training, validation and testing set
           with corresponding labels
     - convert original data set from integer to double by using double()
           function
     - normalize the data to [0, 1]
     - feature selection"""
    
    # list to record all the features that are zero for now
    
    mat = loadmat('mnist_all.mat') #loads the MAT object as a Dictionary

    # Data partition into the validation and training matrix - https://piazza.com/class/ii0wz7uvsf112m?cid=139

    #Pick a reasonable size for validation data
    
    train_data = np.array([])
    train_label = np.array([])
    validation_data = np.array([])
    validation_label = np.array([])
    test_data = np.array([])
    test_label = np.array([])

    try:
     for key, value in mat.iteritems():
      if not key.startswith("__"):  # 'test' in key or 'train'  in key:
        #print("number of samples for ",key," : ",len(value))
        featureData = np.random.permutation(value.astype(np.float64)/255.0)
        num = int(key[-1])
        numSamples = value.shape[0]
        labelData = np.zeros((numSamples,10),dtype = np.uint8)
        trueLabel = np.ones(numSamples)
        labelData[:,num] = trueLabel
        #print("numSamples: ",numSamples," labelData: ",labelData.shape,"featureData: ",featureData.shape," type: ",key)
        #print("featureData size:",featureData.shape)
        if 'test' in key:
          test_data = featureData if test_data.size == 0 else np.vstack([test_data, featureData])
          test_label = labelData if test_label.size == 0 else np.vstack([test_label, labelData])
        elif 'train' in key:
          validation_data = featureData[:1000,:] if validation_data.size == 0 else np.vstack([validation_data,featureData[:1000]])
          validation_label = labelData[:1000,:] if validation_label.size == 0 else np.vstack([validation_label,labelData[:1000]])
          train_data = featureData[1000:,:] if train_data.size == 0 else np.vstack([train_data, featureData[1000:]])
          train_label = labelData[1000:,:] if train_label.size == 0 else np.vstack([train_label, labelData[1000:]])
    except:
      e = sys.exc_info()
      print("error: ",e)
      pass
    print("train_data: ",train_data.shape," train_label: ",train_label.shape)
    print("test_data: ",test_data.shape," test_label: ",test_label.shape)
    print("validation_data: ",validation_data.shape," validation_label: ",validation_label.shape)
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
    obj_val = 0  
    
    train_with_ones = np.column_stack([training_data, np.ones(training_data.shape[0])])    
    z1 =  sigmoid(np.dot(train_with_ones,w1.T))
    print("z1 shape: ",z1.shape)
    z1 = np.column_stack([z1, np.ones(z1.shape[0])])
    o1 = sigmoid(np.dot(z1, w2.T))
    print("z1: ",z1.shape," o1: ",o1.shape)
    y_ol_diff = training_label - o1
    J = np.sum(np.sum(np.square(y_ol_diff),axis=1)*0.5) * (1.0/n_input)
    # this will be zero till all the lambda value is initialised 
    lamb = float(lambdaval) / (2.0 * n_input)

    obj_val = J + (lamb * (np.sum(np.square(w1)) + np.sum(np.square(w2))))
    
    # This is vector calculation

    errorL = y_ol_diff * (1 - o1) * o1
    
    pdb.set_trace()
    grad_w2 = (-errorL) * w2



    #Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    #you would use code similar to the one below to create a flat array
    #obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    obj_grad = np.array([])
    
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


    # creates a column containing only 1's with number of rows as size of feature vector of image
    # This is for introducing the d+1 node in the input layer
    ones_column = np.ones((np.array(data).shape[0], 1), dtype = int)
    # Append a new column to existing matrix
    data = np.column_stack([data, ones_column])
    #Calculate z = w1T.x at hidden layer and apply the sigmoid function
    z = sigmoid(np.dot(data, w1.transpose()))


    #Column of 1's for simulating the bias node at hidden layer
    one_column2 = np.ones((np.array(z).shape[0],1), dtype = int)
    # Append a new column to existing matrix so that we add the bias node
    z = np.column_stack([z, one_column2])
    #Calculate o = w2T.z at output layer and apply the sigmoid function
    o = sigmoid(np.dot(z, w2.transpose()))


    #Calculate the max in every row which gives the actual digit recognized
    ind_matrix = np.argmax(o, axis = 1)


    # Create a matrix of zeros to create the label later
    res_matx = np.zeros(np.shape(ind_matrix), dtype = int)


    #Counter to keep track of index of column with max value in each row of indices matrix indMatx
    i = 0


    #Update the corresponding value in each row's index to '1' leaving the others as '0'
    #so that we label the output to one of the digits 0-9 for each row
    for row in range(res_matx.shape[0]):
        res_matx[row][ind_matrix[i]] = 1
        i += 1

    labels = np.array(res_matx)
    # Your code here

    # Your solution should be able to consider each row as the input vector x and do the forward pass of the
    # neural network and output the class for which the output (ol) is maximum.

    # Related Piazza posts
    # https://piazza.com/class/ii0wz7uvsf112m?cid=128

    return labels
    
"""**************Neural Network Script Starts here********************************"""

train_data, train_label, validation_data,validation_label, test_data, test_label = preprocess();


#  Train Neural Network

# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]; 

# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 50;
                   
# set the number of nodes in output unit
n_class = 10;                  

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden);
initial_w2 = initializeWeights(n_hidden, n_class);

# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()),0)

# set the regularization hyper-parameter
lambdaval = 0;

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

# Test the computed parameters

predicted_label = nnPredict(w1,w2,train_data)

#find the accuracy on Training Dataset

print('\n Training set Accuracy:' + str(100*np.mean((predicted_label == train_label).astype(float))) + '%')

predicted_label = nnPredict(w1,w2,validation_data)

#find the accuracy on Validation Dataset

print('\n Validation set Accuracy:' + str(100*np.mean((predicted_label == validation_label).astype(float))) + '%')


predicted_label = nnPredict(w1,w2,test_data)

#find the accuracy on Validation Dataset

print('\n Test set Accuracy:' + + str(100*np.mean((predicted_label == test_label).astype(float))) + '%')
