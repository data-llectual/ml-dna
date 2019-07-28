#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 19:04:11 2019

"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math 
from sklearn import datasets, linear_model




def main():
   global NUMBER_EPOCH
   global BATCH_SIZE
   
    
   NUMBER_EPOCH=1000
   BATCH_SIZE =32
   batch_sz = 500
   n_batches = 100
  
   
   linear_model_coef, linear_model_intercept = build_model_and_train_baseline()
   print('Coefficient and the intercept of linear model')
   print('Coeff', linear_model_coef)
   print('Intercept', linear_model_intercept)
   
   #Now build Tensorflow model and compare with linear model
   build_and_train_tensorflow_model() 


def read_goog_sp500_data():
    
    googFile = '~/Downloads/GOOG.csv'
    sp500File ='~/Downloads/SP_500.csv'
    
    goog= pd.read_csv(googFile,sep=',',usecols=[0,5], names=['Date','Goog'], header=0)
    
    sp = pd.read_csv(sp500File,sep=',',usecols=[0,5], names=['Date','SP500'],header=0)
  
    # create new colum int 
    goog['SP500'] = sp['SP500']
  
    #Use pandas dateTime to convert to datetime format 
    goog['Date'] = pd.to_datetime(goog['Date'],format='%Y-%m-%d')
  
    # now sort data according to date 
    goog = goog.sort_values(['Date'],ascending=[True])
  

    # Calculate the returns for the adjusted closed col 
    # here use pct_change() method from pandas - make sure we generate pct change only for float and int type as 
    # we do not want to generate anythig for date
    returns = goog[[key for key in dict(goog.dtypes) if dict(goog.dtypes)[key] in ['float64', 'int64']]].pct_change()
  
    return returns
  
    

def build_training_data():
    global len_x_data 
    global TRAIN_SIZE
    #call read_goog_sp500_data() method '
    
    global x_train_data 
    global y_train_data
    
    pct_change_dataframe = read_goog_sp500_data()
    #print('frame ',pct_change_dataframe["SP500"])
    #print('frame 2 ',np.array(pct_change_dataframe["SP500"]))
    #We should ignore only first one 
    #Becasue of the bad data - need to ignore first two record
    x_train_data = np.array(pct_change_dataframe["SP500"][2:])
    y_train_data = np.array(pct_change_dataframe["Goog"][2:])
 
    TRAIN_SIZE = len(x_train_data)
    
    print(TRAIN_SIZE)
   
    return (x_train_data,y_train_data)

def build_model_and_train_baseline():
    #call the training data method 
    x_train_data, y_train_data = build_training_data() 
    #intantiate linear model for google 
    linear_model_google = linear_model.LinearRegression()
    
    #Note that the x data as expected by the linear_model is array of array 
    #reshape the same use np.reshape(-1,1)
    #print(x_train_data.reshape(-1,1))
    linear_model_google.fit(x_train_data.reshape(-1,1),y_train_data.reshape(-1,1))
    
    return (linear_model_google.coef_, linear_model_google.intercept_)


'''
Epoch : one pass through entire set : this includes forward pass + backward pass

batche size: total number training examples present in single batch  -> 
Because it is not optimal to fit all training example in one go 
We create number of batches 

Iteration : number of batches needed to complete one epoch.


example : let us say - 2000 training examples 
so say we divide this whole training corpus into batch of 500 examples 
TOTAL_EXAMPLE =2000
BATCH_SIZE = 500
NUMBER_BATCHES =4 
hence it will take (2000/500) = 4 it will take 4 iterations to complete one epoch.

Hence batch size of 500 and iteration 4 to complete 1 epoch 
TOTAL_EXAMPLES = BATCH_SIZE*iteration 

EXAMPLE :
dataset with 200 samples 
we choose batch size of 5 
number of epoch 1000
batch_size = 200/5  = 40 
hence it will take 40 iteration to complete one epoch

The model weights will be updated after each batch of 5 sample 
This means that one epoch will involve 40 batches or 40 updates to model 
Hence with 1000 epochs the model will be exposed to 40000 batches  
0r 40000 updates to model



This is one neuron operations
'''    


def build_and_train_tensorflow_model():
    #linear model is of form y = W*x+b 
    
    global x
    global y_actual
    global W
    global b
    global objective_function
    # Here W is b are Tensorflow Variables  Variables needs to be initalized 
    
    W = tf.Variable(tf.zeros([1,1]),name='W')
    b = tf.Variable(tf.zeros([1]), name='b')
 
    
    # x is input and hence should be placeholder `
    x = tf.placeholder(tf.float32, [None,1], name='x')
    
    #create operation node 
    Wx = tf.matmul(x,W)
    
    #Linear model in tensorflow - to be computed y_predicted 
    y_predicted = Wx+b
    
    #define placeholder for y actual
    y_actual = tf.placeholder(tf.float32, [None,1], name='y_actual')
    
    #node to compute the loss or cost function
    #cost for the linear model is MSE - Mean squred error
    objective_function  =  tf.reduce_mean(tf.square(y_actual-y_predicted))
    
    
    #Define the optimizer to optimize the objective function 
    # learning rate for ftrl optimizer is 1 - this may not be good learing rate 
    # as there would be be lot of volatility searching the solution in the search space 
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(objective_function)
    
    # Call function that takes - number of epoch, optimizer and         
    train_model_with_minibatch(TRAIN_SIZE,NUMBER_EPOCH,BATCH_SIZE,optimizer)
    
    
    
    

# Now define the function that will actually do the minibatch training for given epochs 
#Note and read this very carefully 
'''
Define this function- with following i/p parameters
1. Epoch
2. Batch size 
3.optimizer

'''    
def train_model_with_minibatch(train_size,epoch,batch_size,optimizer):
    
    NUMBER_BATCHES = math.ceil(train_size/batch_size)
    
    #define init node
    init = tf.global_variables_initializer()
    
    with tf.Session() as sess:
        #Need to initialize global variable 
        sess.run(init)
        
        for i in range(NUMBER_EPOCH):
            
            print('Epoch number ', i)
            #random shuffle the data depending upon the usecase
            for j in range(NUMBER_BATCHES):
                
                
                batch_start_idx = (j*batch_size)
                #print(batch_start_idx)
                if (batch_start_idx+batch_size) > train_size:
                    batch_end_idx = batch_start_idx + (train_size-batch_start_idx)
                else:
                    batch_end_idx = (batch_start_idx +batch_size)
                    
                # reshape the training data as tensorFlow needs it 2D array     
                x_mini_batch = x_train_data[batch_start_idx:batch_end_idx].reshape(-1,1)
                y_mini_batch = y_train_data[batch_start_idx:batch_end_idx].reshape(-1,1)
                
               
                # Create the feed_dict to pass PlaceHolder x, y_actual
                
                feed_dict = {x:x_mini_batch, y_actual:y_mini_batch}
                
                
                #Now train your neural network by calling optimizer
                
                sess.run(optimizer, feed_dict=feed_dict)
                
                #Print results for every 500 Epochs
                
                if(i+1)%500:
                    print('After  Epochs', i)
                    print('W : %f', sess.run(W))
                    print('b : %f', sess.run(b))
                    print('Cost by TF %f', sess.run(objective_function,feed_dict=feed_dict))
                    
                
                    
                
if __name__ == '__main__':
 main()
     
    
