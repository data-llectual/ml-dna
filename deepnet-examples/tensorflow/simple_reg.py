#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 07:41:30 2019


"""

import numpy as np
import tensorflow as tf 
import matplotlib.pyplot as plt
import pandas as pd
import math 


def create_training_dataframe():
    
    global input_data
    input_x = np.linspace(0.0,10.0,1000000) 
    
    noise_term = np.random.randn(len(input_x))

    #Let us create a linear fit model
    
    #y = mx + b
    
    m= 0.5
    b= 5
    y_true_label = (0.5*input_x) +b + noise_term
    
    #There are multiple ways to create a dataset 
    #let us explore pandas dataframe 
    
    x_df = pd.DataFrame(data=input_x, columns=['X Data'])
    y_df = pd.DataFrame(data=y_true_label, columns =['Y Label'])
    
    #concat these datafame along the column which is axis 1 
    
    input_data = pd.concat([x_df,y_df], axis=1)
    

    
    # use the pandas to plot the scatter 
    input_data.sample(n=250).plot(kind='scatter', x='X Data', y='Y Label')
   
    return input_data

def main():
    print('This is main function')
    
    #define and initialize Hyperparameters 
    global NUMBER_EPOCH
    global BATCH_SIZE
   
    NUMBER_EPOCH=1
    BATCH_SIZE =32
    
    input_training_data = create_training_dataframe()
    build_and_train_tensorflow_model(input_training_data)

def build_and_train_tensorflow_model(input_training_data):
     # Here W is b are Tensorflow Variables  Variables needs to be initalized 
     TRAIN_SIZE = len(input_training_data)
     
     global x_train_data 
     global y_train_data
     global x
     global y_actual
     global W
     global b
     global objective_function
     #get x and y data from dataFrame that is being passed to this function
     x_train_data=input_training_data.iloc[:,0].values.reshape(-1,1)
     y_train_data = input_training_data.iloc[:,1].values.reshape(-1,1)
     
     W = tf.Variable(tf.zeros([1,1]),name='W')
     b = tf.Variable(tf.zeros([1]), name='b')
     
     
     #Define placeholder for x and y
     x = tf.placeholder(tf.float32, [None,1], name='x')
     y_actual = tf.placeholder(tf.float32, [None,1], name='y_actual')
     
     Wx = tf.matmul(x,W)
    
     #Linear model in tensorflow - to be computed y_predicted 
     y_predicted = Wx+b
     
     
     #define objectivefunction 
     objective_function = tf.reduce_mean(tf.square(y_actual-y_predicted))
     
     #Define the optimizer to optimize the objective function 
     # learning rate for ftrl optimizer is 1 - this may not be good learing rate 
     # as there would be be lot of volatility searching the solution in the search space 
     optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(objective_function)
     
     # Call function that takes - number of epoch, optimizer and         
     train_model_with_minibatch(TRAIN_SIZE,NUMBER_EPOCH,BATCH_SIZE,optimizer)  
     
     
        
def train_model_with_minibatch(train_size,epoch,batch_size,optimizer):
    
    NUMBER_BATCHES = math.ceil(train_size/batch_size)
    print('Number of batches ', NUMBER_BATCHES)
    #define init node
    init = tf.global_variables_initializer()
    
    with tf.Session() as sess:
        #Need to initialize global variable 
        sess.run(init)
        
        for i in range(NUMBER_EPOCH):
            
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
                    
        print('Model slope predicted by Gradient Descent ', sess.run(W) )      
        print('Model intercept predicted by Gradient Descent ', sess.run(b))
        
        print(sess.run(W)[0,0])
        
        predicted_model_slope = sess.run(W)[0,0]
        predicted_model_intercept = sess.run(b)[0]
        
        
        #let us do some prediction using the fitted model using GD 
        y_prediction = x_train_data*predicted_model_slope + predicted_model_intercept 
        
        #plot th data and the line 
        
        input_data.sample(n=250).plot(kind='scatter', x='X Data', y='Y Label')
        plt.plot(x_train_data,y_prediction,'r')
        plt.show()
        
        
if __name__ == '__main__':
    main()
