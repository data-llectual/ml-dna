#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 09:14:30 2019

"""


import numpy as np 
import tensorflow as tf
import matplotlib.pyplot as plt


#
def main():
    print('Here main')
     #initialize the variable 
    init = tf.global_variables_initializer()
    tf_initial_tensor_constant = tf.constant(
            [
                [
                    [1,2,3,4],
                    [5,6,7,8],
                    [9,10,11,12]
                ]
                ,
                [
                    [13,14,15,16],
                    [17,18,19,20],
                    [21,22,23,24]
                ]
                    
            ]
            , dtype="int32"
            
            )
    print(tf_initial_tensor_constant)
    
    print("Tf reshape .. ")
    
    tf_ex_one_reshaped_tensor_2_by_12 = tf.reshape(tf_initial_tensor_constant,[2,12])
    print(tf_ex_one_reshaped_tensor_2_by_12)
    
    tf_ex_two_reshaped_tensor_2_by_3_by_2_by_2 = tf.reshape(tf_initial_tensor_constant,[2,3,2,2])
    print(tf_ex_two_reshaped_tensor_2_by_3_by_2_by_2)
    
    print("Convert to Vector ")
    
    tf_ex_thr_reshape_tensor_1_by_24 = tf.reshape(tf_initial_tensor_constant,[-1])
    print(tf_ex_thr_reshape_tensor_1_by_24)
    
    with tf.Session() as sess:
        sess.run(init)
        print("Initail constant ")
        print(sess.run(tf_initial_tensor_constant))
      
        print(sess.run(tf_ex_one_reshaped_tensor_2_by_12))
        print(sess.run(tf_ex_two_reshaped_tensor_2_by_3_by_2_by_2))
        print(sess.run(tf_ex_thr_reshape_tensor_1_by_24)) 

if __name__ == '__main__':
 main()