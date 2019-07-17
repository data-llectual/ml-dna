#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 19:04:11 2019

@author: raviraisinghani
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets, linear_model



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
  
    #Now we have converted the data string to datatime - sort it 
    goog = goog.sort_values(['Date'],ascending=[True])
  
    # Calculate the returns for the adjusted closed col 
    # here use pct_change() method from pandas - make sure we generate pct change only for float and int type as 
    # we do not want to generate anythig for date
  

    returns = goog[[key for key in dict(goog.dtypes) if dict(goog.dtypes)[key] in ['float64', 'int64']]].pct_change()
  
    return returns
  
    

def build_training_data():
    #call read_goog_sp500_data() method 
    
    pct_change_dataframe = read_goog_sp500_data()
    x_train_data = np.array(pct_change_dataframe["SP500"])[1:]
    y_train_data = np.array(pct_change_dataframe["Goog"])[1:]
    
    return (x_train_data,y_train_data)

def build_model_and_train():
    #call the training data method 
    x_train_data, y_train_data = build_training_data() 
    
    

def main():
    read_goog_sp500_data()
    print('Here main')
    


if __name__ == '__main__':
 main()