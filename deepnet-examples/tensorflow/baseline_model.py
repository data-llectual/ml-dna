#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 19:04:11 2019

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
  

    # Calculate the returns for the adjusted closed col 
    # here use pct_change() method from pandas - make sure we generate pct change only for float and int type as 
    # we do not want to generate anythig for date
  

    returns = goog[[key for key in dict(goog.dtypes) if dict(goog.dtypes)[key] in ['float64', 'int64']]].pct_change()
  
    return returns
  
    

def build_training_data():
    #call read_goog_sp500_data() method 
    
    pct_change_dataframe = read_goog_sp500_data()
    #print('frame ',pct_change_dataframe["SP500"])
    #print('frame 2 ',np.array(pct_change_dataframe["SP500"]))
    #We should ignore only first one 
    #Becasue of the bad data - need to ignore two recorsd
    x_train_data = np.array(pct_change_dataframe["SP500"][2:])
    y_train_data = np.array(pct_change_dataframe["Goog"][2:])
    print(x_train_data)
    print(y_train_data)
    
    return (x_train_data,y_train_data)

def build_model_and_train_baseline():
    #call the training data method 
    x_train_data, y_train_data = build_training_data() 
    #intantiate linear model for google 
    linear_model_google = linear_model.LinearRegression()
    
    #Note that the x data as expected by the linear_model is array of array 
    #reshape the same use np.reshape(-1,1)
    print(x_train_data.reshape(-1,1))
    linear_model_google.fit(x_train_data.reshape(-1,1),y_train_data.reshape(-1,1))
    
    
    print('Coefficient and the intercept of linear model')
    print('Coeff', linear_model_google.coef_)
    print('Intercept', linear_model_google.intercept_)    

def main():
    #This main function will just call one method
    build_model_and_train_baseline()
    


if __name__ == '__main__':
 main()