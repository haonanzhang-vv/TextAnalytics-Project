#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 23:05:36 2020

@author: zhanghaonan
"""



import logging
logger = logging.getLogger(__name__)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
import pickle
#from src.featurize import featurize
import yaml



def split_data(data,trainSize = 0.7, randomState = 1, train_save_path=None,**kwargs):
    """This function split the train and test data
    Args:
        data (:py:class:`pandas.DataFrame`): the dataframe including features and labelthat are cleaned and tokenized
        train_size (float): proportion of training size default 0.7
        random_state (int): random_state for performance reproduction
        train_save_path (str): train csv save path, default None
    
    Return:
        train (:py:class:`pandas.DataFrame`): the dataframe of training data including features and labelthat are cleaned and tokenized
        test (:py:class:`pandas.DataFrame`): the dataframe of training data including features and labelthat are cleaned and tokenized
    """
    
    ## if the input data is none, then raise error 
    if data is None:
        raise ValueError("No input data is ready to split.")
    
    try:
        index_train, index_test  = train_test_split(np.array(data.index), train_size=trainSize, 
                                                random_state = randomState, stratify=data['positive'])
    except Exception as e:
        logger.error(e)
        logger.error("Fail to split the original data and check the original data dimensions")
    
    # Write training and test sets 
    train = data.loc[index_train,:].copy()
    test =  data.loc[index_test,:].copy()
    # save the train csv
   # train.to_csv(train_save_path)
    ##test data will be saved by user defined directory
    return train, test


### after splitting data into train and test, you will need to run featurize for train which will generate



def train_model_logistic(data, transformed_feature , Cs = 50, fitIntercept=True, penalty="l2", target_column="positive", save_tmo ="data/sentiment_class_prediction.pkl" , **kwargs):
    """This function will train the logistic regression model by training data
    Args: 
        data (:py:class:`pandas.DataFrame` or :py:class:`numpy.Array`): Training data
        transformed_feature (class 'numpy.float64') : sparse matrix of text features importance indicator
        target_column (str): column name of target
        Cs: (int): the range of model complex parameter that will run for cross validation
        fitIntercept (boolean) : boolean to indicate whether to fit intercept
        penalty (str) : either "l1" or "l2"
        save_tmo (str): Path to save the trained model.
        **kwargs: Should contain arguments for specific requirements of model.
        
    Returns:
        logit ('sklearn.linear_model.logistic.LogisticRegression'): Logistic regression model trained.
    """
    ## fit logistic regression

    ### check whether the transformed spare matrix number of rows is the same as the number of rows in the training data  
    if data.shape[0] !=transformed_feature.shape[0]:
        raise ValueError("Input training features sparse matrix does not match the row number of response.")
                
    y_train = data[target_column]
    
    ## perform cross validation
    logit_l2= LogisticRegressionCV(Cs = Cs, fit_intercept = fitIntercept, penalty= penalty, solver='liblinear', scoring='neg_log_loss')
    logit_l2.fit(transformed_feature, y_train)
    
    ## Refit the model with the best complex parameter 
    logit = LogisticRegression(C = logit_l2.C_[0], penalty=penalty, solver='liblinear')
    logit.fit(transformed_feature, y_train)
    
    # Save the trained model object
    if save_tmo is not None:
        with open(save_tmo, "wb") as f:
            pickle.dump(logit, f)
        logger.info("Trained model object saved to %s", save_tmo)

    return logit




    