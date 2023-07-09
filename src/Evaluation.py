#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 02:07:19 2020

@author: zhanghaonan
"""


import logging
import pickle

import numpy as np
import pandas as pd
import yaml
import sklearn
from sklearn import model_selection
from sklearn import linear_model
#from sklearn.metrics import confusion_matrix,classification_report,accuracy_score,roc_auc_score,f1_score

#from helper import load_csv

logger = logging.getLogger(__name__)



def evaluate_model(df, y_predicted, target_column, **kwargs):
    """Evaluate the performance of the model   
    Args:
        df (:py:class:`pandas.DataFrame`): test Dataframe which containing true y label
        target_column (str): column name of target column
        y_predicted (:py:class:`pandas.DataFrame`): Dataframe containing predicted probability and score
    
    Returns: 
        confusion_df (:py:class:`pandas.DataFrame`): Dataframe reporting confusion matrix
        metrics (:py:class:`pandas.DataFrame`): Dataframe reporting metrics including AUC, accuracy and f1-score
        
    """
    
    try:
        # get predicted scores
        y_pred_prob = y_predicted.loc[:,"predicted_proba"]
        y_pred = y_predicted.loc[:,"predicted_class"]
        # get true labels
        y_test = df.loc[:,target_column]
    
    except:
        raise IndexError('Index out of bounds!')
    ## condition we want to unit test 
    if len(y_pred)!=len(y_test):
        raise IndexError('Index out of bounds!')          
        # calculate auc and accuracy and f1_score if specified
    if "auc" in kwargs["metrics"]:
        auc = sklearn.metrics.roc_auc_score(y_test, y_pred_prob)
        print('AUC on test: %0.3f' % auc)
    if "accuracy" in kwargs["metrics"]:
        accuracy = sklearn.metrics.accuracy_score(y_test, y_pred)
        print('Accuracy on test: %0.3f' % accuracy)
    if "f1_score" in kwargs["metrics"]:
        f1 = sklearn.metrics.f1_score(y_test, y_pred)
        print('F1-score on test: %0.3f' % f1)
    
#     ## evaluate the performance
    confusion = sklearn.metrics.confusion_matrix(y_test, y_pred)

    
    print("-------------- Model Performance Evaluation-------------------")
    print('AUC on test: %0.3f' % auc)
    print('Accuracy on test: %0.3f' % accuracy)
    print(pd.DataFrame(confusion,
                  index=['Actual negative','Actual positive'],
                  columns=['Predicted negative', 'Predicted positive']))
    
    ## save confusion matrix
    confusion_df = pd.DataFrame(confusion,
        index=['Actual Negative','Actual Positive'],
        columns=['Predicted Negative', 'Predicted Positive'])
    
    ## save metrics performance as csv
    metrics = pd.DataFrame([auc,accuracy,f1],index =["auc","accuacy","f1"],columns=["Evaluation Metric"])

    return confusion_df, metrics



