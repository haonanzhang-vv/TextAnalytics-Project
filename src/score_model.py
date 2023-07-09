#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 23:40:48 2020

@author: zhanghaonan
"""

import logging
import pickle
import yaml

import numpy as np
import pandas as pd
import yaml
import sklearn
from sklearn import model_selection
from sklearn import linear_model

#from helper import load_csv

logger = logging.getLogger(__name__)



def score_model(transformed_feature,thres, path_to_tmo=None, **kwargs):
    """Get prediction results for the test set.
    Args:
        df (:py:class:`pandas.DataFrame`): Dataframe containing data to run prediction on.(test.csv data)
        transformed_features (sparse matrix): sparse text matrix for test data 
        thres (float) : threshold probability for binary classification 
        path_to_tmo (str): Path to trained model.

    
    Returns:
        df_prediction (:py:class:`pandas.DataFrame`): DataFrame containing predicted scores.
    
    """
    # load the model saved from previous step
    try:
         with open(path_to_tmo, "rb") as f:
             model = pickle.load(f)
    except Exception as e:
         logger.error(e)
         logger.error("Failure to load the model from the directory.")
    

    
    ## generate prediction with handling wrong/inconsistent input of chosen features
    try:
        ypred_proba_test = model.predict_proba(transformed_feature)[:,1]

        ypred_bin_test = np.where(model.predict_proba(transformed_feature)[:,1] > thres,1,0)
    
    except:
        raise ValueError("Test data features dimension does not match the model training feature dimension")
        logger.error("Test data features dimension does not match the model training feature dimension")
    
    ## output the prediction
    df_prediction = pd.DataFrame([ypred_proba_test,ypred_bin_test],index = ["predicted_proba","predicted_class"]).T

    return df_prediction
    