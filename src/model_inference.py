#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 02:16:41 2020

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
import matplotlib.pyplot as plt
import seaborn as sns

#from helper import load_csv

logger = logging.getLogger(__name__)


def model_inference(features, path_to_tmo, path_save_plot, **kwargs):
    """Get model inference and feature importance
    Args:
        features (list): list of text characters used as features output of the featurize
        path_to_tmo (str): Path to trained model.
        save_path (str): Path to save prediction and summary results. 
    
    Returns:
        fitted (:py:class:`pandas.DataFrame`): DataFrame containing odd ratio, coefficient value for each feature, descending ordered 
    
    """
    # load the model saved from previous step
    try:
        with open(path_to_tmo, "rb") as f:
                model = pickle.load(f)
    except Exception as e:
        logger.error(e)
        logger.error("Failure to load the model from the directory.")
    
    fitted = pd.DataFrame(index=features)        

    if len(features)!= len(model.coef_[0]):
        raise ValueError("Input feature name list does not correspond to the model coefficients")
    
    fitted['coefs'] = model.coef_[0]
    fitted['odds_ratio'] = fitted.coefs.apply(np.exp)
    ## sort odd ration descending 
    fitted = fitted.sort_values(by='odds_ratio', ascending=False)
    ## show the feature importance plot
    fig,ax = plot_coefficients(model,features)
    fig.savefig(path_save_plot)
    
    return fitted


def plot_coefficients(model, labels):
    coef = model.coef_
    table = pd.Series(coef.ravel(), index = labels).sort_values(ascending=True, inplace=False)
    
    all_ = True
    if len(table) > 20:
        reference = pd.Series(np.abs(coef.ravel()), index = labels).sort_values(ascending=False, inplace=False)
        reference = reference.iloc[:20]
        table = table[reference.index]
        table = table.sort_values(ascending=True, inplace=False)
        all_ = False
        

    fig, ax = fig, ax = plt.subplots()
    table.T.plot(kind='barh', edgecolor='black', width=0.7, linewidth=.8, alpha=0.9, ax=ax)
    ax.tick_params(axis=u'y', length=0) 
    if all_:
        ax.set_title('Estimated coefficients', fontsize=14)
    else: 
        ax.set_title('Estimated coefficients (twenty largest in absolute value)', fontsize=14)
    sns.despine()

    return fig, ax

