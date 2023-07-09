#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 19:20:27 2020

@author: zhanghaonan
"""

import logging
import pandas as pd 
import nltk
from src.clean import process_text, clean
from src.train_model import split_data, train_model_logistic
from src.featurize import featurize


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from num2words import num2words
import string
logger = logging.getLogger(__name__)



### Used for App real-time prediction generation this function shall be executed when the data_cleaned is prepared
def prediction_helper_predict(text, data_path, config_file):
    """This function is the helper of creating a real-time sentiment prediction for a single model under app application 
    Args:
        text (str): user input string with airline comment
        config_file (str): the configuration file path 
        data_path (str): the path of source data 

    Results:
        model ('sklearn.linear_model.logistic.LogisticRegression'): Logistic regression model trained.
        df_1_sparse (sparse matrix): a sparse test matrix (one line input) ready for making prediction
        features_list1 (list): the feature name list of text 
        text_list (list): list of input text features
             
    """
    
    text_list = process_text(text)
    ## prepare input dataframe
    df = pd.DataFrame(columns = ["tokens"])
    ## create a tokens column 
    df.loc[0,"tokens"] = text
    df_1 = pd.DataFrame(df["tokens"].apply(lambda x:process_text(x)))

    ## in order to obtain the transformed version of test data, we need to load the process to obtain the vectorizer
    
    data = pd.read_csv(data_path)
    data_cleaned = clean(data)

    logger.info("Cleaned the input")
    

    train_new, test_new = split_data(data_cleaned, **config_file["train_model"]["split_data"])

    
    X_train_sparse1, df_1_sparse, features_list1 = featurize(train = train_new, test = df_1, **config_file["featurize"]["featurize"])
    
    model = train_model_logistic(train_new, X_train_sparse1, **config_file["train_model"]["train_model_logistic"])
    
    logger.info("Featurize the input")
    
    logger.info("Successfully process data")
         

    return model, df_1_sparse, features_list1, text_list


