#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 21:44:55 2020

@author: zhanghaonan
"""


import logging
logger = logging.getLogger(__name__)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer



    
def featurize(train,test, method, min_df=5, **kwargs):
    """This function allows user to specify either bag-of-words-counts
    or tf-idf method for creating text features
    Args:
        train (:py:class:`pandas.DataFrame`): train DataFrame that is ready for feature generation 
        test (:py:class:`pandas.DataFrame`): test DataFrame that is ready for feature generation 
        method (str): specify the method for text feature generation including tfidf and bags of words
        min_df (int): used for removing terms that appear too infrequently, default = 5, ignore terms that appear in less than 5 documents".
    
    Return:
        X_train (class 'numpy.float64') : train sparse matrix of text features importance indicator
        X_test  (class 'numpy.float64') : test sparse matrix of text features importance indicator, having same number of columns as X_train
        features_list (list): list of text features, length equal to the number of columns of X_train and X_test
        
        
    """

    train = list(train["tokens"].apply(lambda x: " ".join(x)))
    test = list(test["tokens"].apply(lambda x: " ".join(x)))
    
    ## check which tokenization and feature creation method is chosen
    if method == "tf-idf":
        vectoriser = TfidfVectorizer(min_df = min_df, tokenizer = lambda s: s.split(' '))
    elif method == "bag-of-words":
        vectoriser = CountVectorizer(min_df = min_df, tokenizer = lambda s: s.split(' '))
    else:
        raise ValueError("Unknown text feature methods.")
    
    
#    try:
#        if method == "tf-idf":
#            vectoriser = TfidfVectorizer(min_df = min_df, tokenizer = lambda s: s.split(' '))
#        elif method == "bag-of-words":
#            vectoriser = CountVectorizer(min_df = min_df, tokenizer = lambda s: s.split(' '))
#        else:
#            print("invalid")
#
#    except Exception as e:
#        logger.error(e)
#        logger.error("Fail to specify text processing method. Either tf-idf or bag-of-words")
    
    X_train = vectoriser.fit_transform(train)
    X_test = vectoriser.transform(test)    
    features_list = vectoriser.get_feature_names()
    
    if X_train.shape[1] != X_test.shape[1] and X_test.shape[1]!= len(features_list):
        raise ValueError("input column dimension not correspond")
        
    return X_train, X_test, features_list
  
    
  
    

    
