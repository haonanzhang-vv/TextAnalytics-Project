#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 21:28:25 2020

@author: zhanghaonan
"""



import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore') 
import nltk
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from num2words import num2words
import logging
import string
logger = logging.getLogger(__name__)



def process_text(text):
    """ This function will process the raw text into structural format and tokenize the original text
    Args:
        text (str): a line of textual sentences that can be split and separate
    
    Return:
        tokens (:list): list of tokens
    """
    ## check the input is not none and type invalid
    if text is None:
        raise ValueError("The input value is not valid.")
    elif type(text) is not str:
        raise TypeError("The input type is not string.")
    else:
        pass
    
    Tokenizer = TweetTokenizer()
    ## tokenize the text
    tokenized = Tokenizer.tokenize(text)
    ## define punctuation
    punctuation = list(string.punctuation)
    ## remove the last punctuation
    punctuation.remove('!')
    tokenized_no_punctuation=[word.lower() for word in tokenized if word not in punctuation]
    tokenized_no_stopwords=[word for word in tokenized_no_punctuation if word not in stopwords.words('english')]
    ## extract the stem
    tokens = [PorterStemmer().stem(word) for word in tokenized_no_stopwords if word != '️']
    
    for i in range(len(tokens)):
        try:
            tokens[i]=num2words(tokens[i])
        except:
            pass
        
    return tokens




def clean(data):
    """ This functio will return the clean dataset including token feature columns
    Args:
        data (:py:class:`pandas.DataFrame`): DataFrame of the original dataset
    
    Return:
        data (:py:class:`pandas.DataFrame`): DataFrame that is ready for feature generation
    """
    ## check whether the required column exist or not before proceeding the cleaning
    if "airline_sentiment" not in data.columns:
        raise ValueError("The sentiment label does not exist.")
    
    ## filter out neutral class
    data=data[data['airline_sentiment']!='neutral']
    data=data[data['airline_sentiment_confidence']==1.0]

    
    ## call process_text helper function
    logger.info("Ready to process raw text into tokens")
    data['tokens']=data['text'].apply(process_text)
    logger.info("Successfully process raw text into tokens")
    ## convert the class as integer
    data['positive']=(data['airline_sentiment']=='positive').astype(int)
    ## define the final data for processing
    data = data[["airline","tokens","airline_sentiment","positive"]]
    return data
    
    
