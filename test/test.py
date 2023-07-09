#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 22:52:03 2020

@author: zhanghaonan
"""

import numpy as np
import pandas as pd
import logging
import pytest
import nltk


logger = logging.getLogger(__name__)
logger.setLevel("DEBUG")

import sklearn
from sklearn import model_selection
from sklearn import linear_model
from src.clean import process_text, clean
from src.featurize import featurize
from src.train_model import *
from src.score_model import *
from src.Evaluation import *
from src.model_inference import * 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


### unit test function for process_text
### happy path
def test_process_text_happy():
    """
    Test the functionality of process text input
    """
    
    ### create some text case for input
    case1 = '@VirginAmerica What @dhepburn said.'
    case2 = "@VirginAmerica plus you've added commercials to the experience... tacky."
    case3 = "@VirginAmerica I didn't today... Must mean I need to take another trip!"
    
    
    ## function output
    case1_func = process_text(case1)
    case2_func = process_text(case2)
    case3_func = process_text(case3)
    
    ## expected output
    case1_exp = ['@virginamerica', '@dhepburn', 'said']
    case2_exp = ['@virginamerica', 'plu', 'ad', 'commerci', 'experi', '...', 'tacki']
    case3_exp = ['@virginamerica', 'today', '...','must', 'mean','need','take','anoth','trip',
'!']

    print(case1_func == case1_exp)
    print(case2_func == case2_exp)
    print(case3_func == case3_exp)
     
    assert case1_func == case1_exp
    assert case2_func == case2_exp
    assert case3_func == case3_exp
    
### unit test function for process_text
### unhappy path  

def test_process_text_unhappy_none_input():
    """
    Test the functionality of process text input
    The bad case is when the input is none or the input is not string, 
    which will raise errors.
    """
    with pytest.raises(ValueError) as excinfo:
        input_text = None
        output = process_text(input_text)
      
    print(str(excinfo.value) == "The input value is not valid.")
    assert str(excinfo.value) == "The input value is not valid."
 

def test_process_text_unhappy_nonstring_input():
    """
    Test the functionality of process text input
    The bad case is when the input is none or the input is not string, 
    which will raise errors.
    """
    with pytest.raises(TypeError) as excinfo:
        input_text = 123456
        output = process_text(input)
      
    print(str(excinfo.value) == "The input type is not string.")
    assert str(excinfo.value) == "The input type is not string."
 

### unit test function for clean 
### happy path 

def test_clean_happy():
    """
    This is the function to test clean function, as a happy path
    """
    data = pd.read_csv("test/test_data/test_Tweets.csv")
    
    small_test = data.iloc[:5]
    
    ## function output
    df_func = clean(small_test)
    
    ## expected output
    array = np.array([['Virgin America',
        list(['@virginamerica', 'realli', 'aggress', 'blast', 'obnoxi', 'entertain', 'guest', 'face', 'littl', 'recours']),
        'negative', 0],
       ['Virgin America',
        list(['@virginamerica', 'realli', 'big', 'bad', 'thing']),
        'negative', 0]])
    df_exp = pd.DataFrame(array, columns=[["airline","tokens","airline_sentiment","positive"]])
    
    print(df_func)
    print(df_exp)
    print(df_func.values == (df_exp.values))
    
    assert df_func.values.all() == df_exp.values.all()
    
    

### unit test function for clean
### unhappy path 

def test_clean_unhappy():
    """
    This function will be an unhappy path test
    The unhappy case is when the 
    """
    
    data = pd.read_csv("test/test_data/test_Tweets.csv")
    
    small_test = data.iloc[:5]
    ## make the data sentiment to be removed
    small_test = small_test.drop("airline_sentiment",axis = 1)
    
    print(small_test)
    with pytest.raises(ValueError) as excinfo:
        output = clean(small_test)
      
    print(str(excinfo.value) == "The sentiment label does not exist.")
    assert str(excinfo.value) == "The sentiment label does not exist."


   
def test_featurize_happy():
    """
    This function will be a happy path for the unit test of generating text feature
    """
    data = pd.read_csv("test/test_data/test_Tweets.csv")
    small_train = data.iloc[:100]  
    small_test = data.iloc[100:150]
    ## function output
    small_train = clean(small_train)
    small_test = clean(small_test)
    
    train_func, test_func, features_func = featurize(small_train,small_test,"tf-idf", min_df=5)
    
    
    ## expected output
    train = list(small_train["tokens"].apply(lambda x: " ".join(x)))
    test = list(small_test["tokens"].apply(lambda x: " ".join(x)))
    vectoriser = TfidfVectorizer(min_df = 5, tokenizer = lambda s: s.split(' '))
    
    train_exp = vectoriser.fit_transform(train)
    test_exp = vectoriser.transform(test)    
    features_exp = vectoriser.get_feature_names() 
    
    train_exp_df = pd.DataFrame.sparse.from_spmatrix(train_exp)
    train_func_df = pd.DataFrame.sparse.from_spmatrix(train_func)
    
    test_exp_df = pd.DataFrame.sparse.from_spmatrix(test_exp)
    test_func_df = pd.DataFrame.sparse.from_spmatrix(test_func)
    
    assert test_exp_df.round(10).equals(test_func_df.round(10))
    
    assert train_exp_df.round(10).equals(train_func_df.round(10))
    
    assert features_exp == features_func
        


### unit test function for featurize
### unhappy path 

def test_featurize_unhappy():
    """
    This function will be an unhappy path for the unit test of generating text feature
    The bad case is that the specifying an unknown text processing method.
    """
    
    with pytest.raises(ValueError) as excinfo:
        data = pd.read_csv("test/test_data/test_Tweets.csv")
        ## we deliberately give the unknown name of text processing method
        small_train = data.iloc[:100]  
        small_test = data.iloc[100:150]
        
        ## function output
        small_train = clean(small_train)
        small_test = clean(small_test)
    
        train_func, test_func, features_func = featurize(small_train,small_test,"tfidf", min_df=5)
    
    
    assert str(excinfo.value) == "Unknown text feature methods."

    

### unit test function for split_data
### happy path 

def split_data_happy():
    """
    Test functionality of splitting train test data
    """
    
    split_test_data = pd.read_csv("test/test_data/test_Tweets_clean.csv")
    function_train, function_test = split_data(split_test_data,trainSize = 0.7, randomState = 1)
    
    ## expected output
    index_train, index_test  = train_test_split(np.array(split_test_data.index), train_size=0.7, 
                                                random_state = 1, stratify=split_test_data['positive'])
    
    expected_train = split_test_data.loc[index_train,:].copy()
    expected_test = split_test_data.loc[index_test,:].copy()
    

    assert function_train.equals(expected_train)
    assert function_test.equals(expected_test)
    
    

### unit test function for split_data
### unhappy path 


def split_data_unhappy():
    """
    Test functionality of splitting train test data
    The bad case is there is no input data for splitting
    """
    with pytest.raises(ValueError) as excinfo:
        split_test_data = None
        function_train, function_test = split_data(split_test_data,trainSize = 0.7, randomState = 1)
    
    assert str(excinfo.value) == "No input data is ready to split."



### unit test function for train_model_logistic
### happy path 

def test_train_model_logistic_happy():
    """
    This functionality will test the logistic model training
    A happy test assert the correct model type, correct coefficient value, correct input type
    """
    
    model_test_train = pd.read_csv("test/test_data/test_model_trainset.csv")
    
    model_kwargs = {    "features" : ["visible_mean_distribution","visible_contrast","visible_entropy","visible_second_angular_momentum"],
        "train_model_logistic":{
                          'target_column':'class',
                     "Cs": 50,
                    "fitIntercept": True,
                    "penalty":"l2",
                    "save_tmo": "test/result_data/model_func_result.pkl"}
                   }
    
    X_train = model_test_train.loc[:,model_kwargs["features"]]
    function_output = train_model_logistic(model_test_train, X_train, **model_kwargs["train_model_logistic"])
    
    ### expected
    y_train = model_test_train["class"]
    logit_l2 = LogisticRegressionCV(Cs = 50, fit_intercept = True, penalty= "l2", solver='liblinear', scoring='neg_log_loss')
    logit_l2.fit(X_train, y_train)
    
    ## Refit the model with the best complex parameter 
    expected_output = LogisticRegression(C = logit_l2.C_[0], penalty="l2", solver='liblinear')
    expected_output.fit(X_train, y_train)
    

    ### check the assertion the model types are the same
    assert str(type(expected_output)) == str(type(function_output))
    
    ### check the assert the model coefficients are the same
    assert str(function_output.coef_) == str(expected_output.coef_)
    
    ### check whether the model setting are the same
    assert str(function_output) == str(expected_output)


### unit test function for train_model_logistic
### unhappy path 

def test_train_model_logistic_unhappy():
    """
    This functionality will test the logistic model training
    A unhappy test assert the incorrect input type
    """
    with pytest.raises(ValueError) as excinfo:
        model_test_train = pd.read_csv("test/test_data/test_model_trainset.csv")

        model_kwargs = {    "features" : ["visible_mean_distribution","visible_contrast","visible_entropy","visible_second_angular_momentum"],
            "train_model_logistic":{
                              'target_column':'class',
                         "Cs": 50,
                        "fitIntercept": True,
                        "penalty":"l2",
                        "save_tmo": "test/result_data/model_func_result.pkl"}
                       }

        ## we deliberately select only a half rows of X_train
        X_train = model_test_train.loc[:len(model_test_train)/2,model_kwargs["features"]]
        function_output = train_model_logistic(model_test_train, X_train, **model_kwargs["train_model_logistic"])
    
    assert str(excinfo.value) == "Input training features sparse matrix does not match the row number of response."



### unit test function for score_model
### happy path 

def test_score_predict_happy():
    """
    This functionality will test the score prediction function
    A happy test assert the correct prediction result
    
    """
    ## function result
    predict_testset = pd.read_csv("test/test_data/test_model_testset.csv")
    predict_train = pd.read_csv("test/test_data/test_model_trainset.csv")
    
    model_kwargs = {"features" : ["visible_mean_distribution","visible_contrast","visible_entropy","visible_second_angular_momentum"],
        "train_model_logistic":{
                          'target_column':'class',
                     "Cs": 50,
                    "fitIntercept": True,
                    "penalty":"l2",
                    "save_tmo": "test/result_data/model_func_result.pkl"},
        "score_model": {
        "thres": 0.5,
        "path_to_tmo" : "test/result_data/model_func_result.pkl"} 
    }
    X_train = predict_train.loc[:,model_kwargs["features"]]
    X_test = predict_testset.loc[:,model_kwargs["features"]]
    
    function_output = score_model(X_test,**model_kwargs["score_model"])
    
    function_prob = function_output["predicted_proba"]
    function_class = function_output["predicted_class"]
        
    y_train = predict_train["class"]
    y_test = predict_testset["class"]
    logit_l2 = LogisticRegressionCV(Cs = 50, fit_intercept = True, penalty= "l2", solver='liblinear', scoring='neg_log_loss')
    logit_l2.fit(X_train, y_train)
    ## Refit the model with the best complex parameter 
    expected_output = LogisticRegression(C = logit_l2.C_[0], penalty="l2", solver='liblinear')
    expected_output.fit(X_train, y_train)
    expected_prob = expected_output.predict_proba(X_test)[:,1]
    

    
 #  assert the function probability will between 0 and 1
    assert function_prob.between(0,1,inclusive=True).all()
    # assert the function class is either 0 or 1
    assert function_class.isin([0,1]).all()
    # assert the return probability of function is the same as the probability of expected output
    assert function_prob.all() == expected_prob.all()
    


### unit test function for score_model
### unhappy path 

def test_score_predict_unhappy():
    """
    This functionality will test the score prediction function
    A unhappy test assert the the test data has incorrect dimensions with the model
    
    """
    ## function result
    
    with pytest.raises(ValueError) as excinfo:
            
        predict_testset = pd.read_csv("test/test_data/test_model_testset.csv")
        predict_train = pd.read_csv("test/test_data/test_model_trainset.csv")
    
        model_kwargs = {"features" : ["visible_mean_distribution","visible_entropy","visible_second_angular_momentum"],
        "train_model_logistic":{
                          'target_column':'class',
                     "Cs": 50,
                    "fitIntercept": True,
                    "penalty":"l2",
                    "save_tmo": "test/result_data/model_func_result.pkl"},
        "score_model": {
        "thres": 0.5,
        "path_to_tmo" : "test/result_data/model_func_result.pkl"} 
        }
        
        X_test = predict_testset.loc[:,model_kwargs["features"]]
    
        function_output = score_model(X_test,**model_kwargs["score_model"])
    
        function_prob = function_output["predicted_proba"]
        function_class = function_output["predicted_class"]
     
        
    assert str(excinfo.value) == "Test data features dimension does not match the model training feature dimension"
        


### unit test function for evaluate_model
### happy path 
    
def test_evaluate_model_happy():
    """Test the functionality of evaluate_model."""
    # test data input
    score_input = {"predicted_proba": [0.98,0,0.99,0.94,0.93,0,0.06,0.699,0.04,0.97],
                   "predicted_class": [1,0,1,1,1,0,0,1,0,1]}
    label_input = {'class':[0,1,0,1,0,1,0,0,1,0]}

    score_df = pd.DataFrame(score_input)
    label_df = pd.DataFrame(label_input)

    # desired output dataframe
    output = sklearn.metrics.confusion_matrix(label_df, score_df.iloc[:,1])
    output_df = pd.DataFrame(output,
        index=['Actual Negative','Actual Positive'],
        columns=['Predicted Negative', 'Predicted Positive'])
    
    # add kwargs for function
    pre_defined_kwargs = {"target_column": "class",
                          'metrics':["auc","accuracy","f1_score"]}
    
    ## assert tjhe confusion matrix is the same
    assert output_df.equals(evaluate_model(label_df, score_df, **pre_defined_kwargs)[0])


### unit test function for evaluation model 
### unhappy path 

def test_evaluate_model_unhappy():
    """Test the functionality of evaluate_model."""
    # test data input
    with pytest.raises(IndexError) as excinfo:
        score_input = {"predicted_proba": [0.98,0,0.99,0.94,0.93,0,0.06,0.699],
                       "predicted_class": [1,0,1,1,1,0,0,1]}
        label_input = {'class':[0,1,0,1,0,1,0,0,1,0]}

        score_df = pd.DataFrame(score_input)
        label_df = pd.DataFrame(label_input)

        # add kwargs for function
        pre_defined_kwargs = {"target_column": "class",
                              'metrics':["auc","accuracy","f1_score"]}
        # raise AssertionError if dataframes do not match
        evaluate_model(label_df, score_df, **pre_defined_kwargs)[0]
        ## assert tjhe confusion matrix is the same   
    assert str(excinfo.value) == 'Index out of bounds!'
    


### unit test function for model_inference
### happy path 
def test_model_inference_happy():
    """
    This is the test the functionality of model inference.
    """
    
    model_kwargs = {"features" : ["visible_mean_distribution","visible_contrast","visible_entropy","visible_second_angular_momentum"],
        "train_model_logistic":{
                          'target_column':'class',
                     "Cs": 50,
                    "fitIntercept": True,
                    "penalty":"l2",
                    "save_tmo": "test/result_data/model_func_result.pkl"},
        "score_model": {
        "thres": 0.5,
        "path_to_tmo" : "test/result_data/model_func_result.pkl"},
         "model_inference": {"features" : ["visible_mean_distribution","visible_contrast","visible_entropy","visible_second_angular_momentum"],
        "path_to_tmo": "test/result_data/model_func_result.pkl",
        "path_save_plot": "test/result_data/test_plot.png"
         }           
    }
    ## function result
    result = model_inference(**model_kwargs["model_inference"])
    
    ## expected result
    predict_testset = pd.read_csv("test/test_data/test_model_testset.csv")
    predict_train = pd.read_csv("test/test_data/test_model_trainset.csv")

    X_train = predict_train.loc[:,model_kwargs["features"]]
    X_test = predict_testset.loc[:,model_kwargs["features"]]

    function_output = train_model_logistic(predict_train, X_train, **model_kwargs["train_model_logistic"])
    
    fitted_coef = pd.DataFrame(index=model_kwargs["features"]) 
    fitted_coef['coefs'] = function_output.coef_[0]
    fitted_coef['odds_ratio'] = fitted_coef.coefs.apply(np.exp)
    fitted_coef = fitted_coef.sort_values(by='odds_ratio', ascending=False)
    
    
    assert result.equals(fitted_coef)
    
    
    
    
### unit test function for model_inference
### unhappy path 

def test_model_inference_unhappy():
    """
    This is the test for the unhappy path of model inference function.
    It will test the bad input when the input features list size do not match the 
    model coefficients list size
    
    """
    with pytest.raises(ValueError) as excinfo:
        model_kwargs = {
             "model_inference": {"features" : ["visible_mean_distribution","visible_contrast", "visible_second_angular_momentum"],
            "path_to_tmo": "test/result_data/model_func_result.pkl",
            "path_save_plot": "test/result_data/test_plot.png"
             }           
        }
        ## function result
        result = model_inference(**model_kwargs["model_inference"])
      
    assert str(excinfo.value) == "Input feature name list does not correspond to the model coefficients"
    
        


    
def run_tests(args=None,config_path=None):
    """Runs commands in config file and compares the generated files to those that are expected
    
    """
    nltk.download('stopwords')
    logger.info("---------Running the happy test for process_text happy-------------")
    test_process_text_happy()
    
    logger.info("---------Running the unhappy test for process text with None input -------------")
    test_process_text_unhappy_none_input()
    
    logger.info("---------Running the unhappy test for process text with non-string input -------------")
    test_process_text_unhappy_nonstring_input()
    
    logger.info("---------Running the happy test for clean-------------")
    test_clean_happy()
    
    logger.info("---------Running the unhappy test for clean -------------")
    test_clean_unhappy()
    
    
    logger.info("---------Running the happy test for featurize------------")
    test_featurize_happy()
    logger.info("---------Running the unhappy test for featurize-------------")
    test_featurize_unhappy()
    
    logger.info("---------Running the happy test for train model-------------")
    test_train_model_logistic_happy
    logger.info("---------Running the unhappy test for train model------------")
    test_train_model_logistic_unhappy
    
    logger.info("---------Running the happy test for split data------------")
    split_data_happy()
    logger.info("---------Running the unhappy test for split data-------------")
    split_data_unhappy()
    
    
    
    logger.info("---------Running the happy test for score prediction-------------")
    test_score_predict_happy()
    logger.info("---------Running the unhappy test for score prediction------------")
    test_score_predict_unhappy()
       
    logger.info("---------Running the happy test for evaluate model-------------")
    test_evaluate_model_happy()
    logger.info("---------Running the unhappy test for evaluate model---------")
    test_evaluate_model_unhappy()
    
    
    
    logger.info("---------Running the happy test for model inference-------------")
    test_model_inference_happy()
    logger.info("---------Running the unhappy test for model inference------------")
    test_model_inference_unhappy()
       


    
    

















    
    