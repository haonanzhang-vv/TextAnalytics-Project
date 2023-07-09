import argparse
import logging

import yaml
import pandas as pd
import nltk
logging.basicConfig(format='%(name)-12s %(levelname)-8s %(message)s', level=logging.DEBUG)
logger = logging.getLogger('run-reproducibility')

from src.acquire import load_s3
from src.clean import clean, process_text

from src.featurize import featurize
#from src.helper import load_csv
from src.train_model import split_data, train_model_logistic
from src.Evaluation import evaluate_model
from src.score_model import score_model 
from src.model_inference import model_inference, plot_coefficients
#from test.test_featurize import *
from test.test import run_tests


if __name__ == '__main__':
    
    nltk.download("stopwords")
    parser = argparse.ArgumentParser(description="Acquire, clean, create features, training and evaluating model from cloud")

    parser.add_argument('step', help='Which step to run', choices=['all','acquire', 'clean', 'train_model','score_model','Evaluation','model_inference','test_featurize'])
    parser.add_argument('--input', '-i', default=None, help='Path to input data')
    parser.add_argument('--input1', '-i1', default=None, help='Path to input data')
    parser.add_argument('--config', help='Path to configuration file')
    parser.add_argument('--output', '-o', default=None, help='Path to save output CSV (optional, default = None)')
    parser.add_argument('--output1', '-o1', default=None, help='Path to save output either model .pkl or CSV (optional, default = None)')

    args = parser.parse_args()

    # Load configuration file for parameters and tmo path
    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    logger.info("Configuration file loaded from %s" % args.config)

    # ## load data from the input path
    if args.input is not None:
         input = pd.read_csv(args.input)
         logger.info('Input data loaded from %s', args.input)
    
    if args.input1 is not None:
         input1 = pd.read_csv(args.input1)
         logger.info('Input data loaded from %s', args.input1)

    if args.step == 'acquire':
        output = load_s3(**config['acquire']['load_s3'])
        
    elif args.step ==  "clean":
        ## load the original csv       
        output = clean(input)        

    elif args.step == "train_model":
        
        data_cleaned = clean(input)
        ## train test split
        train_new, test_new = split_data(data_cleaned,**config["train_model"]["split_data"])
        X_train_sparse, X_test_sparse, features_list = featurize(train_new,test_new,**config["featurize"]["featurize"])
        logger.info("feature list of training model created")
        ## fit and save the logistic regression model
        ## already set the model save path in configuration file
        tmo = train_model_logistic(train_new, X_train_sparse, **config["train_model"]["train_model_logistic"])
        ## ready to save test file
        output = test_new
        output1 = train_new
        
    elif args.step == "score_model":
        ## import the dataset with full features
        ### process the test data into sparse matrix
        data_cleaned = clean(input)
        
        train_new, test_new = split_data(data_cleaned,**config["train_model"]["split_data"])
        X_train_sparse, X_test_sparse, features_list = featurize(train_new,test_new,**config["featurize"]["featurize"])
        

        ## generate prediction output dataframe
        output = score_model(X_test_sparse,**config["score_model"]["score_model"])
    
    elif args.step == "Evaluation":
        
        data_cleaned = clean(input)
        ### process the test data into sparse matrix
        train_new, test_new = split_data(data_cleaned,**config["train_model"]["split_data"])
        X_train_sparse, X_test_sparse, features_list = featurize(train_new,test_new,**config["featurize"]["featurize"])
        
        ## input1 is the prediction dataframe  of the test set(the function output from score_model)                
        ## output is the confusion matrix df, and output1 is the metrics dataframe
        output  = evaluate_model(test_new, input1, **config["Evaluation"]["evaluate_model"])[0]
        output1 = evaluate_model(test_new, input1, **config["Evaluation"]["evaluate_model"])[1]
        
    elif args.step == "model_inference":
        ### input data is the csv for original feature data
        ### process the test data into sparse matrix
        data_cleaned = clean(input)
        train_new, test_new = split_data(data_cleaned,**config["train_model"]["split_data"])
        X_train_sparse, X_test_sparse, features_list = featurize(train_new,test_new,**config["featurize"]["featurize"])
      
        output = model_inference(features_list, **config["model_inference"]["model_inference"])
        
    else:
        run_tests(args)

    if args.output is not None:
        output.to_csv(args.output, index=False)
    
    if args.output1 is not None:
        output1.to_csv(args.output1, index=False)

        logger.info("Output saved to %s" % args.output)