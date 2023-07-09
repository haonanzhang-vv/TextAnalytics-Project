import os
import sys
import traceback
from flask import render_template, request, redirect, url_for
import logging.config
from flask import Flask
from app import *
from flask_sqlalchemy import SQLAlchemy
import pickle
import pandas as pd

### import relevant function
import yaml
import sklearn
import pymysql
from sklearn.model_selection import train_test_split
from src.clean import *
from src.featurize import *

from src.Predict_helper import prediction_helper_predict
from src.db_models import Tweet_Sentiment
import nltk
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from num2words import num2words

# Initialize the Flask application
app = Flask(__name__, template_folder="app/templates",static_folder="app/static")

# Configure flask app from flask_config.py
app.config.from_pyfile('config/flaskconfig.py')

# Define LOGGING_CONFIG in flask_config.py - path to config file for setting
# up the logger (e.g. config/logging/local.conf)
logging.config.fileConfig(app.config["LOGGING_CONFIG"])
logger = logging.getLogger(app.config["APP_NAME"])
logger.debug('Test log')

# Initialize the database
db = SQLAlchemy(app)



@app.route('/')
def index():
    """Homepage of the prediction system

    Create view into index page that uses data queried from tweet_sentiment database and
    inserts it into the app/templates/index.html template.

    Returns: rendered html template

    """

    try:
        
        logger.debug("Index page accessed")
        return render_template('Homepage.html')
    except:
      
        logger.warning("Not able to display tweets sentiments, error page returned")
        return render_template('error 2.html')



@app.route('/add', methods=['POST','GET'])
def add_entry():
    """View that process a POST with new tweets input

    :return: redirect to index page
    """
    tweet_text = request.form["tweet_text"]


    try:

        tweet_text = request.form["tweet_text"]
        print(tweet_text)

        logger.info("The text input retrieved.")
        ## load the trained model and call function prediction helper  PATH_TO_MODEL = "data/sentiment_class_prediction.pkl"
        
        logger.info("The path to model config " + app.config["PATH_TO_MODEL"])
        logger.info("The data path config " + app.config["DATA_PATH"])
      
        
        with open(app.config["CONFIG_PATH"], "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

       
        ## call prediction helper
        model_log, test_sparse, features, important_text_list =  prediction_helper_predict(tweet_text,app.config["DATA_PATH"],config)
        
        logger.info("Model loaded")
        
        prediction = model_log.predict_proba(test_sparse)[0]
        ## checking relevant features for discriminating prediction
        fitted = pd.DataFrame(index=features)        
        fitted['coefs'] = model_log.coef_[0]
        ## obtain positive words and negative word
        pos_word = set(fitted[fitted.coefs > app.config["COEF_THRES"]].index)
        neg_word = set(fitted[fitted.coefs < - app.config["COEF_THRES"]].index)
            
        pos_words = [  elem for elem in important_text_list if elem in pos_word]
        neg_words = [  elem for elem in important_text_list if elem in neg_word]
                
        neg_prob = prediction[0]
        pos_prob = prediction[1]
                
        if pos_prob >= neg_prob:
            sentiment = "positive"
        else: 
            sentiment = "negative"
                
        ### add this new input text and its prediction sentiment and score
        tweet_entry = Tweet_Sentiment(
                text = tweet_text,
                airline_sentiment = sentiment,
                airline_sentiment_positive_confidence = str(pos_prob),
                airline_sentiment_negative_confidence = str(neg_prob)
                )
        logger.info("tweet entry for database is ready")
        db.session.add(tweet_entry)
        db.session.commit()
        logger.info("New tweet predicted sentiment added: %s is %s", tweet_text, sentiment)
          

        return render_template('Homepage.html',negative_sentiment = neg_prob, positive_sentiment= pos_prob ,negative_words= neg_words , positive_words = pos_words)
                        
       
    except:
        traceback.print_exc()
        logger.warning("Not able to proceed, error page returned")
        return render_template('error 2.html')


if __name__ == '__main__':
    nltk.download('stopwords')        
    app.run(debug=app.config["DEBUG"], port=app.config["PORT"], host=app.config["HOST"])








