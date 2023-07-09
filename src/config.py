import os
from os import path

# Getting the parent directory of this file. That will function as the project home.
PROJECT_HOME = path.dirname(path.dirname(path.abspath(__file__)))

# App config
APP_NAME = "tweet_sentiment"
DEBUG = True

# Logging
LOGGING_CONFIG = path.join(PROJECT_HOME, 'config/logging.conf')

# Database connection config
DATABASE_PATH = path.join(PROJECT_HOME, 'data/tweet_sentiment.db')
SQLALCHEMY_DATABASE_URI = 'sqlite:////{}'.format(DATABASE_PATH)
SQLALCHEMY_TRACK_MODIFICATIONS = True
SQLALCHEMY_ECHO = False  # If true, SQL for queries made will be printed

## S3 configuration setup
S3_PUBLIC_KEY = os.environ.get('S3_PUBLIC_KEY')
MSIA423_S3_SECRET = os.environ.get('MSIA423_S3_SECRET')

## upload to S3
INPUT_FILE_PATH = os.environ.get("INPUT_FILE_PATH")
BUCKET_NAME = os.environ.get("BUCKET_NAME")
OUTPUT_FILE_PATH = os.environ.get("OUTPUT_FILE_PATH")

## download data
# download data
FILE_SAVE = "../data/Tweets.csv"

