import os


DEBUG = True
LOGGING_CONFIG = "config/logging/local.conf"
PORT = 5000


APP_NAME = "twitter-sentiment"


SQLALCHEMY_TRACK_MODIFICATIONS = True
HOST = "0.0.0.0"
SQLALCHEMY_ECHO = False  # If true, SQL for queries made will be printed
MAX_ROWS_SHOW = 10


# connection string
conn_type = "mysql+pymysql"
user = os.environ.get("MYSQL_USER")
password = os.environ.get("MYSQL_PASSWORD")
host = os.environ.get("MYSQL_HOST")
port = os.environ.get("MYSQL_PORT")
DATABASE_NAME = os.environ.get("DATABASE_NAME")


SQLALCHEMY_DATABASE_URI = os.environ.get('SQLALCHEMY_DATABASE_URI')

if SQLALCHEMY_DATABASE_URI is not None:
    pass
elif host is None:
    SQLALCHEMY_DATABASE_URI = 'sqlite:///data/tweet_sentiment.db'
else:
    SQLALCHEMY_DATABASE_URI = '{dialect}://{user}:{pw}@{host}:{port}/{db}'.format(dialect=conn_type, user=user,
                                                                                  pw=password, host=host, port=port,
                                                                                  db=DATABASE_NAME)


### define other functional related configuration variables
PATH_TO_MODEL = "data/sentiment_class_prediction.pkl"
DATA_PATH = "data/Tweets.csv"
CONFIG_PATH = "config/test.yaml"
COEF_THRES = 1

### call other data processing related configuration variable
FEATURE_METHOD = "tf-idf" ## or "bag-of-words"
FEATURIZE_MIN_DF = 5
TRAIN_SIZE = 0.7
RANDOM_STATE = 1
      #train_save_path: "data/Tweets_train.csv"


