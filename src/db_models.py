"""
Created on 5/9/20

@author: Haonan Zhang

This file defines the function for creating database

"""
import os
import sys
import logging

import sqlalchemy as sql
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, Column, Integer, String, Text, Float
from sqlalchemy.ext.declarative import declarative_base

import argparse 
import config
# Configure flask app from flask_config.py


logging.basicConfig(level=logging.DEBUG, filename="logfile_db.log", filemode="a+",
                        format="%(asctime)-15s %(levelname)-8s %(message)s")
logger = logging.getLogger(__file__)


Base = declarative_base()

class Tweet_Sentiment(Base):
    """Create a data model for the database for tweet sentiment text """
    __tablename__ = 'tweet_sentiment_prediction'

    text = Column(String(255), unique=False, nullable=False,primary_key=True)
    airline_sentiment = Column(String(100), unique=False, nullable=True)
    airline_sentiment_positive_confidence = Column(String(100), unique=False, nullable=True)
    airline_sentiment_negative_confidence = Column(String(100), unique=False, nullable=True)





def get_engine_string(conn_type="mysql+pymysql"):
    """Get database engine path.
    Args:
        conn_tyep (str): Name of sql connection.

    Returns:
        engine_string (str): String defining SQLAlchemy connection URI.
    """
    
    user = os.environ.get("MYSQL_USER")
    password = os.environ.get("MYSQL_PASSWORD")
    host = os.environ.get("MYSQL_HOST")
    port = os.environ.get("MYSQL_PORT")
    DATABASE_NAME = os.environ.get("DATABASE_NAME")
    engine_string = "{}://{}:{}@{}:{}/{}".format(conn_type, user, password, host, port, DATABASE_NAME)


    logging.info("engine string is ready and impute to database: %s"%DATABASE_NAME)
    return  engine_string 


def create_db(args,engine=None):
    """Creates a database with the data models inherited from `Base`.
    Args:
        engine (`str`, default None): String defining SQLAlchemy connection URL in the form of
            `dialect+driver://username:password@host:port/database`. If None, `engine` must be provided.
        args: Parser arguments.
    Returns:
        engine (:py:class:`sqlalchemy.engine.Engine`, default None): SQLAlchemy connection engine.
        
    """
    if engine is None:
        if args.RDS:
            engine_string = get_engine_string()
        else:
            engine_string = args.local_URI
        logger.info("RDS:%s"%args.RDS)
        engine = sql.create_engine(engine_string)

    Base.metadata.create_all(engine)
    logging.info("database created") 

    return engine


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create defined tables in database")
    parser.add_argument("--RDS", default=False, action="store_true", help="True if want to create in RDS else None")

    parser.add_argument("--local_URI",default=config.SQLALCHEMY_DATABASE_URI)
    args = parser.parse_args()
    
    engine = create_db(args)


    # create a db session
    Session = sessionmaker(bind=engine)  
    session = Session()

    first_tweet = Tweet_Sentiment(
        text = "@VirginAmerica plus you've added commercials to the experience... tacky.",
        airline_sentiment = "positive",
        airline_sentiment_positive_confidence = str(1-0.3486),
        airline_sentiment_negative_confidence = str(0.3486)

        )

    session.add(first_tweet)
    session.commit()


    logger.info("Data added")

    query = "SELECT * FROM tweet_sentiment_prediction"
    result = session.execute(query)

    session.close()