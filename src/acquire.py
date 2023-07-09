#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 10:03:24 2020

@author: zhanghaonan
"""



#import requests
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def load_s3(sourceurl,**kwargs):
    """Function to get data as a dataframe from an online source.
    Args:
        sourceurl (str): URL of raw data in s3.
        file_location (str): Location where the csv file is saved.
    Returns:
        df (:py:class:`pandas.DataFrame`): DataFrame containing extracted features and target.
    """
    try:
        df = pd.read_csv(sourceurl)
        logger.info("Download from s3 bucket.")
    except Exception as e:
        logger.error("Error: Unable to download file.",e)

    # load data
  #  df.to_csv(file_location)

    return df


