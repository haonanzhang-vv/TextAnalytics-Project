"""
Created on 5/9/20
@author: Haonan Zhang
"""

import os
import sys
import argparse
import boto3
import logging

logging.basicConfig(level=logging.DEBUG, filename="logfile_acquisition.log", filemode="a+",
                        format="%(asctime)-15s %(levelname)-8s %(message)s")
logger = logging.getLogger(__file__)


S3_PUBLIC_KEY = os.environ.get('S3_PUBLIC_KEY')
MSIA423_S3_SECRET = os.environ.get('MSIA423_S3_SECRET')
INPUT_FILE_PATH = os.environ.get("INPUT_FILE_PATH")
BUCKET_NAME = os.environ.get("BUCKET_NAME")
OUTPUT_FILE_PATH = os.environ.get("OUTPUT_FILE_PATH")


def upload_data1():
    """upload data file to a specific S3 path
        Args:
                args (src): None
        Return:
                None(upload to S3)
    """
    try:
        s3 = boto3.client("s3", aws_access_key_id=S3_PUBLIC_KEY, aws_secret_access_key=MSIA423_S3_SECRET)
        logger.info("Valid key and credential")
    except:
        logger.error("Invalid AWS key and credential")

    try:    
        s3.upload_file(INPUT_FILE_PATH,BUCKET_NAME, OUTPUT_FILE_PATH)
        logger.info("Successfully load the file to S3")
    except:
        logger.error("File not uploaded to S3. Check the input file path, bucker name and output file path.")



if __name__ == "__main__":

    upload_data1()
