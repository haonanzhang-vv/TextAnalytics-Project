# Twitter Airline Sentiment Analysis
### Developer: Haonan Zhang
### QA: Nancy Zhang

### commit to test2

# Final Repository

## 1. Build the image and run the model pipeline

The instructors have access to my S3 bucket to access the data.

```

export AWS_ACCESS_KEY_ID=<your aws access key ID>

export AWS_SECRET_ACCESS_KEY=<secret access key>

```

```

docker build -f Dockerfile -t tweet_sentiment .

```

Execute all model pipeline using makefile. Here the dockerfile has entry point make.
If you want to change the output directory, you can change the output paths in the make file.

```

docker run --mount type=bind,source="$(pwd)"/data,target=/app/data --env AWS_ACCESS_KEY_ID --env AWS_SECRET_ACCESS_KEY tweet_sentiment all

```

Execute the unit test

```

docker run --mount type=bind,source="$(pwd)"/test,target=/app/test tweet_sentiment test

```



## 2. Run Web App

Warning: You must run model pipeline and contain a model in your setting and related datasets for feature creation 
before proceeding to the web application

The procedure generally includes exporting the environment variable and building the docker image for web app.

### 2.1. Apps interfacing with a local database

If you use local database in the app for inserting the prediction result:(When the local database is not available, please consult the following section 

Creating local or AWS RDS database schema):

```bash

cd ~/2020-msia423-Zhang-Haonan

export SQLALCHEMY_DATABASE_URI='sqlite:///data/tweet_sentiment.db'

```

Build up the docker image

```bash

docker build -f app/Dockerfile -t tweet .

```

Docker run command to access the webpage.

```bash

docker run --mount type=bind,source="$(pwd)"/data,target=/app/data --env SQLALCHEMY_DATABASE_URI -p 5000:5000 --name test tweet


```

### 2.2. Apps interfacing with AWS RDS

If you use AWS RDS in the app for inserting the prediction result:

```
Edit your mysql config file accordingly 

```bash
cd ~/2020-msia423-Zhang-Haonan

cd config

vi mysqlconfig.env

```

This mysqlconfig.env is related to the mysql account for writing the prediction to the database. If you do not have 
a tweet_sentiment_prediction table, please look at section 3.2 create aws for reference. 

* Set `MYSQL_USER` to the "master username" that you used to create the database server.
* Set `MYSQL_PASSWORD` to the "master password" that you used to create the database server.
* Set `MYSQL_HOST` to be the RDS instance endpoint from the console
* Set `MYSQL_HOST` to be `3306`
* Set `DATABASE_NAME` = msia423_db
 

```bash

docker build -f app/Dockerfile -t tweet .

```

```bash 

docker run --mount type=bind,source="$(pwd)"/data,target=/app/data --env-file config/mysqlconfig.env -p 5000:5000 --name testhaonanzhangvv tweet

```


Either way works well with clear specification the environment variable.
Then you can access the website http://0.0.0.0:5000/. 

The website contains a text box where you can either type and paste the text. After clicking the submit button,
you will get access to the sentiment score, and the discrinimative words.



(The following section gives instruction that can be used for course instructor to read my writted record using msia423instructor account)

##### 2.3 If you want to read and check the ingested result in my AWS RDS


You can use the MySQL client again to see that a table has been added and data generated.
You can run the Docker container by using the `run_mysql_client.sh` script.
Change the mysql configuration.

```
vi .mysqlconfig
```

* Set `MYSQL_USER` to the "master username" that you used to read the database server(msia423instructor).
* Set `MYSQL_PASSWORD` to the "master password" that you used to create the database server.
* Set `MYSQL_HOST` to be the RDS instance endpoint from the console
* Set `MYSQL_HOST` to be `3306`
* Set `DATABASE_NAME` = msia423_db

(please check the submission engine string to set these environment variable)
 
Set the environment variables in your `~/.bashrc`

```bash
echo 'source .mysqlconfig' >> ~/.bashrc
source ~/.bashrc
```

```bash
sh run_mysql_client.sh
```

```
show databases;

return:

+--------------------+
| Database           |
+--------------------+
| information_schema |
| innodb             |
| msia423_db         |
| mysql              |
| performance_schema |
| sys                |
+--------------------+

use msia423_db;

show tables;

return:

+----------------------------+
| Tables_in_msia423_db       |
+----------------------------+
| tweet_sentiment            |
| tweet_sentiment1           |
| tweet_sentiment2           |
| tweet_sentiment4           |
| tweet_sentiment5           |
| tweet_sentiment6           |
| tweet_sentiment_prediction |
+----------------------------+

select * from tweet_sentiment_prediction;

return:

+--------------------------------------------------------------------------+-------------------+---------------------------------------+---------------------------------------+
| text                                                                     | airline_sentiment | airline_sentiment_positive_confidence | airline_sentiment_negative_confidence |
+--------------------------------------------------------------------------+-------------------+---------------------------------------+---------------------------------------+
| @VirginAmerica plus you've added commercials to the experience... tacky. | positive          | 0.6514                                | 0.3486                                |
| I love analytics value chain!                                            | positive          | 0.9047008828580841                    | 0.09529911714191586                   |
| The analytics value chain is a fruitful course!                          | negative          | 0.1946295273040157                    | 0.8053704726959843                    |
+--------------------------------------------------------------------------+-------------------+---------------------------------------+---------------------------------------+




```

tweet_sentiment_prediction is the table that will include the newly injected value from the webapp

Whenever you use the web app to add a new entry, it will be reflected in the aws rds. 


##(supplementary) 3. Some remarks on creating database for your first time 

If you want to write the database on your own and ingest the data via the web app, here are the procedures of creating database.


# Creating local or AWS RDS database schema


### 3.1 Create a local database 

* If you want to create a local database:

* Before running the following command, if there is a database under data folder, please delete it first.

Run:
```

cd 2020-msia423-Zhang-Haonan

docker build -f Dockerfile-py3 -t models .

docker run --mount type=bind,source="$(pwd)"/data,target=/app/data models src/db_models.py


```

Then you can check the tweet_sentiment.db under the data folder.


### 3.2 If you want to create AWS RDS:


##### Connecting from your computer 

_Note: You will need to be on the Northwestern VPN for the remaining portions of the tutorial._

Edit your mysql config file accordingly 

```bash

cd 2020-msia423-Zhang-Haonan

vi .mysqlconfig

```

Please specify the RDS user input for creating a database.

* Set `MYSQL_USER` to the "master username" that you used to create the database server.
* Set `MYSQL_PASSWORD` to the "master password" that you used to create the database server.
* Set `MYSQL_HOST` to be the RDS instance endpoint from the console
* Set `MYSQL_HOST` to be `3306`
* Set `DATABASE_NAME` = msia423_db
 

Set the environment variables in your `~/.bashrc`

```bash
echo 'source .mysqlconfig' >> ~/.bashrc
source ~/.bashrc
```

**VERIFY THAT YOU ARE ON THE NORTHWESTERN VPN BEFORE YOU CONTINUE ON**

##### 2. Creating schema in your database (msia423_db) with SQLAlchemy

Build the provided Docker image 

```bash
docker build -f Dockerfile-py3 -t tweet_sentiment_mysql .
```

Set your database name and run the Docker container 

```bash
##export DATABASE_NAME=msia423_db
sh run_docker.sh

```

Then a table is created under your database. You can then go through the app web creation part to have your own app!





# Midpoint Check (Not Related to Final Model pipeline and web app)
<!-- toc -->
In addition to the master branch, I add the following scripts:

  
acquire_data.py: data acquisition from a URL

upload_data.py: data uploading to S3 bucket

models.py: code for creating either the tweet_sentiment schema on AWS RDS (msia423_db) or creating a local database under data folder


I also need to have the following files:

run_docker.sh

run_mysql_client.sh

requirements.txt

Dockerfile



# Data Acquisition and Ingestion to S3

## acquire_data.py(For users, you can skip this part for downloading the original data csv.)
<!-- toc -->
The dataset is from a kaggle competition. The developer has already acquired and download data.

This file is responsible for acquiring data from the original source URL, which is a static link in the github repo.

Download data from a GitHub repo from ./data/sample/Tweets.csv

command to run: 

```
cd ./src 
```
Specify save_path to ./data/Tweets.csv inside the acquire_data.py

```python 
python acquire_data.py
```

Then the csv file is under your data folder. For users, you can skip this part.

<!-- toc -->
## upload_data.py
<!-- toc -->
This file uploads data to the targeted s3 bucket, and make sure you configure AWS credential.

Make sure you specify the access_key_id, secret_access_key, input_path, bucker_name, output_path in the config.env file.

```

S3_PUBLIC_KEY=<access_key_id: private s3 bucket access key id>

MSIA423_S3_SECRET=<secret_access_key: private s3 bucket secret access key>

INPUT_FILE_PATH=data/Tweets.csv

BUCKET_NAME=nw-haonanzhang-s3

OUTPUT_FILE_PATH=Tweets.csv 

```

Change directory, and build the docker image acquire_data, and load the csv to S3

```

cd 2020-msia423-Zhang-Haonan

docker build -f Dockerfile -t acquire_data .

docker run --env-file=config.env  acquire_data src/upload_data.py 

```

Therefore, the file from the data folder has been uploaded to S3.



```
show databases;

return:

+--------------------+
| Database           |
+--------------------+
| information_schema |
| innodb             |
| msia423_db         |
| mysql              |
| performance_schema |
| sys                |
+--------------------+

use msia423_db;

show tables;

return:

+----------------------+
| Tables_in_msia423_db |
+----------------------+
| tweet_sentiment      |
| tweet_sentiment1     |
+----------------------+

select tweet_id, text from tweet_sentiment;

return:

+------------+--------------------------------------------------------------------------+
| tweet_id   | text                                                                     |
+------------+--------------------------------------------------------------------------+
| 2147483647 | @VirginAmerica plus you've added commercials to the experience... tacky. |
+------------+--------------------------------------------------------------------------+

```

You can check logfile_db.log to see whether db is successfully created.




# Outline
<!-- toc -->
- [Project Charter](#Project-Charter)
- [Project Planning](#Project-Planning)
- [Repo structure](#repo-structure)
- [Build the image and run the model pipeline](#Build-the-image-and-run-the-model-pipeline)
- [Running the app](#Running-the-app)
  * [1. Initialize the database](#1-initialize-the-database)
    + [1.1 Local SQLite database ](#1.1-Local-SQLite-database )
    + [1.2 Create an AWS RDS](#1.2-Create-an-AWS-RDS)
  * [2. Configure Flask app in Docker](#2-Configure-Flask-app-in-Docker)
  * [3. Run the Flask app in Docker](#3-run-the-flask-app-in-Docker)
  * [4. Kill the container ](#4-Kill-the-container)


3. Run the Flask app 

<!-- tocstop -->
## Project Charter
#### Vision
This project aims to help airline companies to predict whether customers have a positive opinion or negative opinion about their flights and services and experience based on tweets; 
The airline companies also want to identify the customer loyalty, improve upon negative terms and at the same keep their most appreciated aspect intact.

#### Mission
To achieve the mission, we build up a classification model to predictive whether the customer feels positive, negative or neutral towards the airline experience based on the twitters airline textual data. 

#### Success criteria
The machine learning performance metric are AUC, correct classification rate, precision, recall and false negative rate  and the model aims to achieve 80% of these metrics. 

As for business performance metric, we want to use customer loyalty score (The higher customer loyalty, the larger life-time value of the client).

Data Source: Twitter U.S. Airline Sentiment data from Kaggle. 
https://www.kaggle.com/crowdflower/twitter-airline-sentiment



<!-- tocstop -->
## Project Planning

#### Initiative: 
Explore the textual features in Twitters’ passengers’ comments with positive and negative attitude to provide insights on top frequent positive and negative tokens
Develop classification methods for identifying whether a given tweet will have positive or negative attitude based on the pre-processed text features
Continuously predict the sentiment score, in order to timely identify unsatisfied customers and negative  and implement strategy to improve.

**Epic 1**: Process and explore text features and understand their distributions 

  * **Story 1**: Using Natural Language Toolkit to perform tokenization, removing uninformative punctuation, removing stopwords, and performing stemming and lemmatization.

  * **Story 2**: Display distribution of different text attributes across positive and negative groups to see if there is a differentiated pattern. Examine how the negative and positive words are distributed across different airlines. Highlight those most frequent tokens, and group these most common tokens by sentiment. Identify those tokens that are most discriminative 

  * **Story 3**: Conduct feature engineering via bag-of-word model, and prepare the design matrix 

**Epic 2**: Predict whether a customer’s next comment will be positive or negative.

  * **Story 1**: Use a baseline model to select most important features that contribute to predict the customer’s sentiment

  * **Story 2**: Model this classification problem using selected features through different model approaches.

  * **Story 3**: Tune hyperparameters (model complexity and performance-wise) to achieve the success criteria.

  * **Story 4**: As new customer’ tweets being entered in the app, it will predict the probability of the customers’ tweet being positive and return the class of sentiment. Possibly highlight the key words he or she mentions that disclose the attitude.

**Epic 3**: Deploy the model and generate the airline sentiment prediction App. Keep track of customers' comments sentiment changes as more airline comments are added.

  * **Story 1**: Examine the top frequent features each trial/quarter/year as new comments are added through the app on a continuous basis. The airline company want to identify how the general public see and say their services(understand what percentage of customers are disappointing with their service each year, and the top reasons. If the complaints in one aspect are severe, the airline should take initiatives to aks for further feedback and reduce the customer loss.


#### Backlog:
  1.  Initiative.epic1.story1: General Data Preprocessing (2 points)
  2.  Initiative.epic1.story2: Exploratory Feature Analysis (2 points)
  3.  Initiative.epic1.story3: Feature Engineering  (2 points)
  4.  Initiative.epic2.story1: Feature Selection (1 point)
  5.  Initiative.epic2.story2: Model Fitting (4 points)
  6.  Initiative.epic2.story3: Hyperparameter Tuning (4 points)
  7.  Initiative.epic2.story4: Prediction (4 points)
  8.  Initiative.epic3.story1: Deployment (8 points)

#### Icebox:
  1.  Develop the App and hopefully put it into AWS.

       * **Time Estimation**: 8 points

       * Will break this icebox down as become more familiar with AWS and App development.

  2.  Augment the training data, the tweets for Airline into the app. The updated information shall be incorporated into the new model each month/quarter/year.

       * **Time Estimation**: 8 points

  3.  Continuously examine the top frequent positive and negative words and see how they change dynamically

       * **Time Estimation**: 8 points since a large time interval required to see trends.


<!-- tocstop -->
## Repo structure 

```
├── README.md                         <- You are here
├── app
│   ├── static/                       <- CSS, JS files that remain static
│   ├── templates/                    <- HTML (or other code) that is templated and changes based on a set of inputs
│   ├── boot.sh                       <- Start up script for launching app in Docker container.
│   ├── Dockerfile                    <- Dockerfile for building image to run app  
│
├── config                            <- Directory for configuration files 
│   ├── local/                        <- Directory for keeping environment variables and other local configurations that *do not sync** to Github 
│   ├── logging/                      <- Configuration of python loggers
│   ├── flaskconfig.py                <- Configurations for Flask API 
│
├── data                              <- Folder that contains data used or generated. Only the external/ and sample/ subdirectories are tracked by git. 
│   ├── output/                       <- Model pipeline output, feature importance plot, evaluation metrics, etc
│   ├── sample/                       <- Sample data used for code development and testing, will be synced with git
    ├── Tweets.csv
    ├── Tweets_clean.csv
    ├── Tweets_test.csv
    ├── Tweets_train.csv
│   ├── sentiment_class_prediction.pkl.csv
    ├── tweet_sentiment.db

├── deliverables/                     <- Any white papers, presentations, final work products that are presented or delivered to a stakeholder 
│
├── docs/                             <- Sphinx documentation based on Python docstrings. Optional for this project. 
│
├── figures/                          <- Generated graphics and figures to be used in reporting, documentation, etc
│
├── models/                           <- Trained model objects (TMOs), model predictions, and/or model summaries
│
├── notebooks/
│   ├── archive/                      <- Develop notebooks no longer being used.
│   ├── deliver/                      <- Notebooks shared with others / in final state
│   ├── develop/                      <- Current notebooks being used in development.
│   ├── template.ipynb                <- Template notebook for analysis with useful imports, helper functions, and SQLAlchemy setup. 
│
├── reference/                        <- Any reference material relevant to the project
│
├── src/                              <- Source data for the project 
│
├── test/                             <- Files necessary for running model tests (see documentation below) 
│
├── app.py                            <- Flask wrapper for running the model 
├── run.py                            <- Simplifies the execution of one or more of the src scripts  
├── requirements.txt                  <- Python package dependencies 
```

<!-- tocstop -->
## Build the image and run the model pipeline

```

export AWS_ACCESS_KEY_ID=AKIAIKSCD4XUY7IJ6QLQ

export AWS_SECRET_ACCESS_KEY=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

```

```

docker build -f Dockerfile -t tweet_sentiment .

```

Execute all model pipeline using makefile. Here the dockerfile has entry point make.
```

docker run --mount type=bind,source="$(pwd)"/data,target=/app/data --env AWS_ACCESS_KEY_ID --env AWS_SECRET_ACCESS_KEY tweet_sentiment all

```


Execute the unit test

```

docker run tweet_sentiment test

```


<!-- tocstop -->
## Running the app
### 1. Initialize the database

(You can skip this part if the database is already created) 

#### Create the database with a single tweet entry

##### 1.1 Local SQLite database 

To create the database in the location configured in `config.py` with one initial song, run: 


```

cd 2020-msia423-Zhang-Haonan

docker build -f Dockerfile-py3 -t models .

docker run --mount type=bind,source="$(pwd)"/data,target=/app/data models src/db_models.py

```


##### 1.2 Create an AWS RDS

Note: You will need to be on the Northwestern VPN for the remaining portions of the tutorial._

###### (1). Edit your mysql config file accordingly 

```bash
cd 2020-msia423-Zhang-Haonan

vi .mysqlconfig
```

This .mysqlconfig is the file that contains your database information for creating a table tweet_sentiment_prediction.

* Set `MYSQL_USER` to the "master username" that you used to create the database server.
* Set `MYSQL_PASSWORD` to the "master password" that you used to create the database server.
* Set `MYSQL_HOST` to be the RDS instance endpoint from the console
* Set `MYSQL_HOST` to be `3306`
* Set `DATABASE_NAME` = msia423_db
 

Set the environment variables in your `~/.bashrc`

```bash
echo 'source .mysqlconfig' >> ~/.bashrc
source ~/.bashrc
```

**VERIFY THAT YOU ARE ON THE NORTHWESTERN VPN BEFORE YOU CONTINUE ON**


###### (2). Creating schema in your database (msia423_db) with SQLAlchemy

Build the provided Docker image 

```bash
docker build -f Dockerfile-py3 -t tweet_sentiment_mysql .
```

Set your database name and run the Docker container 

```bash
##export DATABASE_NAME=msia423_db
sh run_docker.sh
```

<!-- tocstop -->
### 2. Configure Flask app in Docker

`config/flaskconfig.py` holds the configurations for the Flask app. It includes the following configurations:

```python
DEBUG = True  # Keep True for debugging, change to False when moving to production 
LOGGING_CONFIG = "config/logging/local.conf"  # Path to file that configures Python logger
HOST = "0.0.0.0" # the host that is running the app. 0.0.0.0 when running locally 
PORT = 5000  # What port to expose app on. Must be the same as the port exposed in app/Dockerfile 
SQLALCHEMY_DATABASE_URI = 'sqlite:///data/tweet_sentiment.db'  
or '{dialect}://{user}:{pw}@{host}:{port}/{db}'.format(dialect=conn_type, user=user,
                                            pw=password, host=host, port=port,
                                      db=DATABASE_NAME) # URI (engine string) for database that contains tracks
APP_NAME = "twitter-sentiment"
SQLALCHEMY_TRACK_MODIFICATIONS = True 
SQLALCHEMY_ECHO = False  # If true, SQL for queries made will be printed
MAX_ROWS_SHOW = 10 # Limits the number of rows returned from the database 
```


<!-- tocstop -->
### 3. Run the Flask app in Docker

To run the Flask app, run: 


#### 3.1. If you use local database to ingest records from website html:
```

cd ~/2020-msia423-Zhang-Haonan

export SQLALCHEMY_DATABASE_URI='sqlite:///data/tweet_sentiment.db'

```

Build up the docker image

The Dockerfile for running the flask app is in the `app/` folder. To build the image, run from this directory (the root of the repo): 

```

docker build -f app/Dockerfile -t tweet .

```

Docker run command to access the webpage.
```

docker run --mount type=bind,source="$(pwd)"/data,target=/app/data --env SQLALCHEMY_DATABASE_URI -p 5000:5000 --name test tweet

```


#### 3.2 if you use AWS RDS:


##### 3.2.1 Edit your mysql config file accordingly 


```bash

cd ~/2020-msia423-Zhang-Haonan

cd config

vi mysqlconfig.env

```

This mysqlconfig.env is the file under config folder. It is used when you want to build up connection between 
your app and your database. If you have not created a database in RDS, please check section 1.2 Create an AWS RDS

* Set `MYSQL_USER` to the "master username" that you used to create the database server.
* Set `MYSQL_PASSWORD` to the "master password" that you used to create the database server.
* Set `MYSQL_HOST` to be the RDS instance endpoint from the console
* Set `MYSQL_HOST` to be `3306`
* Set `DATABASE_NAME` = msia423_db
 


##### 3.2.2. Run the apps and use AWS RDS 

The Dockerfile for running the flask app is in the `app/` folder. To build the image, run from this directory (the root of the repo): 

```bash

cd ~/2020-msia423-Zhang-Haonan

docker build -f app/Dockerfile -t tweet .

```

```

docker run --mount type=bind,source="$(pwd)"/data,target=/app/data --env-file config/mysqlconfig.env -p 5000:5000 --name test tweet

```

You should now be able to access the app at http://0.0.0.0:5000/ in your browser.


This command runs the `tweet` image as a container named `test` and forwards the port 5000 from container to your laptop so that you can access the flask app exposed through that port. 

If `PORT` in `config/flaskconfig.py` is changed, this port should be changed accordingly (as should the `EXPOSE 5000` line in `app/Dockerfile`)

You need to change the name <test> whenever it is occupied.


If now you want read what you write to your RDS:

You can use the MySQL client again to see that a table has been added and data generated.
You can run the Docker container by using the `run_mysql_client.sh` script.
Change the mysql configuration.

```
vi .mysqlconfig
```

* Set `MYSQL_USER` to the "master username" that you used to read the database server(msia423instructor).
* Set `MYSQL_PASSWORD` to the "master password" that you used to create the database server.
* Set `MYSQL_HOST` to be the RDS instance endpoint from the console
* Set `MYSQL_HOST` to be `3306`
* Set `DATABASE_NAME` = msia423_db

(please check the submission engine string to set these environment variable, MYSQL_USER=msia423instructor)

 
Set the environment variables in your `~/.bashrc`

```bash
echo 'source .mysqlconfig' >> ~/.bashrc
source ~/.bashrc
```

```bash
sh run_mysql_client.sh
```

```
show databases;

return:

+--------------------+
| Database           |
+--------------------+
| information_schema |
| innodb             |
| msia423_db         |
| mysql              |
| performance_schema |
| sys                |
+--------------------+

use msia423_db;

show tables;

return:

+----------------------------+
| Tables_in_msia423_db       |
+----------------------------+
| tweet_sentiment            |
| tweet_sentiment1           |
| tweet_sentiment2           |
| tweet_sentiment4           |
| tweet_sentiment5           |
| tweet_sentiment6           |
| tweet_sentiment_prediction |
+----------------------------+

select * from tweet_sentiment_prediction;

return:

+--------------------------------------------------------------------------+-------------------+---------------------------------------+---------------------------------------+
| text                                                                     | airline_sentiment | airline_sentiment_positive_confidence | airline_sentiment_negative_confidence |
+--------------------------------------------------------------------------+-------------------+---------------------------------------+---------------------------------------+
| @VirginAmerica plus you've added commercials to the experience... tacky. | positive          | 0.6514                                | 0.3486                                |
| I love analytics value chain!                                            | positive          | 0.9047008828580841                    | 0.09529911714191586                   |
| The analytics value chain is a fruitful course!                          | negative          | 0.1946295273040157                    | 0.8053704726959843                    |
+--------------------------------------------------------------------------+-------------------+---------------------------------------+---------------------------------------+




```

tweet_sentiment_prediction is the table that will include the newly injected value from the webapp

Whenever you use the web app to add a new entry, it will be reflected in the aws rds. 


<!-- tocstop -->
### 4. Kill the container 

Once finished with the app, you will need to kill the container. To do so: 

```bash
docker kill test 
```

where `test` is the name given in the `docker run` command.
