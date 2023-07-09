.PHONY: all test 

all: acquire clean train_model score_model Evaluation model_inference


acquire:
	python3 run.py acquire --config=config/test.yaml --output=data/Tweets.csv

clean:
	python3 run.py clean --config=config/test.yaml --input=data/Tweets.csv --output=data/Tweets_clean.csv

train_model:
	python3 run.py train_model --config=config/test.yaml --input=data/Tweets.csv --output=data/Tweets_test.csv --output1=data/Tweets_train.csv

score_model:
	python3 run.py score_model --config=config/test.yaml --input=data/Tweets.csv --output=data/output/test_prediction.csv 


Evaluation:
	python3 run.py Evaluation --config=config/test.yaml --input=data/Tweets.csv --input1=data/output/test_prediction.csv --output=data/output/confusion_matrix.csv --output1=data/output/metrics_performance.csv


model_inference:
	python3 run.py model_inference --config=config/test.yaml --input=data/Tweets.csv --output=data/output/model_inference.csv


test:
	pytest test/test.py





