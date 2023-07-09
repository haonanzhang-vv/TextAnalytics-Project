FROM ubuntu:18.04

RUN apt-get update -y && apt-get install -y python3-pip python3-dev git gcc g++

WORKDIR /app

COPY ./requirements.txt /app/requirements.txt

RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt
RUN python3 -m nltk.downloader stopwords


ENV PYTHONIOENCODING=utf-8

COPY . /app

ENTRYPOINT [ "make" ]