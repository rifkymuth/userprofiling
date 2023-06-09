import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import io
import torch
import pandas as pd
import numpy as np
from models.BERTopic import BERTopicModel
from models.IndoBERTModel import IndoBERTModel
from utils.datapreprocessing import data_preprocessing

from flask import Flask, request, jsonify

bertopic = BERTopicModel()
indobert = IndoBERTModel()
data_preprocessor = data_preprocessing()
model_path = "model_indobert_sentiment_analysis.pkl"


def preprocess_raw_file(raw_text_file):
    dataframe = pd.read_csv(raw_text_file)
    dataframe["tweet"] = dataframe["tweet"].astype(str)
    '''
    TODO
    dataframe["tweet"] = dataframe["tweet"].astype(str)
    '''
    return dataframe

def text_cleansing(tweet_post):
    return data_preprocessor.text_cleansing(tweet_post)

def topic_modeling_predict(tweet_post_cleansed):
    topics, topic_labels = bertopic.predic_topic(tweet_post_cleansed)
    return pd.Series(data=topics), topic_labels # df["topic"]

def sentiment_analysis_predict(tweet_post):
    indobert.load_model(model_path)
    predicted = indobert.predict(tweet_post)

    return pd.Series(data=predicted).sub(1) # df["sentimen"]

def user_profile(raw_text_file):
    dataframe = preprocess_raw_file(raw_text_file)
    tweet_post_cleansed = text_cleansing(dataframe['tweet'])
    dataframe['topic'], topic_labels = topic_modeling_predict(tweet_post_cleansed)
    dataframe['sentimen'] = sentiment_analysis_predict(dataframe['tweet'])

    result = dataframe.groupby('topic')['sentimen'].value_counts(ascending=True, normalize=True).mul(100).round(1).to_dict()

    response = {}

    for key in topic_labels:
        response[key] = {}

    for key in result:
        i = key[0]
        j = key[1]
        response[i]['top_words'] = topic_labels[i].split("_")[1:]
        response[i][j] = result[key]

    response.pop(-1)
    return response

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get('file')
        if file is None or file.filename == "":
            return jsonify({"error": "no file"})

        try:
            raw_text_file = file
            result = user_profile(raw_text_file)
            return jsonify(result)
        except Exception as e:
            return jsonify({"error": str(e)})

    return "OK"


if __name__ == "__main__":
    app.run(debug=True)