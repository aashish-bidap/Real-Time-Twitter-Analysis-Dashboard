from time import sleep
from json import dumps
from kafka import KafkaProducer
from kafka import KafkaConsumer, KafkaProducer
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
from datetime import datetime, timedelta
from datetime import datetime, timedelta
from tensorflow import keras 
from config import *
from gensim.models import Word2Vec,KeyedVectors
import tweepy
import time
import json
import os
import pandas as pd
import re
import numpy as np

KAFKA_BROKER_URL = os.environ.get('KAFKA_BROKER_URL')
topic_name = 'topic_test'

# Creating the authentication object
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)

# Setting your access token and secret
auth.set_access_token(access_token, access_token_secret)

# Creating the API object by passing in auth information
api = tweepy.API(auth)


producer = KafkaProducer(
    bootstrap_servers=KAFKA_BROKER_URL,
    value_serializer=lambda value: json.dumps(value).encode('utf-8'),
)

def token_check(x,model):
    """
        1.Check if the token exists in the word2vec model vocab. 
        2.Check if the length of the token is greater than 3 
    """
    my_list=[]
    for i in x:
        if len(i) >= 3 and i in model.vocab:
            my_list.append(i)
        else:
            continue
    return my_list

def word_vector(tokens, size,model):
    """
        1.Averaging the word vectors 
    """
    vec = np.zeros(size).reshape((1, size))
    count = 0
    for word in tokens:
        vec += model[word].reshape((1, size))
        count += 1.
    if count != 0:
        vec /= count
    return vec

def remove_pattern(input_txt, pattern):
    """
        Method for removing specific input pattern from the 
        input text for cleaning tweets.
    """
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)
    return input_txt

def clean_tweet_content(tweet):
    """
        Cleaning the tweets.
    """
    print("convert to lowercase")
    tweet = tweet.lower()
 
    print("removing the tweets username.")
    tweet = np.vectorize(remove_pattern)(tweet, "@[\w]*")
    
    print("removing all the RT: text from the Tweets")
    tweet = np.vectorize(remove_pattern)(tweet, "RT :")

    print("removing links")
    tweet = np.vectorize(remove_pattern)(tweet, "r'^https?:\/\/.*[\r\n]*'")
    
    tweet = np.vectorize(remove_pattern)(tweet,"\n")
    
    tweet = np.array_str(tweet) 
    
    tweet = tweet.replace('[^a-zA-Z#]',' ')

    return tweet

def prediction(prediction):
    """
        Prediction with a threshold = 0.5
    """
    for i in prediction:
        for j in i:
            if j > 0.5:
                return "Positive"
            else:
                return "Negative"

def get_twitter_data():

    #load Sentiment Classifier model-
    model_NN = keras.models.load_model('./Twitter_Sentiment_NN_model.h5')

    #Load Word2Vec Google Word Embeeding Model
    model = KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin.gz',binary=True,limit=100000)

    resp = api.search(SEARCH_TERM,tweet_mode='extended')

    for twe in resp:
        record = {}

        if twe.lang == 'en':

            #Basic User Fields
            record['user_id'] = str(twe.user.id_str)
            tweet = str(twe.full_text)
            record['create_date'] = str(twe.created_at)
            record['tweet']= str(tweet)
            record['location']= str(twe.user.location)
            record['fav_count']= str(twe.favorite_count)
            record['retweet_count']= str(twe.retweet_count)
            record['verified_account']= str(twe.user.verified)
            
            #Sentiment Prediction
            clean_tweet = clean_tweet_content(tweet)
            tweet_tokens = list(clean_tweet.split(" "))
            tweet_tokens_filtered = token_check(tweet_tokens,model)
            _arrays = np.zeros((1, 300))
            _arrays[0,:] = word_vector(tweet_tokens_filtered,300,model)
            vectorized_array = pd.DataFrame(_arrays)
            pred = model_NN.predict([vectorized_array.iloc[:,0:300]])
            record['Sentiment'] = prediction(pred)

            #HashTags & User_Mentions
            hash_tags=[]
            user_mentions=[]
            for key,val in twe.entities.items():
                if len(twe.entities[key]) > 0:
                    for attr in twe.entities[key]:
                        for key,val in attr.items():
                            if key == 'text':
                                hash_tags.append(val)
                            elif key == 'screen_name':
                                user_mentions.append(val)

            record['hashtags'] = hash_tags
            record['user_mentions'] = user_mentions

            #Sending data to the topic
            producer.send(topic_name,value=record)


def periodic_work(interval):
    """
        Get new twitter tweets after specific time interval.
    """
    while True:
        get_twitter_data()
        time.sleep(interval)

periodic_work(60 * 0.1)  # get data every couple of minutes
