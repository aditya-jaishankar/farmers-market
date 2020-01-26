# %% [markdown]
# # Training a binary classifier to identify accounts that are likely 
# commercial business vs those that are likely real human users
# 
# The goal of this file is to be able to identify which of the followers that I
#  selected are commercial followers or otherwise small businesses. This 
# corrupts my input user base with non-people, so this is an attempt to remove
# these followers.
# %% [markdown]
# ## Imports

# %%
# General imports
import json
import glob
import pickle
import collections
import random
from tqdm import tqdm as tqdm
import config
import time
import os
dirpath = os.path.dirname(os.path.realpath('__file__'))
from pprint import pprint

# import logging
# logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)
# logging.root.level = logging.INFO

# NLP imports
import nltk
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['https', 'http'])
import re
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel
import spacy
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

# To check later if a words is in english or not. Note that to include some
# additional words as stop words, I just removed them from this dictionary
with open('./words_dictionary.json') as filehandle:
    words_dictionary = json.load(filehandle)
english_words = words_dictionary.keys()

# Visualization imports
import pyLDAvis
import pyLDAvis.gensim
pyLDAvis.enable_notebook()
import matplotlib.pyplot as plt

# Other imports
import pandas as pd
import numpy as np
import tweepy

# %% [markdown]

## Loading the data generated in `commercial-markets-downloader.py`

# %%
with open('./data/dataset_tweets_random_users.data', 'rb') as filehandle:
    dataset_tweets_random_users = pickle.load(filehandle)

# %% [markdown]

# Note that the structure of the `dataset_tweets_random_users` dictionary is 
# ```
# {
#     screen_name: {
#                     'tweets': [{tweet1_json}, ..., {tweet}],
#                     'label': 1
#                  }
# }
# ```

# We now want to build all the usual data cleaning tokenziation, lemmatization
# pipeline and then build the corpus of words. For this, I am just copying and
# functions from `LDA.ipynb`. Ideally, I would want to have a `utils` file and
# then just import functions as needed from `utils.py`. With more time, I can
# probably clean up my code base significantly. Dividing the code into those 
# that download data and then export things a dictionaries, and those that 
# utilize functions from `utils.py` to keep everything clean and modular.

# I finally want a dictionary of the form

# ```
# {
#     user: {
#               'hashtags': [..., ..., ...],
#               'full_text': [cleaned, tokenized, lemmatized, words of tweets],
#               'label': 0 or 1 
#           }
# }
# ```

# First define some utility functions

# %%
def get_user(tweet):
    """
    input: tweet dictionary
    returns: return the username
    """
    return tweet['user']['screen_name']


def get_hashtag_list(tweet):
    """
    input: tweet dictionary
    returns: list of all hashtags in both the direct tweet and the
    retweet 
    """

    l = []
    for d in tweet['entities']['hashtags']:
        l += [d['text']]

    if 'retweeted_status' in tweet.keys():
        for d in tweet['retweeted_status']['entities']['hashtags']:
            l += [d['text']]
    return l


def tokenizer_cleaner_nostop_lemmatizer(text):
    """
    This function tokenizes the text of a tweet, cleans it off punctuation,
    removes stop words, and lemmatizes the words (i.e. finds word roots to remove noise)
    I am largely using the gensim and spacy packages 

    Input: Some text
    Output: List of tokenized, cleaned, lemmatized words
    """

    tokenized_depunkt = gensim.utils.simple_preprocess(text, min_len=4, deacc=True)
    tokenized_depunkt_nostop = ([word for word in tokenized_depunkt 
                                 if (word not in stop_words and word in english_words)])
    
    # Lemmatizer while also only allowing certain parts of speech.
    # See here: https://spacy.io/api/annotation
    allowed_pos = ['ADJ', 'ADV', 'NOUN', 'PROPN','VERB']
    doc = nlp(' '.join(tokenized_depunkt_nostop))
    words_final = [token.lemma_ for token in doc if token.pos_ in allowed_pos]
    return words_final

    
def get_tweet_words_list(tweet):
    """
    This function takes in a tweet and checks if there is a retweet associated 
    with it. It then returns a list of tokenized words without punctuation.
    input: tweet
    output: list of tokenized words without punctuation
    """

    text = tweet['full_text']
    clean_words = tokenizer_cleaner_nostop_lemmatizer(text)
    
    if 'retweeted_status' in tweet.keys():
        retweet_text = tweet['retweeted_status']['full_text']
        retweet_clean_words = tokenizer_cleaner_nostop_lemmatizer(retweet_text)
        clean_words += retweet_clean_words
    return clean_words

# %%
lda_dict_random_users = {}
for user in tqdm(dataset_tweets_random_users):
    lda_dict_random_users[user] = {}
    lda_dict_random_users[user]['hashtags'] = []
    lda_dict_random_users[user]['fulltext'] = []

    
    tweets = dataset_tweets_random_users[user]['tweets']
    label = dataset_tweets_random_users[user]['label']

    for tweet in tweets:
        hashtags = get_hashtag_list(tweet)
        words = get_tweet_words_list(tweet)

        lda_dict_random_users[user]['hashtags'].extend(hashtags)
        lda_dict_random_users[user]['fulltext'].extend(words)

with open('./data/lda_dict_random_users.data', 'wb') as filehandle:
    pickle.dump(lda_dict_random_users, filehandle, 
                protocol=pickle.HIGHEST_PROTOCOL)



