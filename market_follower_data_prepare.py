# %% 
# Imports

import json
import pickle
import random
import os
from tqdm import tqdm as tqdm
import config


import tweepy

# NLP imports
from nltk.corpus import stopwords
import gensim
import gensim.corpora as corpora
import spacy

dirpath = os.path.dirname(os.path.realpath('__file__'))
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
stop_words = stopwords.words('english')
stop_words.extend(['https', 'http'])

# To check later if a words is in english or not. Note that to include some
# additional words as stop words, I just removed them from this dictionary
with open('./words_dictionary.json') as filehandle:
    words_dictionary = json.load(filehandle)
english_words = words_dictionary.keys()


# %%
# Defining functions

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
    removes stop words, and lemmatizes the words (i.e. finds word roots to 
    remove noise) I am largely using the gensim and spacy packages 

    Input: Some text
    Output: List of tokenized, cleaned, lemmatized words
    """

    tokenized_depunkt = gensim.utils.simple_preprocess(text, min_len=4, 
                                                       deacc=True)
    tokenized_depunkt_nostop = ([word for word in tokenized_depunkt 
                                 if (word not in stop_words and word in 
                                     english_words)])
    
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
# Twitter credentials


consumer_key = config.consumer_key
consumer_secret = config.consumer_secret
auth = tweepy.AppAuthHandler(consumer_key, consumer_secret)

api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

# %%
# Load the pre-saved dict containing all the tweets from the market

with open('./data/market_tweets_dict.data', 'rb') as filehandle:
    market_tweets_dict = pickle.load(filehandle)

# Include the tweets only if there are more than 500, and then add label 1

dataset_tweets_markets_followers = {}
for market in tqdm(market_tweets_dict):
    if len(market_tweets_dict[market]) >= 500:  # If market has >=500 tweets
        dataset_tweets_markets_followers[market] = {}
        tweets = market_tweets_dict[market]
        dataset_tweets_markets_followers[market]['label'] = 1
        dataset_tweets_markets_followers[market]['tweets'] = []
        for tweet in tweets:
            words = get_tweet_words_list(tweet)
            dataset_tweets_markets_followers[market]['tweets'] += words
            
# %% 
# Now we import the tweets from the followers dictionary that has been
# pre-saved. Note that we only need about 500. We use master_dict.data for this
# We go about this by picking a market at random and then a follower at random

with open('./data/master_dict.data', 'rb') as filehandle:
    master_dict = pickle.load(filehandle)

# %%
# Find 500 random followers from a random market.

master_dict_keys = list(master_dict.keys())

for _ in range(500):
    random_market = random.choice(master_dict_keys)

    follower_keys = list(master_dict[random_market].keys())
    random_follower = random.choice(follower_keys)
    tweets = master_dict[random_market][random_follower]['fulltext']
    dataset_tweets_markets_followers[random_follower] = {}
    dataset_tweets_markets_followers[random_follower]['tweets'] = tweets
    dataset_tweets_markets_followers[random_follower]['label'] = 0

# Write the dictionary to file
with open('./data/dataset_tweets_markets_followers.data', 'wb') as filehandle:
    pickle.dump(dataset_tweets_markets_followers, filehandle,
                protocol=pickle.HIGHEST_PROTOCOL)