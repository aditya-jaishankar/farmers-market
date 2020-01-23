# %%
# General imports
import json
import glob
import pickle
import collections
import random
from tqdm import tqdm as tqdm
import config
import os
dirpath = os.path.dirname(os.path.realpath('__file__'))
import time

# NLP imports
import nltk
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'http', 'https'])
import re
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel
import spacy
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

# To check later if a words is in english or not
with open('./words_dictionary.json') as filehandle:
    words_dictionary = json.load(filehandle)
english_words = words_dictionary.keys()

# Visualization imports
import pyLDAvis
import pyLDAvis.gensim
import matplotlib.pyplot as plt

# Other imports
import pandas as pd
import numpy as np
import tweepy


# %%

# Defining some helper functions
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
                                 if word not in stop_words and word in english_words])
    
    # Lemmatizer while also only allowing certain parts of speech.
    # See here: https://spacy.io/api/annotation
    allowed_pos = ['ADJ', 'ADV', 'NOUN', 'PROPN','VERB']
    doc = nlp(' '.join(tokenized_depunkt_nostop))
    words_final = [token.lemma_ for token in doc if token.pos_ in allowed_pos]
    return words_final

    
def get_tweet_words_list(tweet):
    """
    This function takes in a tweet and checks if there is a retweet associated with it
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
# Changing up the format of the dictionary

# with open('./data/all_tweets_dict.data', 'rb') as filehandle:
#     all_tweets_data = pickle.load(filehandle)

# master_dict = {}

# for market in all_tweets_data:
#     followers = all_tweets_data[market]
#     master_dict[market] = {}

#     for follower in tqdm(followers):
#         tweets = all_tweets_data[market][follower] # list of tweet_.json
#         master_dict[market][follower] = {}
#         master_dict[market][follower]['hashtags'] = []
#         master_dict[market][follower]['fulltext'] = []
#         for tweet in tweets:
#             hashtags = get_hashtag_list(tweet)
#             words = get_tweet_words_list(tweet)
            
#             master_dict[market][follower]['hashtags'].extend(hashtags)
#             master_dict[market][follower]['fulltext'].extend(words)
# print('wait')

# with open('./data/master_dict.data', 'wb') as filehandle:
#     pickle.dump(master_dict, filehandle, protocol=pickle.HIGHEST_PROTOCOL)

# # %% [markdown]

# # Doing the actual LDA

# # # %% 
with open('./data/master_dict.data', 'rb') as filehandle:
    master_dict = pickle.load(filehandle)

def get_docs(d, market):
    """
    Accepts a market and then returns the documents for the market. A document
    is a list of of word lists for each user in the market city i.e. it is a list of lists.
    Each outer list is a follower and the innner list is the cleaner, tokenized, depunkt, 
    lematized set of words for that follower.
    """
    docs = []
    for user in d[market]:
        text_list = d[market][user]['fulltext']
        docs.append(text_list)
    return docs

# for market in master_dict:
markets = list(master_dict.keys())
docs = get_docs(master_dict, markets[0])
id2word = corpora.Dictionary(docs)
corpus = [id2word.doc2bow(doc) for doc in docs]

# with open('./data/corpus_market0.data', 'wb') as filehandle:
#     pickle.dump(corpus, filehandle, protocol=pickle.HIGHEST_PROTOCOL)

# with open('./data/id2word_market0.data', 'wb') as filehandle:
#     pickle.dump(id2word, filehandle, protocol=pickle.HIGHEST_PROTOCOL)
def compute_lda(corpus, id2word, alpha=.01):
    """
    Performs the LDA and returns the computer model.
    Input: Corpus, dictionary and hyperparameters to optimize
    Output: the fitted/computed LDA model
    """
    lda_model = gensim.models.ldamulticore.LdaMulticore(corpus=corpus, 
                                                        id2word=id2word,
                                                        num_topics=22,
                                                        random_state=100,
                                                        # update_every=1,
                                                        chunksize=10,
                                                        passes=50,
                                                        alpha=alpha,
                                                        per_word_topics=True)
    return lda_model

def main():
    coherence_scores = []
    alpha_range = [.001, .005, .01, .05, .1]
    for alpha in tqdm(alpha_range):
        lda_model = compute_lda(corpus, id2word, alpha=alpha)

        coherence_model_lda = CoherenceModel(model=lda_model,
                                            texts=docs,
                                            dictionary=id2word,
                                            coherence='c_v')
        coherence_lda = coherence_model_lda.get_coherence()
        coherence_scores.append(coherence_lda)
        print('alpha = ', alpha, ', coherence score:', coherence_lda)
    return coherence_scores

if __name__ == "__main__":
    alpha_range = [.001, .005, .01, .05, .1]
    coherence_scores = main()
    plt.plot(alpha_range, coherence_scores)
    plt.show()




