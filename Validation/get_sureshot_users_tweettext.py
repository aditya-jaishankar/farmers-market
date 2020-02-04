# %%
# File to get the set of sureshot users and all their tweets. I will then run
# the LDA model trained on the followers data set on this set and on a random
# users dataset and find the matrix similarity between the each and the 
# followers dataset. If the similarity is higher, then we are good. 

# %%
# Imports

# General imports
import json
import pickle
from tqdm import tqdm as tqdm
import config

# NLP imports
from nltk.corpus import stopwords
import gensim
import spacy

# Other imports
import pandas as pd
import tweepy

stop_words = stopwords.words('english')
stop_words.extend(['https', 'http'])

with open('./words_dictionary.json') as filehandle:
    words_dictionary = json.load(filehandle)
english_words = words_dictionary.keys()
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

# %%
# Define functions


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
    allowed_pos = ['ADJ', 'ADV', 'NOUN', 'PROPN','VERB'] # try removing propn
    # allowed_pos = ['ADJ', 'ADV', 'NOUN','VERB'] # try removing propn and seeing
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
# Configure twitter API

consumer_key = config.consumer_key
consumer_secret = config.consumer_secret
auth = tweepy.AppAuthHandler(consumer_key, consumer_secret)

api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)


# %%
users = pd.read_csv('./data/list_of_sureshots.csv', header=None)
users

# %%
# Get tweets. Comment this out after the first download of the tweets

# dataset_tweets_sureshot_users = {}
# for user in tqdm(users[0]):
#     try:
#         tweets = tweepy.Cursor(api.user_timeline,
#                                screen_name=user,
#                                tweet_mode='extended',
#                                lang='en',
#                                count=1000).items(1000)
#         dataset_tweets_sureshot_users[user] = []
#         for tweet in tweets:
#             dataset_tweets_sureshot_users[user] += [tweet._json]
#     except:
#         pass

# with open('./data/dataset_tweets_sureshot_users.data', 'wb') as filehandle:
#     pickle.dump(dataset_tweets_sureshot_users, filehandle, 
#                 protocol=pickle.HIGHEST_PROTOCOL)

# %%
# Load the tweets, construct documents, etc. Can largely copy previous code

with open('./data/dataset_tweets_sureshot_users.data', 'rb') as filehandle:
    dataset_tweets_sureshot_users = pickle.load(filehandle)


# %%
dataset_tweettext_sureshot_users = {}
for user in tqdm(dataset_tweets_sureshot_users):
    tweets = dataset_tweets_sureshot_users[user]
    dataset_tweettext_sureshot_users[user] = []
    for tweet in tweets:
        words = get_tweet_words_list(tweet)

        dataset_tweettext_sureshot_users[user] += words


with open('./data/dataset_tweettext_sureshot_users.data', 'wb') as filehandle:
    pickle.dump(dataset_tweettext_sureshot_users, filehandle, 
                protocol=pickle.HIGHEST_PROTOCOL)