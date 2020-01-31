# %%
# imports

import spacy
import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
import warnings
import gensim
from pprint import pprint
from tqdm import tqdm

# %%
# File loads
nlp = spacy.load('en_core_web_lg')
categories = pd.read_pickle('./data/categories_final_cleaned_filtered.df')
categories.head()

# %%
# Defining functions

def drop_check(category):
    """
    
    """
    drop_words = ['me my', 'cheese dishes', 'fruit juice', 'animal hair',
                  'food technology', 'fish sauce', 'wedding photography',
                  'economic law', 'student awards', 'know nothing',
                  'cancer patients', 'food ingredients', 'do it yourself',
                  'dead like me', 'bad girls characters', 'me my songs']
    return (category.lower() not in drop_words)

def get_doc2vec_matrix(df):
    """
    Accepts: a dataframe containing the doc2vec vectors for wikipedia categories
    Returns: The normalized (-1, 300) doc2vec matrix
    """
    
    matrix = np.array([df['doc2vec']]).reshape(-1, 300)
    matrix_norm = normalize(matrix, axis=1)
    return matrix_norm


def get_minimum_distance_category(matrix, topic_words):
    """
    Input: The normalized doc2vec matrix for wikipedia categories, the words
           that constitute a topic.
    Returns: the argmax of cosine similarity, which can be used to extract the
             text
    """
    terms_vector = normalize(nlp(topic_words).vector.reshape(300, 1), axis=0)
    prod = np.matmul(matrix, terms_vector)
    a = np.argmax(prod)
    return categories['categories'].iloc[a]


def get_topic_words(lda_model):
    """
    Given a topic model, display all the words
    Input: An lda topic model
    returns: a list of words in the topic
    """
    all_topics = lda_model.show_topics(num_topics=12, num_words=30,
                                       formatted=False)
    topics_words = [(topic[0], [word[0] for word in topic[1]]) 
                        for topic in all_topics]

    return topics_words


# %%
# Section that loads a topic and generates the word list

tqdm.pandas()
categories = categories[categories.apply(lambda row:
                drop_check(row['categories']), axis=1)]

matrix = np.array([categories['doc2vec']]).reshape(-1, 300).astype('float32')

topic_labels = {}
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for market_index in range(7):
        topic_labels[market_index] = {}
        filename = './ldamodels/market' + str(market_index) + '/model.model'
        lda_model = gensim.models.ldamodel.LdaModel.load(filename)
        all_topics_words = get_topic_words(lda_model)
        for topic, words in all_topics_words:
            words_string = ' '.join(words)
            topic_label = get_minimum_distance_category(matrix, words_string)
            topic_labels[market_index][topic] = topic_label
pprint(topic_labels)
            
# %%
