# %%
# imports

import spacy
import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
import warnings
import gensim
from tqdm import tqdm
from pprint import pprint
import pickle

# %%
# File loads
nlp = spacy.load('en_core_web_lg')
# categories = pd.read_pickle('./data/categories_final_cleaned_filtered.df')

# %%
# Defining functions


def get_docvec(category):
    """
    input: a category string
    output: doc2vec vector calculated using spacy
    """
    # Consider doing a weighted mean instead of blind word2vec. Currently, this
    # implementation of doc2vec is just doing a mean of the individual word2vec
    doc = nlp(category)
    return doc.vector


def get_doc2vec_matrix(df):
    """
    Accepts: a dataframe containing the doc2vec vectors for wiki categories
    Returns: The normalized (-1, 300) doc2vec matrix
    """
    
    matrix = np.array([df['doc2vec']]).reshape(-1, 300)
    matrix_norm = normalize(matrix, axis=1)
    return matrix_norm


def get_minimum_distance_category(matrix, topic_words):
    """
    Input: The normalized doc2vec matrix for wikipedia categories, the words
           that constitute a topic.
    Returns: one of the top three topic labels, based on a cosine similarity 
            score
    """
    terms_vector = normalize(nlp(topic_words).vector.reshape(300, 1), axis=0)
    prod = np.matmul(matrix, terms_vector)
    prod_with_index = [(i, elem) for (i, elem) in enumerate(prod)]
    prod_sorted = sorted(prod_with_index, key=lambda tup: tup[1], reverse=True)
    a = np.random.randint(low=0, high=3)
    chosen_index = prod_sorted[a][0]
    return categories['categories'].iloc[chosen_index]


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


def generate_topic_labels(lda_model):
    """
    Accepts an lda_model for a particular market and then returns the labels
    for all of the topics for that particular market.
    Accepts: lda_model for a particular market
    returns: a dictionary of the format {topic_number: 'label}
    """
    all_topics_words = get_topic_words(lda_model)
    topic_labels = {}
    chosen_labels = []
    for topic, words in all_topics_words:
        words_string = ' '.join(words)
        unique_flag = True
        while unique_flag: # Make sure each topic has a unique label
            topic_label = get_minimum_distance_category(matrix, words_string)
            topic_labels[topic] = topic_label
            unique_flag = topic_label in chosen_labels
            if unique_flag:
                pass
            chosen_labels += [topic_label]
    return topic_labels


def get_topic_labels_for_markets():
    """
    Generates all topic labels across all markets for across all topic groups.
    """
    topic_labels_markets = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for market_index in range(7):
            topic_labels_markets[market_index] = {}
            filename = './ldamodels/market' + str(market_index) + '/model.model'
            lda_model = gensim.models.ldamodel.LdaModel.load(filename)
            topic_labels_markets[market_index] = generate_topic_labels(lda_model)
    return topic_labels_markets


def main():
    topics_dict = get_topic_labels_for_markets()
    # pprint(topics_dict)
    return topics_dict

# %%

if __name__ == '__main__':

    # Load the file
    # The twitter cateogry data was collected from:
    # https://www.97thfloor.com/blog/twitter-interest-category-list

    # Check out this list of events - maybe find category and then closest
    # match to the best possible event.
    # https://www.eventmanagerblog.com/event-ideas#community Also, try and scrape
    # https://www.eventmanagerblog.com/event-ideas It looks like there are a lot of
    # ideas that I can save as a dataframe to sample from. Even if it is just a 
    # single idea.

    categories = pd.read_csv('./data/new_categories_list.txt',
                            error_bad_lines=False,
                            warn_bad_lines=False)
    categories.columns = ['categories']
    categories['doc2vec'] = categories.apply(lambda row: 
                                                get_docvec(row['categories']),
                                                axis=1)

    matrix = (np.array([categories['doc2vec']]).reshape(-1, 300)
                        .astype('float32'))
    topics_dict = main()
    pprint(topics_dict)

    with open('./data/topic_labels_dict.data', 'wb') as filehandle:
        pickle.dump(topics_dict, filehandle, protocol=pickle.HIGHEST_PROTOCOL)

    # %%
    # I am also interested in finding out what kind of unique events keep showing
    # up, so I am going to run this maybe a 100 times, find all the events across 
    # the markets. Once I have a tagged list of topic labels, I can then select
    # them and see which ones occur so that I have topic labels for them.
    
    # unique = set()
    # for _ in tqdm(range(100)):
    #     topics_dict = main()
        
    #     for market_index in topics_dict.keys():
    #         labels = topics_dict[market_index].values()
    #         unique = unique.union(labels)
    # print(unique)
    # print('Number of unique topic labels: ', len(unique))

# %%
