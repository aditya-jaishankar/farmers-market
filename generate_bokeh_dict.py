# %%
# Imports

import spacy
import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
import warnings
import gensim
from tqdm import tqdm
from pprint import pprint
import pickle

import pandas as pd
from bokeh.plotting import show, figure
from bokeh.models import ColumnDataSource, LinearColorMapper, ColorBar, BasicTicker, PrintfTickFormatter, HoverTool
from bokeh.palettes import Viridis256
from bokeh.transform import transform
from bokeh.io import output_file

import matplotlib.colors as mcolors
from pprint import pprint


# %%
# Defining functions


def get_topic_words(lda_model):
    """
    Given a topic model, display all the words
    Input: An lda topic model
    returns: a list of words in the topic
    """
    all_topics = lda_model.show_topics(num_topics=7, num_words=15,
                                       formatted=False)
    topics_words = [[word[0] for word in topic[1]] 
                        for topic in all_topics]

    return topics_words


def get_number_of_documents_per_dominant_topic(lda_model, corpus, start=0, end=1):
    """
    Accepts: An lda_model and a corpus
    Returns: A dataframe that counts for each topic, how many documents had
    that particular topic number as its dominant topic.
    """
    corpus_sel = corpus[start:end]
    dominant_topics = []
    topic_percentages = []
    for i, corp in enumerate(corpus_sel):
        topic_percs, wordid_topics, wordid_phivalues = lda_model[corp]
        dominant_topic = sorted(topic_percs, key = lambda x: x[1], reverse=True)[0][0]
        dominant_topics.append((i, dominant_topic))
        topic_percentages.append(topic_percs)
    df = pd.DataFrame(dominant_topics, columns=['Document_Id', 'Dominant_Topic'])
    dominant_topic_in_each_doc = df.groupby('Dominant_Topic').size()
    df_dominant_topic_in_each_doc = dominant_topic_in_each_doc.to_frame(
                                                    name='count').reset_index()
    return(df_dominant_topic_in_each_doc)

# %%
# Loading the necessary files


for market_index in tqdm(range(7)):
    with open('./data/topic_labels_dict.data', 'rb') as filehandle:
        topic_labels_dict = pickle.load(filehandle)

    lda_model_filename = './ldamodels/market' + str(market_index) + '/model.model'
    lda_model = gensim.models.ldamodel.LdaModel.load(lda_model_filename)

    filename_corpus = './ldamodels/market' + str(market_index) + '/corpus.corpus'
    with open(filename_corpus, 'rb') as filehandle:
        corpus = pickle.load(filehandle)

    dominant_topic_df = get_number_of_documents_per_dominant_topic(
                                                        lda_model, corpus, end=-1)
    # %%

    topic_label_dict = topic_labels_dict[market_index]
    dominant_topic_df_labels = dominant_topic_df
    # Convert the topic index numbers to topic labels 
    dominant_topic_labels = dominant_topic_df.apply(lambda row: 
                                topic_label_dict[row['Dominant_Topic']], axis=1).to_list()
    # Find the count of document for each dominant topic
    dominant_topic_number_of_docs = dominant_topic_df['count'].to_list()

    # %%
    # Find words only for the dominant topics

    all_topic_words = get_topic_words(lda_model)
    selected_words = [all_topic_words[i] for i in dominant_topic_df['Dominant_Topic']]

    # %%
    # Write all the above lists to a dictionary that I can later import into Bokeh

    scale = 1
    counts = np.array(dominant_topic_number_of_docs)
    subject = [' ' for _ in range(len(dominant_topic_labels))]
    bokeh_dict = {'topic': dominant_topic_labels,
                'subject': subject,
                'counts': counts,
                'counts_scaled': counts/scale,
                'words': selected_words}

    filename = './data/bokeh_dict_' + str(market_index) + '.data'

    with open(filename, 'wb') as filehandle:
        pickle.dump(bokeh_dict, filehandle)
