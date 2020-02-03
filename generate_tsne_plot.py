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

from sklearn.manifold import TSNE
from bokeh.plotting import figure, output_file, show
from bokeh.models import HoverTool, ColumnDataSource
from bokeh.io import output_file
from bokeh.palettes import all_palettes

import matplotlib.colors as mcolors
#
# %%
# Defining functions


def get_topic_weights(lda_model, corpus):
    topic_weights = []
    for i, row_list in enumerate(lda_model[corpus]):
        topic_weights.append([weight for i, weight in row_list[0]])
    return topic_weights

    

def get_topic_words(lda_model):
    """
    Given a topic model, display all the words
    Input: An lda topic model
    returns: a list of words in the topic
    """
    all_topics = lda_model.show_topics(num_topics=12, num_words=50,
                                       formatted=False)
    topics_words = [(topic[0], [word[0] for word in topic[1]]) 
                        for topic in all_topics]

    return topics_words


market_index = 6
filename = './ldamodels/market' + str(market_index) + '/model.model'
lda_model = gensim.models.ldamodel.LdaModel.load(filename)

filename_corpus = './ldamodels/market' + str(market_index) + '/corpus.corpus'
with open(filename_corpus, 'rb') as filehandle:
    corpus = pickle.load(filehandle)

topic_weights = get_topic_weights(lda_model, corpus)

arr = pd.DataFrame(topic_weights).fillna(0).values

# Only keep well-separated points
# arr = arr[np.amax(arr, axis=1) > 0.35]

# Dominant toic number in each document
topic_num = np.argmax(arr, axis=1)

# tSNE dimension reduction
tsne_model = TSNE(n_components=2, verbose=1, random_state=0, angle=.99,
                  init='pca')
tsne_lda = tsne_model.fit_transform(arr)
tsne_lda = pd.DataFrame(tsne_lda, columns=['x', 'y'])
tsne_lda['hue'] = topic_num
mycolors = np.array([color for name, color in mcolors.TABLEAU_COLORS.items()])


n_topics = 5
source = ColumnDataSource(data=dict(x=tsne_lda.x,
                                    y=tsne_lda.y,
                                    colors=[all_palettes['Set1'][8][i] for i in tsne_lda.hue],
                                    # colors=mycolors[:n_topics],
                                    alpha=[1]*tsne_lda.shape[0]))
                                    # size=[8]*2*tsne_lda.shape[0]))

# pca1 = tsne_lda[:,0]
# pca2 = tsne_lda[:, 1]

# # Plot the topic clusters using Bokeh
output_file('./data/bokeh_plots/market6_testing.html')

plot = figure(plot_width=900, plot_height=700)
plot.scatter('x', 'y', source=source, color='colors', size=8)
show(plot)
# %%
