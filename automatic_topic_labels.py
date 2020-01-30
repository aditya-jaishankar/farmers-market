# %%
# imports

import spacy
from gensim.parsing.preprocessing import strip_multiple_whitespaces
import pandas as pd
from tqdm import tqdm as tqdm
import numpy as np
from sklearn.preprocessing import normalize
import warnings

nlp = spacy.load('en_core_web_lg')

# %%
# Defining functions
def get_similarity(words_1, words_2):
    """

    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
    doc1 = nlp(words_1)
    doc2 = nlp(words_2)

    return doc1.similarity(doc2)

def get_minimum_distance_word(df, topic_words):
    """
    Returns the wikipedia category that has the largest cosine similarity
    with respect to the input text.
    Input: a dataframe df and the words constituting a topic
    Returns: 
    """
    terms_vector = normalize(nlp(topic_words).vector.reshape(300, 1))
    
    category_vectors = np.array([df['doc2vec']]).reshape(-1, 300)
    numerator = np.dot(terms_vector, category_vectors)
    denominator = np.linalg.norm(terms_vector) * np.linalg.norm(category_vectors)
    cosine_distance = numerator/denominator

    return df['categories'].loc[np.argmax(cosine_distance)]
    
# %%
# Loading the data


categories = pd.read_pickle('./data/categories_final_cleaned_filtered.df')
categories.head()

# %%
topic_words = nlp("""restaurant recipe meal wine garden beer artist cocktail
                    taste sweet style breakfast winner street spring mention
                    delicious chef unique cook review ride bird downtown
                    episode purchase truck bike campus buy""")

# %%
