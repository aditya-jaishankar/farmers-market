# %%
# imports

import spacy
from gensim.parsing.preprocessing import strip_punctuation, strip_numeric
from gensim.parsing.preprocessing import strip_multiple_whitespaces
import pandas as pd
from tqdm import tqdm as tqdm

nlp = spacy.load('en_core_web_lg')
# %%
# Function definitions


def cleaner(s):
    """
    Takes in string s, replaces the underscores with spaces, strips punctuation
    and returns cleaned, depnkt list of words in the title
    """
    x = s.replace('_', ' ')
    x = strip_punctuation(x)
    x = strip_numeric(x)
    x = x.split()
    return(x)


def proper_noun_checker(words):
    """
    Checks to see if there is a proper noun in the title, and if there is, we
    drop that title from the dataframe. We use spacy's parts of speech tagging
    to check this.

    input: list of words in page_title
    returns: Boolean
    """
    doc = nlp(' '.join(words))
    for token in doc:
        if token.pos_ == 'PROPN':  # If the title contains a proper noun,reject
            return False
    return True


def cleanup(row):
    """
    Takes in a row and performs two clean up operations

    input: accepts a string version of the list
    returns: list of strings
    """
    words_list = row['categories']
    # words_list = list(map(lambda e: e.replace("'", ''), entry))
    return strip_multiple_whitespaces(' '.join(words_list))


def get_docvec(category):
    """
    input: a category string
    output: doc2vec vector calculated using spacy
    """
    # Consider doing a weighted mean instead of blind word2vec. Currently, this
    # implementation of doc2vec is just doing a mean of the individual word2vec
    doc = nlp(category)
    return doc.vector


# %%
# Load the file
# The wikipedia category data is available here:
# https://www.upriss.org.uk/fca/cstiw11wiki.html


categories = pd.read_csv('./data/cat_concept.txt',
                         error_bad_lines=False,
                         warn_bad_lines=False)
categories.columns = ['categories']
tqdm.pandas()
categories['categories'] = categories.apply(
                                lambda row: cleaner(row['categories']), axis=1)
# %%
# Various processing steps


# Remove categories longer than 4 words
length_mask = categories.apply(lambda row: len(row['categories']) in [1,2], 
                               axis=1)
categories = categories[length_mask]

# %%
# Check and remove categories with proper nouns
print('Removing proper nouns')
proper_noun_mask = categories.progress_apply(lambda row: 
                                        proper_noun_checker(row['categories']),
                                        axis=1)
categories = categories[proper_noun_mask]

#Make list of words into single string
print('Converting string of list to list of strings')
categories['categories'] = categories.progress_apply(cleanup, axis=1)

# We do some further cleaning and filtering

empty_mask = categories['categories'] != ''
categories = categories[empty_mask]
categories = categories.drop_duplicates()

# %%
# Calculate the doc2vec for categories and add a column to dataframe


print('Calculating doc2vec for wiki categories')
categories['doc2vec'] = categories.progress_apply(lambda row: 
                                              get_docvec(row['categories']),
                                              axis=1)
categories = categories.reset_index(drop=True)
# Some words don't exist, and are given zero vectors, so I am going to remove it


# %%
# Write the new dataframe to file
empty_mask = np.sum(np.array([categories['doc2vec']]).reshape(-1, 300), axis=1) != 0
categories = categories[empty_mask]
# categories.to_pickle('./data/categories_final_cleaned_filtered.df')

# %%
