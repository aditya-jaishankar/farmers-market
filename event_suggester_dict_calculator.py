# %%
# Imports

import pickle
import pandas as pd

# %%
# Define functions

def clean_strings(row):
    """
    Accepts a string of comma seprated topic names and converts that into a
    list. Also removes the extraneous space
    """

    topic_names = row['Topic name']
    topic_names = topic_names.replace(', ', ',')  # Remove spaces
    topic_names = topic_names.split(',')
    return topic_names


def get_list_event_dicts(topic):
    """
    Input: accepts a topic label
    Returns: a list of dictionaries. Each dictionary has key event name and 
    value event description. 
    """


# %%

events_descriptions = pd.read_csv('./data/eventname_description_table_final.csv',
                                    skiprows=0)

events_descriptions['Topic name'] = events_descriptions.apply(lambda row: 
                                                    clean_strings(row), axis=1)
# %%
# We resort to manually constructing the dictionary from which to choose events
# Given more time, I would use a doc2vec, or better still, BERT embeddings to
# find more suitable events.

with open('./data/topic_labels_dict.data', 'rb') as filehandle:
    topics_dict = pickle.load(filehandle)

unique_topic = set()
for market_index in topics_dict.keys():
    labels = topics_dict[market_index].values()
    unique_topic = unique_topic.union(labels)

# %%
# Building the dictionary

# The idea is to first make a dictionary by grouping by topic label. The key
# for the topic label is to have a list of dictionaries with event name as key
# and event description as label.

event_suggester_dict = {}

num_rows = events_descriptions.shape[0]
for topic in unique_topic:
    event_suggester_dict[topic] = []
    for row in range(num_rows):
        topics_list = events_descriptions['Topic name'].iloc[row]
        if topic in topics_list:
            temp_dict = {}
            event_name = events_descriptions['Event name'].iloc[row]
            temp_dict[event_name] = events_descriptions['Event description'].iloc[row]
            event_suggester_dict[topic] += [temp_dict]

with open('./data/event_suggester_dict.data', 'wb') as filehandle:
    pickle.dump(event_suggester_dict, filehandle)
# %%
