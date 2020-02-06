# %%
# Using streamlit to make my web app

import streamlit as st
import pandas as pd
# import numpy as np
# import warnings
# import gensim
# from tqdm import tqdm
from pprint import pprint
import pickle
import os
import random

from bokeh.plotting import show, figure
from bokeh.models import ColumnDataSource, LinearColorMapper, ColorBar, BasicTicker, HoverTool
from bokeh.palettes import Viridis256
from bokeh.transform import transform
from bokeh.io import output_file

import matplotlib.colors as mcolors
from pprint import pprint
from PIL import Image

import numpy as np

# %%
# Define functions


def generate_event_suggestion(topic_label):
    """
    Accepts a topic label and generates an event name and a description for the
    event
    Accepts: String of topic_label
    Returns: tuple of (event name, event description)
    """
    random_dict = random.choice(event_suggester_dict[topic_label])
    return(list(random_dict.items()))


def insert_newline_char(words_list):
    words_list_newline = []
    for word in words_list:
        words_list_newline.append(word)
        words_list_newline.append('\n')
    return words_list_newline


# %%
# General required file loads

list_markets = pd.read_excel('./list_of_farmers_markets.xlsx')
list_markets = list_markets.drop([1, 4, 9]).reset_index(drop=True)

with open('./data/topic_labels_dict.data', 'rb') as filehandle:
    topic_labels_dict = pickle.load(filehandle)

with open('./data/event_suggester_dict.data', 'rb') as filehandle:
    event_suggester_dict = pickle.load(filehandle)

# %%
# Sidebar content

st.sidebar.title('Magnetic Markets')
image = Image.open('./farmers-markets.jpg')
st.sidebar.image(image, use_column_width=True)
product_summary = """
                  The number of visitors to and purchases made at farmers
                  markets has been decreasing for the past several years.
                  Organizing events at markets is a demonstrated way to increase
                   market attendance and customer retention. Magnetic Markets 
                  uses Twitter data to analyze topics discussed by people 
                  interested in farmers markets. Market managers can use this 
                  information to organize events at markets. This approach is
                  data-driven, location-specific, and real-time. 
                  """

interaction_description = """First, select a city. Inspect the resulting topics of 
                         discussion in the bubble plot. The size and color
                         of the bubble for each topic represents the number of 
                         followers for whom this was the most dominant topic of
                         discussion. Hover on the bubbles to see the keywords
                         that contitute a topic. Finally, choose one of the topics
                         that the city is talking about to generate an event
                         suggestion related to that topic!
                         """

# st.sidebar.markdown(product_summary)

st.sidebar.markdown('## Interacting with this product')
st.sidebar.markdown(interaction_description)


# Headers, etc.
st.title('Magnetic Market')
tagline = 'Attracting customers to farmers markets with data-driven event ideas.'
st.subheader(tagline)
st.write(product_summary)



# %%
# Choose a location and load the appropriate files and dictionaries
st.header('Choose a city:')
location = st.selectbox('', list_markets['Location'])
market_index = list_markets[list_markets['Location'] == location].index.to_list()[0]
bokeh_dict_filename = './data/bokeh_dict_' + str(market_index) + '.data'

# Load dictionary for Bokeh plot
with open(bokeh_dict_filename, 'rb') as filehandle:
    bokeh_dict = pickle.load(filehandle)


# %%
# Generate the Bokeh plot

st.header('Topic Dominance plot')
st.write('The bubble plot below displays the different topics discussed in {}'.format(location))
st.markdown("""The **dominance** of a topic in a document is characterized by 
        the proportion of all words in the document that belong to that 
        particular topic. For example, if half of all words in a document are 
        in the bag of words corresponding to topic 1, the dominance of topic 1
        is 0.5. The most dominant topic for a document is that topic which has
        the highest dominance in the document. The size of the bubbles for each
        topic below is proportional to the number of documents that had that
        topic as being most dominant. Hover on the bubble to see a sample of 
        keywords that constitute the topic.""")


df = pd.DataFrame(data=bokeh_dict)
df['words'] = df['words'].apply(lambda l: ['\n' + s for s in l])

# df['words'] = df['words'].apply(lambda l: np.array(l).reshape(15, 1))
source = ColumnDataSource(df)
plot = figure(x_range=df['topic'].unique(),
        y_range=df['subject'].unique(),
        plot_width=800,
        plot_height=300)

color_mapper = LinearColorMapper(palette=Viridis256,
                                low=df['counts'].min(),
                                high=df['counts'].max())
color_bar = ColorBar(color_mapper=color_mapper,
                    location=(0, 0),
                    ticker=BasicTicker())
plot.add_layout(color_bar, 'right')
plot.xaxis.major_label_text_font_size = "13pt"

plot.scatter(x='topic', y='subject',
        size='counts_scaled',
        fill_color=transform('counts', color_mapper),
        source=source)

TOOLTIPS = """
<div style="width:100px;">
<font size="+2">
<strong>
Words:
</strong>
</font>
<br>
<font size="+0.5">
@words
</font
</div>
"""
# plot.add_tools(HoverTool(tooltips=[('Words', '@words')]))
plot.add_tools(HoverTool(tooltips=TOOLTIPS))

st.bokeh_chart(plot)

# %%
# Generate a dropdown list of the different topics for the given market

location_topics = topic_labels_dict[market_index]
st.header('Choose a specific topic that {} talks about!'.format(location))
specific_topic_label = st.selectbox('', list(set(location_topics.values())))

generated = generate_event_suggestion(specific_topic_label)
event_name = generated[0][0]
description = generated[0][1]

st.header("Here's your event suggestion!")
st.subheader(event_name)
st.write(description)
