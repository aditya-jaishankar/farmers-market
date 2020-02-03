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


# %%
# General required file loads

list_markets = pd.read_excel('./list_of_farmers_markets.xlsx')
list_markets = list_markets.drop([1, 4, 9]).reset_index(drop=True)

with open('./data/topic_labels_dict.data', 'rb') as filehandle:
    topic_labels_dict = pickle.load(filehandle)

with open('./data/event_suggester_dict.data', 'rb') as filehandle:
    event_suggester_dict = pickle.load(filehandle)

# %%
# Headers, etc.
st.title('Magnetic Market')
st.markdown("""### Attracting customers to farmers markets with data-driven options for engaging events""")
image = Image.open('./farmers-markets.jpg')
st.image(image, use_column_width=True)


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
st.write("""In the plot below, this is an explanation of the plot and how to 
         read it.""")


df = pd.DataFrame(data=bokeh_dict)
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

plot.scatter(x='topic', y='subject',
        size='counts_scaled',
        fill_color=transform('counts', color_mapper),
        source=source)
plot.add_tools(HoverTool(tooltips=[('Words', '@words')]))
st.bokeh_chart(plot)

# %%
# Generate a dropdown list of the different topics for the given market

location_topics = topic_labels_dict[market_index]
st.header('Choose a specific topic that {} talks about!'.format(location))
specific_topic_label = st.selectbox('', list(location_topics.values()))

generated = generate_event_suggestion(specific_topic_label)
event_name = generated[0][0]
description = generated[0][1]

st.header("Here's your event suggestion!")
st.subheader(event_name)
st.write(description)
