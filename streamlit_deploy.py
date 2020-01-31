# %%
# Using streamlit to make my web app

import streamlit as st
import pandas as pd
import numpy as np
import webbrowser
import os
import graphviz as graphviz
from PIL import Image


dirpath = os.path.dirname(os.path.realpath(__file__))

# %%
#

list_markets = pd.read_excel('./list_of_farmers_markets.xlsx')
list_markets = list_markets.drop([1, 4, 9]).reset_index(drop=True)

#%%

st.title('Magnetic Market')
st.markdown("""### Attracting customers to farmers markets with data-driven options for engaging events""")

image = Image.open('./farmers-markets.jpg')

st.image(image, use_column_width=True)

location = st.selectbox('Choose a city',
                        list_markets['Location'])

if location == 'Austin, TX':
    topic = st.selectbox('Choose a Topic',
                            ['Management education',
                             'Education schools',
                             'Organic food',
                             'Food ingredients',
                             'Government officials',
                             'Say anything songs',
                             'Chicken dishes'])
    
    if topic == 'Management education':
        u = graphviz.Digraph(
            node_attr={'color': 'lightblue2', 'style': 'filled', 'ratio': 'expand'})
        u.attr(size='12,20')
        u.edge('Management education', 'conference')
        u.edge('Management education', 'skill')
        u.edge('Management education', 'technology')
        u.edge('Management education', 'professional')
        u.edge('Management education', 'leadership')
        u.edge('Management education', 'development')
        u.edge('Management education', 'tech')
        st.graphviz_chart(u)
    elif topic == 'Education schools':
        u = graphviz.Digraph(
            node_attr={'color': 'lightblue2', 'style': 'filled', 'ratio': 'expand'})
        u.attr(size='12,20')
        u.edge('Education schools', 'educator')
        u.edge('Education schools', 'scholarship')
        u.edge('Education schools', 'language')
        u.edge('Education schools', 'grade')
        u.edge('Education schools', 'insurance')
        u.edge('Education schools', 'talent')
        u.edge('Education schools', 'rank')
        st.graphviz_chart(u)
    elif topic == 'Organic food':
        u = graphviz.Digraph(
            node_attr={'color': 'lightblue2', 'style': 'filled', 'ratio': 'expand'})
        u.attr(size='12,20')
        u.edge('Organic food', 'farm')
        u.edge('Organic food', 'sustainable')
        u.edge('Organic food', 'organic')
        u.edge('Organic food', 'plastic')
        u.edge('Organic food', 'garden')
        u.edge('Organic food', 'planet')
        u.edge('Organic food', 'recycle')
        st.graphviz_chart(u)



    b = st.button('Click here for a detailed dashboard!')
    if b:
        market_index = 6
        # url =  ('file://' + dirpath + '/LDAvis_prepared/market' + 
        #         str(market_index) + '/LDAvis.html')
        url = 'file:///home/ec2-user/testing/LDAvis.html'
        st.write(url)
        webbrowser.open(url, new=0)  # open in new tab