# Using streamlit to make my web app

import streamlit as st
import pandas as pd
import numpy as np
import webbrowser
import os
dirpath = os.path.dirname(os.path.realpath(__file__))

data = pd.read_excel('./list_of_farmers_markets.xlsx')
size = data.shape[0]

location = st.selectbox('Please select a city:',
                        data['Location'])
display_type = st.radio('What kind of display would you like?',
                        ['Word cloud', 'Detailed dashboard'])

if st.button('Go!'):
    if display_type == 'Detailed dashboard':
        market_index = data[data['Location'] == str(location)].index.values[0]
        url =  ('file://' + dirpath + '/LDAvis_prepared/market' + 
                str(market_index) + '/LDAvis.html')
        webbrowser.open(url, new=0)  # open in new tab
    