#%%
# imports 
import urllib
from bs4 import BeautifulSoup
from pprint import pprint
import pandas as pd

# %%
# Notes

# The websites that were scraped:
# https://www.eventbrite.com/blog/70-event-ideas-and-formats-to-inspire-your-next-great-event-ds00/
# https://www.updater.com/blog/resident-event-ideas

# %%
# Defining functions
    
def extract_event_names_descriptions_eventbrite(results):
    """
    Takes in a list of events from the eventbrite page.
    Returns: Appropriate event names stripped of extraneous text and html tags,
    zipped with the event descriptions
    """
    events = results.find_all('h2')
    descriptions = results.find_all('p')
    event_names = list(map(lambda event: event.text.replace('\xa0', ' '),
                        events))[:-1]
    event_names = [event_name.split(':')[-1] for event_name in event_names]
    descriptions = list(map(lambda description: description.text.replace(
                            '\xa0', ' '), descriptions))[2:-1]
    return zip(event_names, descriptions)


def extract_event_names_descriptions_updater(results):
    """
    Takes in a list of events from the updater page.
    Returns: Appropriate event names stripped of extraneous text and html tags,
    zipped with the event descriptions.
    """
    events = results.find_all('h3')
    descriptions = results.find_all('p')
    event_names = list(map(lambda event: event.text.split('. ')[-1], events))
    descriptions = list(map(lambda description: 
                                description.text, descriptions))[4:-2]
    return zip(event_names, descriptions)
    

#%%
# Loading the html file and calling functions
with open('./data/eventbrite_ideas.html', 'r', encoding='utf8') as filehandle:
    page_eb = filehandle.read()

soup_eb = BeautifulSoup(page_eb, 'html.parser')
results_eventbrite = soup_eb.find('div',
    class_='post-content context-content-single context-content--eb-helpers')
eventnames_descriptions_eb = dict(extract_event_names_descriptions_eventbrite(
                                                        results_eventbrite))

with open('./data/updater_ideas.html', 'r', encoding='utf8') as filehandle:
    page_up = filehandle.read()

soup_up = BeautifulSoup(page_up, 'html.parser')
results_updater = soup_up.find('div', class_='col sqs-col-12 span-12')
eventnames_descriptions_up = dict(extract_event_names_descriptions_updater(
                                                            results_updater))

eventnames_descriptions_eb.update(
                            eventnames_descriptions_up)
eventnames_descriptions = eventnames_descriptions_eb

# %%
# export to file so that I can further edit the wording and the verbiage
df = pd.DataFrame(eventnames_descriptions.items(), 
                            columns=['Event Name', 'Event Description'])

df.to_csv('./data/all_events_names_descriptions.csv', index=False)
# %%
