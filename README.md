
# Motivation

Farmers markets serve a critical function in society. Apart from stimulating local economies, they also offer several social benefits such as accepting SNAP coupons, offering healthier food options, and allowing for environmentally conscious food purchasing habits. Furthermore, they serve as community gathering spots and can bring people together who share a common interest and purpose.  

Unfortunately, farmers markets are beginning to show a decrease in customer growth. [Combining farmers markets with events](https://www.usda.gov/media/blog/2016/08/08/building-businesses-helping-communities-celebrating-fruits-farmers-markets) has shown promise in increasing customer attendance and retention. However, most farmers markets make these decisions rather intuitively; things like music concerts, dances, and fairs are common events, but these do not align with the local desires, needs, and wishes of the community. In today's ever changing landscape, shoppers care about financial fitness, healthy living, dual careers, stress management, climate change and social action. These topics might vary from city-to-city, and might also be seasonal. Instead of historic data or *a-priori* guesses, organizing events at farmers markets based on data-driven insights would serve not only a helpful social function by assisting communities with their needs, but also increase business and encourage customer growth. 

In this project I scrape twitter data to find these desires of the local communities served by the farmers markets. I apply a host of natural language processing techniques such a Latent Dirichlet Allocation, `word2vec` embeddings, binary classification with support vector machines, and a radial basis function kernel. 

View the presentation [here](https://docs.google.com/presentation/d/1zIkJ2WxlK7GXpil40jjBi6bHxGFkWSCU9s6_7BIKPLg/edit?usp=sharing) for more details.

# Implementation

## File structures

### `utils.py`

This file contains a lot of the boiler plate code that

### `user_list_generator.ipynb`

I have a list of the biggest farmers markets in the US. I use the Twitter accounts of these farmers markets. For each farmers market, I find a list of 2000 twitter followers using `api.followers`. The reasoning here is that people who follow the accounts of farmers marketsI generate `followers_dict`, which has the form:

```
{
    market1: [{user1_json}, {user2_json}, ..., {user2000_json}],
    market2: [{user1_json}, {user2_json}, ..., {user2000_json}],
    .
    .
    market10: [{user1_json}, {user2_json}, ..., {user2000_json}],    
}
```
Because these followers are selected at random, there is a lot of followers who might be bots, etc. So I use a criterion to select followers: use only those who have more than 300 followers and that have more than 1000 tweets. This filters my data set to followers who have something meaningful to say. At the end of this process, I have the selected followers in `followers_dict_trimmed` which has the form:
```
{
    market1: [{user-1_json}, {user-2_json}, ..., {user-m1_json}],
    market2: [{user-1_json}, {user-2_json}, ..., {user-m2_json}],
    .
    .
    market10: [{user-1_json}, {user-2_json}, ..., {user-m10_json}],    
}
```

Now comes the major chunk of the downloading. For each market, and each selected follower in each market, I download 500 of their most recent tweets. All this data is dumped into `all_tweets_dict.data`. The structure of this dictionary is:
```
{
    market1: {
                screen_name_1: [{tweet1, ..., tweetn}],
                .
                .
                .
                screen_name_m: [{tweet1, ..., tweetn}]
            }
    .
    .
    .
    market10: {
                screen_name_1: [{tweet1, ..., tweetn}],
                .
                .
                .
                screen_name_2028: [{tweet1, ..., tweetn}]
             }
}
```


### `commercial-markets-downloader.py`

To implement the above idea of being able to filer out commercial farmers market followers, I am first found a list of all farmers markets in the US from census data, available in `farmers_market_twitter_all.csv`. I am reducing the scope of my project to urban farmers markets for now (because they have more resources to organize events), so I first use more US census data and the `shapely` package to only select those farmers markets that lie in urban areas. Moreover, I had to do some cleaning of the twitter handles. Once this is done, I download 500 tweets from each of the eligible farmers markets. Because I know for sure that these are all farmers markets, I label all of these 1. 

Next, I want to augment this dataset with documents that are labeled 0. For this I create a random number generator that outputs a 64 bit unsigned integer and use this as a user_id (this is the format twitter requires) and then try and see if this id corresponds to an account with over 500 tweets and 500 followers. I then download all tweets of these eligible accounts and then label them as 0. 

Finally I convert these tweet dictionaries into a more useful format so that I can create docs out of them and then feed them into code that performs LDA, determines the distribution of topics per document and then performs binary classification given the labels. The feature vectors are the topic probability distributions for each document. 

### `LDA.ipynb`

This code does all the major work of implementing the LDA analysis. Before actually doing the LDA, I first create `master_dict` which takes `all_tweets_dict` in all it's gory detail and simplifies it into a structure of the form: 

```
{
    market_1: {
                screen_name_1: 
                    {
                        hashtags: [list of hashtags from each tweet], 
                        fulltext: [list of all cleaned/depunkt words across all tweets]
                    },
                .
                .
                screen_name_m: 
                    {
                        hashtags: [list of hashtags from each tweet], 
                        fulltext: [list of all cleaned/depunkt words across all tweets]
                    }
              }
    .
    .
    .
    market_k: {
                screen_name_1: 
                    {
                        hashtags: [list of hashtags from each tweet], 
                        fulltext: [list of all cleaned/depunkt words across all tweets]
                    },
                .
                .
                screen_name_m: 
                    {
                        hashtags: [list of hashtags from each tweet], 
                        fulltext: [list of all cleaned/depunkt words across all tweets]
                    }
              }
}
```
To do this, I define several helper functions to get the user name, to get the list of hashtags, to do all the cleaning and pre-processing, and then returning the list of such words. This dictionary is called `master_dict.data`. 

I then use the function `get_docs(master_dict, market)` to return a list of the form `[[all-tweet-words-of-user-1], ..., [all-tweet-words-of-user-m1]]` given the market that I am interested in. 

Finally, I train the LDA with a choice of hyperparameters (used the coherence score to keep track of how many topics I need), and then export the trained model to the appropriate folder under `./ldamodels/`. 

### `commercial-markets-downloader.py`

This file downloads thousands of tweets of several hundred farmers markets in the US.  I find this list by using [this dataset](https://catalog.data.gov/dataset/farmers-markets-geographic-data) of all farmers markets in the US along with their twitter handles. To limit the scope of this project, I only was interested in farmers markets that lie in urban geographic domains. For this, I used shapely and defined my own `is_urban` function that returns a Boolean. Finally, for urban market, I obtain tweets, clean, preprocess and output `market_tweets_dict` with market as keys and a list of `tweet._json` objects as the values. This file also generates a set of tweets of random users, which is then used later in the validation. 

### `market_follower_data_prepare.py`

This file helps me build a dictionary of human followers + businesses. This is then fed into a binary classifier to only pass humans into when I am doing the actual LDA. From `master_dict`, this file picks 500 followers at random, constructs a dictionary `dataset_tweets_markets_followers` of the form:

```
{
    random_follower: {
                        'tweets': [list of the fulltext of all tweets]
                        'label': 0/1
                     }
}
```

The labels are assigned to be 0 (not commercial) and 1 (yes commercial). 

### `market_follower_binary_classifer.py`

This file loads `dataset_tweets_markets_followers` and then fits an lda model, makes inferences on each individual document and constructs probability feature vectors. I also tack on the label field from before, so now I have a set of feature vectors and labels that I pass through a SVM binary classifier with an RBF kernel. I also hold out some data to do the validation and I achieve a 90% model accuracy. I then save and export this classifier. I use this file to filter out who gets to accepted into my 'list of humans'

### `web_scraping_events.py`

This file used `BeautifulSoup` to web scrape a list of common community events so that I feed this into my event recommendation part of the code. 

### `automatic_topic_labels.py`

This file takes in the list of twitter categories, finds the `word2vec` vectors. It then compares these vectors with the mean `word2vec` vectors for the list of words that constitute each of the topics for each of the cities. 

### `event_suggester_dict_calculator.py`

This file outputs a dictionary of event title and event description for each for each market and then for each of the topics within that market.

### `follower-random-feature-vector-prediction`

This file calculates feature vectors matrices for the set of followers and the set of random users on twitter (who are likely to not talk anything about related to farmers markets)

### `generate_bokeh_dict.py`

This file outputs a dictionary with keys `topic`, `subject`, `counts`, `counts_scaled`, `words`. This is then pulled in by the streamlit code to generate the bokeh plots. 

### `streamlit_deploy`

This file runs the streamlit app. It draws all the appropriate dictionaries and files and then constructs the bokeh file and other outputs necessary for the rest of the display. 
