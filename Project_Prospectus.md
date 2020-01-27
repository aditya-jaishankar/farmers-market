
# Motivation

Farmers markets serve a critical function in society. They are good for the environment because the carbon footprint of the food sold is typically much smaller, meats and poulty are often raised in more humane ways, they offer healthier options for fresh fruits and groceries. They offer an opportunity for communities to gather and socialize. They also serve an important economic function and help support local communities and businesses. Perhaps most importantly, they can be particularly helpful to low income communities by offering them economic incentives to make healthier food choices, for example by [doubling the value of food stamps](https://www.wholesomewave.org/how-we-work/doubling-snap) for SNAP shoppers. 

Unfortunately, farmers markets are beginning to show a decrease in customer growth.This is primarily because the market is [beginning to saturate](https://www.npr.org/sections/thesalt/2019/03/17/700715793/why-are-so-many-farmers-markets-failing-because-the-market-is-saturated). People who previously wished to go to farmers markets but couldn't because of not living near one now have much easier to access to markets. In essence, customers who care deeply about the notion of a farmers market can now usually find ways to make it to one on a regular basis. [Combining farmers markets with events](https://www.usda.gov/media/blog/2016/08/08/building-businesses-helping-communities-celebrating-fruits-farmers-markets) has shown promise in increasing customer attendance and retention. However, most farmers markets make these decisions rather intuitively; things like music concerts, dances, and fairs are common events, but these do not align with the local desires of the community. In today's ever changing landscape, shoppers care about financial fitness, healthy living, dual careers, stress management, climate change and social action. These topics might vary from city-to-city, and might also be seasonal. Instead of historic data or *a-priori* guesses, organizing events at farmers markets based on data-driven insights would serve not only a helpful social function by assisting communities with their needs, but also increase business and encourage customer growth. 

In this project I scrape twitter data find the tweets 

# Implementation

## File structures

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


**TODO:** Perhaps I can hand select a set of commercial followers? Check out the `farmers_market_twitter_all.csv` file. I have a list of farmers markets with their handles. I could use this is select all them, download all their tweets, so a topic model, train a binary classifier (simple), test_train_split (will need to include some non-market account, say random accounts from the corresponding city). Then expand the selected_follower function to include this case. 

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

Finally, I train the LDA with a choice of hyperparameters (used the coherence score to keep track of how many topics I need), and then export the trained model to the appropriate folder. Note that if I want to use the `pyLDAvis` library later to generate visuals, I need to also export the corpus. 

### `LDAvis.ipynb`

This file is solely for importing the trained LDA models, generating the `pyLDAvis` visuals and then exporting these visuals as HTML files. I then use `streamlit` later to display these visuals in the browser. This is just a few lines of code in `streamlit.py`. 



