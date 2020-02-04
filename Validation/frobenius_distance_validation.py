# %%
# In this file, we calculate the frobenius distance between 30 follower feature
# vectors and each of 30 random twitter users and 30 sureshot users. The hope
# is that the followers are closer to the sureshots. We will also try a cosine
# similarity distance as a metric

# %%
# Imports

import numpy as np
import pickle
import scipy.spatial as sp

# %%
# Loading all the list of lists and converting them to numpy arrays

with open('./data/sureshot_augmented_feature_vectors.data', 'rb') as filehandle:
    sureshot_vectors = np.array(pickle.load(filehandle))

with open('./data/augmented_feature_vectors_followers_all.data', 'rb') as filehandle:
    followers_vectors_all = np.array(pickle.load(filehandle))

with open('./data/augmented_feature_vectors_followers.data', 'rb') as filehandle:
    followers_vectors = np.array(pickle.load(filehandle))

with open('./data/augmented_feature_vectors_random.data', 'rb') as filehandle:
    random_vectors = np.array(pickle.load(filehandle))

# %%
# Calculate the Frobenious distance

follower_random_diff = np.sum((followers_vectors - random_vectors)**2)
follower_sureshot_diff = np.sum((followers_vectors - sureshot_vectors)**2)


# %%
# For the cosine difference, we calculate the cosine distance matrix which
# turns out to be n x n i.e. the cosine difference between every pair of 
# people in the follower-random and follower-sureshots
follower_random_cosine_diff = 1 - sp.distance.cdist(followers_vectors, 
                                                    random_vectors, 'cosine')
follower_sureshot_cosine_diff = 1 - sp.distance.cdist(followers_vectors, 
                                                    sureshot_vectors, 'cosine')

mean_follower_random_cosine_diff = np.mean(follower_random_cosine_diff)
# (turns out to be 0.45)
mean_follower_sureshot_cosine_diff = np.mean(follower_sureshot_cosine_diff)
# (turns out to be 0.68)

# %%
# %%
# Calculating the cosine similarity metric between the first half of followers
# and the second half of the followers

num_followers = (followers_vectors_all.shape[0])//2
first_half_followers = followers_vectors_all[:num_followers]
second_half_followers = followers_vectors_all[num_followers:2*num_followers]

first_second_cosine_sim = 1 - sp.distance.cdist(first_half_followers,
                                                second_half_followers,
                                                'cosine')
print(np.mean(first_second_cosine_sim))
# %%
