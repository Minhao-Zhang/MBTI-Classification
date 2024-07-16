import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# data = pd.read_csv('data/reddit_post.csv')
# labels = data['mbti']

# # split data into train and test set using stratified sampling
# train_indices, test_indices = train_test_split(np.arange(data.shape[0]), stratify=labels, test_size=0.2, random_state=42)

# # sort the indices
# train_indices = np.sort(train_indices)
# test_indices = np.sort(test_indices)

# # print first 10 indices
# print(train_indices[:10])
# print(test_indices[:10])

# # save the indices for reproducibility
# np.save('data/train_indices.npy', train_indices)
# np.save('data/test_indices.npy', test_indices)

# load the indices
train_indices = np.load('data/train_indices.npy')
test_indices = np.load('data/test_indices.npy')

# split indicies into 4 parts where 
# part 1 is values that 0 to 99999
# part 2 is 100000 to 199999
# part 3 is 200000 to 299999
# part 4 is 300000 to end

train_indices_0 = train_indices[train_indices < 100000]
train_indices_1 = train_indices[(train_indices >= 100000) & (train_indices < 200000)]
train_indices_2 = train_indices[(train_indices >= 200000) & (train_indices < 300000)]
train_indices_3 = train_indices[train_indices >= 300000]

test_indices_0 = test_indices[test_indices < 100000]
test_indices_1 = test_indices[(test_indices >= 100000) & (test_indices < 200000)]
test_indices_2 = test_indices[(test_indices >= 200000) & (test_indices < 300000)]
test_indices_3 = test_indices[test_indices >= 300000]

data0 = np.load("pickled/word2vec_data_part_0.npy")