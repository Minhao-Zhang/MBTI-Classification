import numpy as np
import pandas as pd
import pickle 

with open('pickled/train_indices.pkl', 'rb') as f:
    train_indices = pickle.load(f)
with open('pickled/test_indices.pkl', 'rb') as f:
    test_indices = pickle.load(f)

data = pd.read_csv('data/reddit_post.csv')
# 19626893
# 19626894

# split data 
train_data = data.iloc[train_indices]
test_data = data.iloc[test_indices]

train_data.to_csv('data/train_data.csv', index=False)
test_data.to_csv('data/test_data.csv', index=False)

del data, train_data, test_data

with open('pickled/word2vec_dataset.pkl', 'rb') as f:
    word2vec_dataset = pickle.load(f)

train_word2vec_dataset = word2vec_dataset[train_indices]
test_word2vec_dataset = word2vec_dataset[test_indices]

with open('pickled/train_word2vec_dataset.pkl', 'wb') as f:
    pickle.dump(train_word2vec_dataset, f)
with open('pickled/test_word2vec_dataset.pkl', 'wb') as f:
    pickle.dump(test_word2vec_dataset, f)

del word2vec_dataset, train_word2vec_dataset, test_word2vec_dataset