import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

DATA_PATH = "../data/train_test_split/"

data = pd.read_csv(DATA_PATH + "reddit_post_cleaned.csv")

# generate split indicies
train_indices, test_indices = train_test_split(data.index, test_size=0.2, random_state=42, stratify=data['mbti'])

train = data.loc[train_indices]
test = data.loc[test_indices]

train.to_csv(DATA_PATH + "train.csv", index=False)
test.to_csv(DATA_PATH + "test.csv", index=False)

# save the split indices
np.save(DATA_PATH + "train_indices.npy", train_indices)
np.save(DATA_PATH + "test_indices.npy", test_indices)