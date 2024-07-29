import pickle 
import pandas as pd
import numpy as np

PICKLE_PATH = "./tmp/"
DATA_PATH = "./data/"
SPLIT_PATH = "../data/train_test_split/"

all_files = [f'{PICKLE_PATH}word2vec_{i}.pkl' for i in range(4)]

new_dataset = pd.DataFrame()
for file in all_files:
    with open(file, 'rb') as f:
        tokenized_chunk = pickle.load(f)
        new_dataset = pd.concat([new_dataset, tokenized_chunk], ignore_index=True)
        print(f"Loaded {file}")
        del tokenized_chunk

new_dataset.drop(columns=['author'], inplace=True)

train_indices = np.load(f'{SPLIT_PATH}train_indices.npy')
test_indices = np.load(f'{SPLIT_PATH}test_indices.npy')

train_data = new_dataset.iloc[train_indices]
test_data = new_dataset.iloc[test_indices]

del new_dataset

with open(f'{DATA_PATH}test.pkl', 'wb') as f:
    pickle.dump(test_data, f)
    print("Saved test")
    print(test_data.iloc[:10])

del test_data

with open(f'{DATA_PATH}train.pkl', 'wb') as f:
    pickle.dump(train_data, f)
    print("Saved train")
    print(train_data.iloc[:10])