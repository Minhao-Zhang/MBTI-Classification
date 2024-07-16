import numpy as np
import pandas as pd
from gensim.models import Word2Vec
import pickle

# Function to load tokenized chunks from disk
def load_tokenized_chunk(file_path):
    with open(file_path, 'rb') as f:
        chunk = pickle.load(f)
    return chunk

# Function to transform text data into word vectors
def document_vector(doc, model):
    # Remove out-of-vocabulary words
    doc = [word for word in doc if word in model.wv.key_to_index]
    return np.mean(model.wv[doc], axis=0) if len(doc) > 0 else np.zeros(model.vector_size)

# Load the trained Word2Vec model
model = Word2Vec.load("models/word2vec.model")

# List of all tokenized chunk files
all_files = [f'pickled/tokenized_chunk_{i}.pkl' for i in range(20)]

# Create an empty DataFrame to hold the new dataset
new_dataset = pd.DataFrame()

# Process each tokenized chunk file
for file in all_files:
    with open(file, 'rb') as f:
        tokenized_chunk = pickle.load(f)
        
        # Transform each document in the chunk into a vector
        tokenized_chunk['vector'] = tokenized_chunk['tokens'].apply(lambda x: document_vector(x, model))
        
        # Drop the original tokens column
        tokenized_chunk = tokenized_chunk.drop(columns=['tokens', 'body'])
        float_list_df = pd.DataFrame(tokenized_chunk['vector'].tolist(), columns=[f'feature_{i}' for i in range(len(tokenized_chunk['vector'][0]))])
        expanded_df = pd.concat([tokenized_chunk[['author', 'mbti']], float_list_df], axis=1)
        
        # Append the transformed chunk to the new dataset
        new_dataset = pd.concat([new_dataset, expanded_df], ignore_index=True)
        print(f'Processed {file}')

# ERROR
# After running the script, there will be a weird behavior of 
# repeated rows on line 1M. The reason for this is still unknown.
# This might have happened in previous processing steps.
# To fix this, we can simply drop the duplicates.
new_dataset.drop(index=1000000, inplace=True)

# split the data into 4 parts and save them
new_dataset_1 = new_dataset.iloc[:2500000]
new_dataset_2 = new_dataset.iloc[2500000:5000000]
new_dataset_3 = new_dataset.iloc[5000000:7500000]
new_dataset_4 = new_dataset.iloc[7500000:]

new_dataset_1.to_csv('data/word2vec_data_1.csv', index=False)
new_dataset_2.to_csv('data/word2vec_data_2.csv', index=False)
new_dataset_3.to_csv('data/word2vec_data_3.csv', index=False)
new_dataset_4.to_csv('data/word2vec_data_4.csv', index=False)