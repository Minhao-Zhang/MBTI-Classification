import numpy as np
import pandas as pd
import nltk
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
        
        # Append the transformed chunk to the new dataset
        new_dataset = pd.concat([new_dataset, tokenized_chunk], ignore_index=True)
        print(f'Processed {file}')
        
        
        
new_dataset.to_pickle('pickled/word2vec_dataset.pkl')