import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
import pickle

DATASET = '../data/train_test_split/reddit_post_cleaned.csv'
PICKLE_PATH = './tmp/'

nltk.download('punkt')

# Function to process and save chunks
def process_and_save_chunk(chunk, chunk_index):
    chunk['tokens'] = chunk['body'].apply(word_tokenize)
    chunk = chunk.drop(columns=['body'])
    with open(f'{PICKLE_PATH}tokenized_chunk_{chunk_index}.pkl', 'wb') as f:
        pickle.dump(chunk, f)
    del chunk  # Free memory

# Calculate the number of rows in the full dataset
total_rows = sum(1 for _ in open(DATASET)) - 1  # Minus 1 for the header row 
print(f'Total rows in dataset: {total_rows}')
chunk_size = 800000  # Adjust this value based on your system's memory
num_chunks = total_rows // chunk_size + (1 if total_rows % chunk_size != 0 else 0)

# Process dataframe in chunks
for i in range(num_chunks):
    start_index = i * chunk_size
    end_index = min((i + 1) * chunk_size, total_rows)
    chunk = pd.read_csv(DATASET, skiprows=range(1, start_index), nrows=chunk_size)
    process_and_save_chunk(chunk, i)
    print(f'Processed and saved chunk {i + 1} of {num_chunks}')
