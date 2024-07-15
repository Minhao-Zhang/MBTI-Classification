from gensim.models import Word2Vec
import pickle

# Function to load tokenized chunks from disk
def load_tokenized_chunk(file_path):
    with open(file_path, 'rb') as f:
        chunk = pickle.load(f)
    return chunk['tokens'].tolist()

# List of all tokenized chunk files
all_files = [f'pickled/tokenized_chunk_{i}.pkl' for i in range(20)]

# Initialize Word2Vec model with the first chunk
first_chunk_tokens = load_tokenized_chunk(all_files[0])
model = Word2Vec(vector_size=100, window=5, min_count=1, sg=0)
model.build_vocab(first_chunk_tokens)

# Train Word2Vec model incrementally with the remaining chunks
for file in all_files[1:]:
    tokens = load_tokenized_chunk(file)
    model.build_vocab(tokens, update=True)  # Update vocabulary with new tokens
    model.train(tokens, total_examples=len(tokens), epochs=model.epochs)  # Train model on the new tokens
    print(f'Trained on {file}')

# Save the trained Word2Vec model
model.save("models/word2vec.model")