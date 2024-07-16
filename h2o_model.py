import numpy as np
import pandas as pd
import pickle 
from sklearn.model_selection import train_test_split
import h2o
from h2o.automl import H2OAutoML

# Load the dataset
with open('pickled/train_word2vec_dataset.pkl', 'rb') as f:
    data = pickle.load(f)

# Split the dataset into training and validation sets based on the label    
train, valid = train_test_split(data, test_size=0.3, stratify=data['mbti'])

# Initialize the H2O cluster
h2o.init(max_mem_size="32G")  # Allocating 30GB to the JVM

response_column = "mbti"

# Save validation set as H2OFrame
h2o_valid = h2o.H2OFrame(valid)
h2o_valid[response_column] = h2o_valid[response_column].asfactor()
h2o_valid_path = "h2o/h2o_valid.hex"
h2o.save_frame(h2o_valid, h2o_valid_path)
print("Validation set saved")
del h2o_valid  # Free up memory

# Split training data into smaller chunks
chunk_size = 1000000  # Adjust chunk size based on available memory
num_chunks = int(np.ceil(len(train) / chunk_size))

# Save each chunk as a separate H2OFrame in binary format
for i in range(num_chunks):
    chunk = train.iloc[i * chunk_size:(i + 1) * chunk_size]
    h2o_chunk = h2o.H2OFrame(chunk)
    h2o_chunk[response_column] = h2o_chunk[response_column].asfactor()
    h2o.save_frame(h2o_chunk, f"h2o/h2o_train_chunk_{i}.hex")
    print(f"Chunk {i + 1}/{num_chunks} saved")
    del h2o_chunk  # Free up memory

predictors = ['vector']

# Train a classification model using H2OAutoML
aml = H2OAutoML(max_runtime_secs=7200, seed=42, balance_classes=True)

# Load each chunk one by one during training
for i in range(num_chunks):
    h2o_chunk = h2o.upload_frame(f"h2o/h2o_train_chunk_{i}.hex")
    if i == 0:
        aml.train(x=predictors, y=response_column, training_frame=h2o_chunk, validation_frame=h2o_valid)
    else:
        aml.train(x=predictors, y=response_column, training_frame=h2o_chunk)
    del h2o_chunk  # Free up memory

# Load validation set for evaluation
h2o_valid = h2o.upload_frame(h2o_valid_path)

# View the AutoML Leaderboard
lb = aml.leaderboard
print(lb)

# Evaluate the model performance
performance = aml.leader.model_performance(h2o_valid)
print(performance)

# Save the best model
model_path = h2o.save_model(model=aml.leader, path="./h2o", force=True)
print(f"Model saved to: {model_path}")

# Shutdown H2O cluster
h2o.shutdown(prompt=False)
