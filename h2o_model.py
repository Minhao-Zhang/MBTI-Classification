import h2o
from h2o.automl import H2OAutoML
import pandas as pd
import pickle

# Initialize the H2O cluster
h2o.init(max_mem_size="30G")

# Function to load a part of the dataset
def load_part(file_path):
    with open(file_path, 'rb') as f:
        part = pickle.load(f)
    return part

# Load the first part of the data
part_file = 'pickled/small_train_word2vec.pkl'
print(f"Loading {part_file}")
part_df = load_part(part_file)
part_df.drop(columns=['author'], inplace=True)  # Drop the author column

# Convert the DataFrame to an H2OFrame
h2o_frame = h2o.H2OFrame(part_df)
del part_df  # Free up memory

# Set the response column and predictor columns
response_column = 'mbti'
predictors = [f'feature_{i}' for i in range(100)]

# Ensure that the response column is set as a factor for classification
h2o_frame[response_column] = h2o_frame[response_column].asfactor()

# Train the model using H2OAutoML
aml = H2OAutoML(max_runtime_secs=7200, seed=42, balance_classes=True)
aml.train(x=predictors, y=response_column, training_frame=h2o_frame)

# View the AutoML Leaderboard
lb = aml.leaderboard
print(lb)

# Save the best model
best_model = aml.leader
model_path = h2o.save_model(model=best_model, path="./best_model_small", force=True)
print(f"Model saved to: {model_path}")

# Shutdown H2O cluster
h2o.shutdown(prompt=False)
