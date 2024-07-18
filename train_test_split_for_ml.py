import pickle 
import pandas as pd
from sklearn.model_selection import train_test_split

all_files = [f'pickled/tokenized_chunk_{i}.pkl' for i in range(6)]

new_dataset = pd.DataFrame()
for file in all_files:
    with open(file, 'rb') as f:
        tokenized_chunk = pickle.load(f)
        new_dataset = pd.concat([new_dataset, tokenized_chunk], ignore_index=True)
        print(f"Loaded {file}")
        del tokenized_chunk

new_dataset.drop(columns=['author'])
print("Dropped author")
train_data, test_data = train_test_split(new_dataset, test_size=0.2, stratify=new_dataset['mbti'], random_state=42)
print("Splitted data")
del new_dataset

with open(f'ml_data/test.pkl', 'wb') as f:
    pickle.dump(test_data, f)
    print("Saved test")
    print(test_data.iloc[:10])

del test_data

num = train_data.shape[0]
num = num // 5

for i in range(4):
    with open(f'ml_data/train_{i}.pkl', 'wb') as f:
        pickle.dump(train_data.iloc[i*num:(i+1)*num], f)
with open(f'ml_data/train_4.pkl', 'wb') as f:
    pickle.dump(train_data.iloc[4*num:], f)