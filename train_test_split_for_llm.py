import pandas as pd 
from sklearn.model_selection import train_test_split

data = pd.read_csv("data/reddit_post_combined.csv")
data = data.drop(columns=["author"])

train_data, test_data = train_test_split(data, test_size=0.2, stratify=data["mbti"], random_state=42)

train_data.to_csv("llm_data/train.csv", index=False)
test_data.to_csv("llm_data/test.csv", index=False)