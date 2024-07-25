import pandas as pd 

DATA_PATH = "../data/"

data = pd.read_csv(DATA_PATH + "reddit_post_combined.csv")

# add a column for the length of the post 
data['length'] = data['body'].apply(lambda x: len(x.split()))
# sort by length 
data = data.sort_values(by='length', ascending=False)
# remove rows with length < 110 or > 210 
data = data[(data['length'] > 110) & (data['length'] < 210)]
# remove the length column
data = data.drop(columns=['length'])
# save the data
data.to_csv(DATA_PATH + "reddit_post_cleaned.csv", index=False)