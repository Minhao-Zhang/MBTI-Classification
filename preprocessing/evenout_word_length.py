import pandas as pd 

DATA_PATH = "../data/temp/"
FINAL_DATA_PATH = "../data/train_test_split/"

data = pd.read_csv(DATA_PATH + "reddit_post_combined.csv")

# add a column for the length of the post 
data['length'] = data['body'].apply(lambda x: len(x.split()))
# sort by length 
data = data.sort_values(by='length', ascending=False)
# remove rows with length < 110 or > 210 
data = data[(data['length'] > 110) & (data['length'] < 210)]
# remove the length column
data = data.drop(columns=['length'])
# split the mbti column into 4 columns
data['E-I'] = data['mbti'].apply(lambda x: x[0])
data['N-S'] = data['mbti'].apply(lambda x: x[1])
data['F-T'] = data['mbti'].apply(lambda x: x[2])
data['J-P'] = data['mbti'].apply(lambda x: x[3])
# save the data
data.to_csv(FINAL_DATA_PATH + "reddit_post_cleaned.csv", index=False)