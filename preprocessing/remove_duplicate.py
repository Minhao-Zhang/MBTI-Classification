import pandas as pd 

DATA_PATH = "../data/temp/"

# read in all the data 
indexes = [str(i).zfill(2) for i in range(18)]
data = [pd.read_csv(f"{DATA_PATH}cleaned_" + index + ".csv") for index in indexes]

# append all the data into one dataframe 
data = pd.concat(data)

# only consider unique author and mbti pair
unique_data = data[["author", "mbti"]].drop_duplicates()

# groupy by author, check mbti's value is unique
grouped = unique_data.groupby("author")["mbti"].nunique()

# remove authors with multiple mbti values
authors = grouped[grouped == 1]

# keep only the authors with one mbti value
unique_data = unique_data[unique_data["author"].isin(authors.index)]

# keep rows with unique author and mbti pairs
data = data[data["author"].isin(unique_data["author"])]

# After these steps, you will get about 19.6M rows of data.
# Unfortunately, many of them are still duplicates.
# We will remove duplicates by checking all columns
data = data.drop_duplicates()
# remove any post by Daenyx due to inappropriate content
data = data[data['author'] != 'Daenyx']
# remove Daenyx from unique_author 
unique_data = unique_data[unique_data['author'] != 'Daenyx']

# save to file
data.to_csv(f"{DATA_PATH}reddit_post.csv", index=False)
unique_data.to_csv(f"{DATA_PATH}unique_author.csv", index=False)