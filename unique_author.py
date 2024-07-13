import pandas as pd 

# read in all the data 
indexes = [str(i).zfill(2) for i in range(18)]
data = [pd.read_csv("cleaned_data/fp_" + index + ".csv") for index in indexes]

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

# save to file
unique_data.to_csv("data/unique_author_mbti.csv", index=False)
data.to_csv("data/unique_author_data.csv", index=False)