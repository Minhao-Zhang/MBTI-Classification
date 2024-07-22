import pandas as pd 

DATA_PATH = "../data/"

data = pd.read_csv(f'{DATA_PATH}reddit_post.csv')

# Function to concatenate body strings of each author
def concatenate_bodies(group, min_length=500):
    concatenated = ''
    for body in group['body']:
        if len(body) >= min_length:
            yield body
        else:
            if len(concatenated) > 0:
                concatenated += ' '
            concatenated += body
            if len(concatenated) >= min_length:
                yield concatenated
                concatenated = ''
    if len(concatenated) > 0:
        yield concatenated

# List to collect the new rows
new_rows = []

n = 700

# Group by 'author' and process each group
for author, group in data.groupby('author'):
    for concatenated_body in concatenate_bodies(group, n):
        new_rows.append({
            'author': author,
            'body': concatenated_body,
            'mbti': group.iloc[0]['mbti']  # assuming 'mbti' is the same for all rows of the same author
        })
        

# Create a new DataFrame from the new rows
new_df = pd.DataFrame(new_rows)

# remove the rows with less than n char, print them 
new_df = new_df[new_df['body'].str.len() >= n]
# remove any value that are more than 90th percentile
new_df = new_df[new_df['body'].str.len() <= new_df['body'].str.len().quantile(0.9)]

# shuffle the data  
new_df = new_df.sample(frac=1).reset_index(drop=True)

# save it to a new CSV file
new_df.to_csv(f'{DATA_PATH}reddit_post_combined.csv', index=False)