import pandas as pd 

DATA_PATH = "../data/temp/"

data = pd.read_csv(f'{DATA_PATH}reddit_post.csv')

def split_2000(data: pd.DataFrame):
    # find data that are longer than 2000 characters 
    data_long = data[data['body'].str.len() > 2000]
    # remove data that are longer than 2000 characters
    data = data[data['body'].str.len() <= 2000]
    # split the data that are longer than 2000 characters into multiple rows
    new_rows = []
    # find the index of the first space after 1000 characters
    for index, row in data_long.iterrows():        
        body = row['body']
        first_space = body.find(' ', 800)
        new_rows.append({
            'author': row['author'],
            'body': body[:first_space].strip(),
            'mbti': row['mbti']
        })
        second_space = body.find(' ', first_space+800)
        new_rows.append({
            'author': row['author'],
            'body': body[first_space:second_space].strip(),
            'mbti': row['mbti']
        })
        new_rows.append({
            'author': row['author'],
            'body': body[second_space:].strip(),
            'mbti': row['mbti']
        })
    temp_df = pd.DataFrame(new_rows)
    data = pd.concat([data, temp_df])
    return data
data = split_2000(data)

def split_1400(data: pd.DataFrame):
    # find data that are longer than 1600 characters 
    data_long = data[data['body'].str.len() > 1400]
    # remove data that are longer than 1600 characters
    data = data[data['body'].str.len() <= 1400]
    # split the data that are longer than 1600 characters into multiple rows
    new_rows = []
    # find the index of the first space after 800 characters
    for index, row in data_long.iterrows():        
        body = row['body']
        sep = len(body) // 2
        first_space = body.find(' ', sep)
        new_rows.append({
            'author': row['author'],
            'body': body[:first_space].strip(),
            'mbti': row['mbti']
        })
        new_rows.append({
            'author': row['author'],
            'body': body[first_space:].strip(),
            'mbti': row['mbti']
        })
    temp_df = pd.DataFrame(new_rows)
    data = pd.concat([data, temp_df])
    return data
data = split_1400(data)

def split_1000(data: pd.DataFrame):
    # find data that are longer than 1000 characters 
    data_long = data[data['body'].str.len() > 1000]
    # remove data that are longer than 1000 characters
    data = data[data['body'].str.len() <= 1000]
    # split the data that are longer than 1000 characters into multiple rows
    new_rows = []
    # find the index of the first space after 500 characters
    for index, row in data_long.iterrows():        
        body = row['body']
        # find the last space after 1000 characters
        first_space = body.find(' ', 900)
        new_rows.append({
            'author': row['author'],
            'body': body[:first_space].strip(),
            'mbti': row['mbti']
        })
        new_rows.append({
            'author': row['author'],
            'body': body[first_space:].strip(),
            'mbti': row['mbti']
        })
    temp_df = pd.DataFrame(new_rows)
    data = pd.concat([data, temp_df])
    return data
data = split_1000(data)

# Function to concatenate body strings of each author
def concatenate_bodies(group, min_length):
    concatenated = ''
    for body in group['body']:
        if len(body) >= min_length:
            yield body.strip()
        else:
            if len(concatenated) > 0:
                concatenated += ' '
            concatenated += body.strip()
            if len(concatenated) >= min_length:
                yield concatenated
                concatenated = ''
    if len(concatenated) > 0:
        yield concatenated

# List to collect the new rows
new_rows = []

# Group by 'author' and process each group
for author, group in data.groupby('author'):
    for concatenated_body in concatenate_bodies(group, 700):
        new_rows.append({
            'author': author,
            'body': concatenated_body,
            'mbti': group.iloc[0]['mbti']  # assuming 'mbti' is the same for all rows of the same author
        })

# Create a new DataFrame from the new rows
new_df = pd.DataFrame(new_rows)

def trim_body(data: pd.DataFrame, n: int):
    data_long = data[data['body'].str.len() > n]
    data = data[data['body'].str.len() < n]
    new_rows = []
    for index, row in data_long.iterrows():
        body = row['body']
        last_space = body.rfind(' ', 0, n)
        new_rows.append({
            'author': row['author'],
            'body': body[:last_space].strip(),
            'mbti': row['mbti']
        })
    temp_df = pd.DataFrame(new_rows)
    data = pd.concat([data, temp_df])
    return data
new_df = trim_body(new_df, 1000)

new_df = new_df[new_df['body'].str.len() >= 700]
new_df = new_df[new_df['body'].str.len() <= 1000]

print(new_df.describe())

# shuffle the data  
new_df = new_df.sample(frac=1).reset_index(drop=True)

# save it to a new CSV file
new_df.to_csv(f'{DATA_PATH}reddit_post_combined.csv', index=False)