import pandas as pd 
DATA_PATH = "full_pull/full_pull_v20000000000"
OUTPUT_PATH = "cleaned_data/fp_"

def clean_data(index: str):
    
    print(f"Start processing full_pull_v20000000000{index}.csv")
    
    data = pd.read_csv(DATA_PATH + index + ".csv")
    
    # write a regex that will only extract mbti types from the text then apply to the flair_text column and save to a new column called mbti
    data['mbti'] = data['flair_text'].str.extract(r'([I|E][N|S][F|T][J|P])')

    # remove rows where mbti is missing 
    data = data.dropna(subset=['mbti'])

    # make everything lowercase 
    data['body'] = data['body'].str.lower()

    # remove any url from the body column 
    data['body'] = data['body'].replace(r'http\S+', '', regex=True)

    # remove any post with non-english characters
    data = data[data['body'].str.contains(r'[^\x00-\x7F]+') == False]

    # remove any reddit like link from the body column like r/ or u/
    data['body'] = data['body'].replace(r'[r|u]\/\S+', '', regex=True)

    # remove any punctuation from the body column
    data['body'] = data['body'].replace(r'[^\w\s]', '', regex=True)
    
    # replace all line breaks with space
    data['body'] = data['body'].replace(r'\n', ' ', regex=True)
    
    # remove any rows with 20 < body length < 3000
    data = data[data['body'].str.len() > 20]
    data = data[data['body'].str.len() < 3000]

    # keep only author, body, mbti column 
    data = data[['author', 'body', 'mbti']]
    
    # save to file 
    data.to_csv(OUTPUT_PATH + index + ".csv", index=False)
    
    print(f"Done processing full_pull_v20000000000{index}.csv")

# use multiprocessing to clean all the data
from multiprocessing import Pool
import os
import time

if __name__ == '__main__':
    
    indexes = [str(i).zfill(2) for i in range(18)]
    start_time = time.time()
    p = Pool(os.cpu_count()//2)
    p.map(clean_data, indexes)
    print("--- %s seconds ---" % (time.time() - start_time))