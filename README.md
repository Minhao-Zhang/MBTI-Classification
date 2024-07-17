# MBTI Classifier 

This project is aimed to use large language models and online chat message to predict someone's personality. 
We will be using the MBTI indicator as our target. 

## Data Curation 

Please see the detailed desciption [here](./data_summary.md)

## Data Cleaning

If you want to process the data yourself, you can run 

```bash
./setup.sh
```

This will download the raw data (which may take a while) and unzip them into csv files. 

You can start the data cleaning by running 
    
```bash
python3 clean_data.py
python3 remove_duplicate.py
```

## Pre-Processing 

As the nature of chat messages, the length of each message varies a lot. 
A very short message may not contain enough information to predict someone's personality. 
Thus, I decided to combine shorter messages into longer ones with a minimum length of 300 characters (including spaces). 

You can run the following command to combine the data. 

```bash
python3 combine_short_text.py
```

