# MBTI Classifier 

This project is aimed to use large language models and online chat message to predict someone's personality. 
We will be using the MBTI indicator as our target. 

## Data Curation 

Please see the detailed desciption [here](./data_summary.md)

## Get Started 

If you want to process the data yourself, you can run 

```bash
./setup.sh
```

This will download the raw data (which may take a while) and unzip them into csv files. 

You can start the data cleaning by running 
    
```bash
python3 clean_data.py
python3 unique_users.py
```
