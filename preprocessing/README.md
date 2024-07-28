# Data Processing 

Before you do anything, please create necessary directories by running the following command in the project root directory:

```bash
mkdir -p data/raw data/temp data/train_test_split
```

To obtain the raw data, run the `setup.sh` in project root directory. This will download the data from the source and save it in the `data/raw` directory. Then, execute the following command to preprocess the data:

```bash
python clean_data.py
python remove_duplicates.py
python combine_short_text.py
python evenout_word_length.py
```

Then, you can run the train-test split script to split the data into training and testing sets. I forgot to set seed in one place, resulting in different splits every time. I will fix this in the future. 
