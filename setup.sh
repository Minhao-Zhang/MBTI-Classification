#!/bin/bash
sudo apt-get update
sudo apt-get install aria2
mkdir -p data data/raw data/temp data/train_test_split
aria2c -x 8 -s 8 -i preprocessing/data_urls.txt -d data/raw
gunzip data/raw/*.gz