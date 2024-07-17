#!/bin/bash

sudo apt-get install aria2
mkdir -p full_pull cleaned_data data pickled models
aria2c -x 16 -s 16 -i data_urls.txt -d full_pull
cd full_pull
gunzip *.gz
cd ..