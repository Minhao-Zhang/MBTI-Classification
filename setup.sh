#!/bin/bash
sudo apt-get update
sudo apt-get install aria2
mkdir -p full_pull data
aria2c -x 16 -s 16 -i preprocessing/data_urls.txt -d full_pull
gunzip full_pull/*.gz