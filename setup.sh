mkdir -p full_pull
mkdir -p cleaned_data 
mkdir -p data
aria2c -x 16 -s 16 -i data_urls.txt -d full_pull
cd full_pull
gunzip *.gz
cd ..