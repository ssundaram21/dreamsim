#!/bin/bash
mkdir -p ./dataset
cd dataset

mkdir -p ref
mkdir -p distort

# Download NIGHTS dataset
for i in $(seq -f "%03g" 0 99); do
    wget https://data.csail.mit.edu/nights/nights_chunked/ref/$i.zip
    unzip -q $i.zip -d ref/ && rm $i.zip
    wget https://data.csail.mit.edu/nights/nights_chunked/distort/$i.zip
    unzip -q $i.zip -d distort/ && rm $i.zip
done
