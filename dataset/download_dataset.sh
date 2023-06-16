#!/bin/bash
mkdir -p ./dataset
cd dataset

# Download NIGHTS dataset
wget -O nights.zip https://data.csail.mit.edu/nights/nights.zip

unzip nights.zip
rm nights.zip
