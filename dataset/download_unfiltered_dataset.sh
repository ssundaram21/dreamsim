#!/bin/bash
mkdir -p ./dataset_unfiltered
cd dataset_unfiltered

mkdir -p ref
mkdir -p distort

# Download NIGHTS dataset

# store those in a list and loop through wget and unzip and rm
for i in {0..99..25}
do
    wget https://data.csail.mit.edu/nights/nights_unfiltered/ref_${i}_$(($i+24)).zip
    unzip -q ref_${i}_$(($i+24)).zip -d ref
    rm ref_*.zip

    wget https://data.csail.mit.edu/nights/nights_unfiltered/distort_${i}_$(($i+24)).zip
    unzip -q distort_${i}_$(($i+24)).zip -d distort
    rm distort_*.zip
done
