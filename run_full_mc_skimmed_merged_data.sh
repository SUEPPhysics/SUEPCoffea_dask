#!/bin/bash

# Input options
# -s : signal
# -b : background
# -d : data

# By default, run signal and background

all=1
signal=0
background=0
data=0
tag="test"

while getopts 'sbdt:' flag; do
  case "${flag}" in
    s) all=0; signal=1 ;;
    b) all=0; background=1 ;;
    d) all=0; data=1 ;;
    t) tag="${OPTARG}" ;;
    *) echo "Unexpected option ${flag}" ;;
  esac
done

if [ $all -eq 1 ]; then
    signal=1
    background=1
    data=1
fi

if [ $signal -eq 1 ]; then
    echo "Processing signal..."
    python dask/runner.py \
        --workflow SUEP_data -o ${tag} --memory 4GB \
        --json filelist/SUEP_signal_leptonic_2018.json \
        --executor futures --chunk 5000 \
        --trigger TripleMu --era 2018 --isMC
fi

if [ $background -eq 1 ]; then
    echo "Processing BKG..."
    python dask/runner.py \
        --workflow SUEP_data -o ${tag} --memory 4GB \
        --json filelist/full_mc_skimmed_merged.json \
        --executor dask/lpc --chunk 10000 \
        --skimmed --trigger TripleMu \
        --era 2018 --isMC
fi

if [ $data -eq 1 ]; then
    echo "Processing data..."
    python dask/runner.py \
        --workflow SUEP_data -o ${tag} --memory 4GB \
        --json filelist/data_Run2018A_1fb_unskimmed.json \
        --executor dask/lpc --chunk 5000 \
        --trigger TripleMu \
        --era 2018
fi
