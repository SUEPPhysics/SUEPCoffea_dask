#!/bin/bash

# Input options
# -s : signal
# -b : background

# By default, run signal and background

all=1
signal=0
background=0
tag="test"

while getopts 'sbt:' flag; do
  case "${flag}" in
    s) all=0; signal=1 ;;
    b) all=0; background=1 ;;
    t) tag="${OPTARG}" ;;
    *) echo "Unexpected option ${flag}" ;;
  esac
done

if [ $all -eq 1 ]; then
    signal=1
    background=1
fi

if [ $signal -eq 1 ]; then
    echo "Processing signal..."
    python dask/runner.py \
        --workflow SUEP_ttbar_sources -o $tag --memory 4GB \
        --json filelist/SUEP_signal_leptonic_2018.json \
        --executor dask/lpc --chunk 300 \
        --trigger TripleMu --era 2018 --isMC
fi

if [ $background -eq 1 ]; then
    echo "Processing BKG - part 1..."
    python dask/runner.py \
        --workflow SUEP_ttbar_sources -o $tag --memory 4GB \
        --json filelist/full_mc_skimmed_merged_1.json \
        --executor dask/lpc --chunk 500 \
        --skimmed --trigger TripleMu \
        --era 2018 --isMC

    echo "Processing BKG - part 2..."
    python dask/runner.py \
        --workflow SUEP_ttbar_sources -o $tag --memory 4GB \
        --json filelist/full_mc_skimmed_merged_2.json \
        --executor dask/lpc --chunk 500 \
        --skimmed --trigger TripleMu \
        --era 2018 --isMC
fi
