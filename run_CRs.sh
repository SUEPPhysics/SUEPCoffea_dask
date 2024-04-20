#!/bin/bash

# Example usage:
# ./run_CRs.sh -s -b -d -o -t CR_prompt_combine -w SUEP_combine -r CR_prompt -c

# Input options
# -s : signal
# -b : background
# -d : data
# -o : other background
# -u : unskimmed MC
# -t : tag
# -w : workflow
# -g : trigger
# -r : region
# -c : run extra commands

# By default, run signal and background

all=1
signal=0
background=0
other_bkg=0
data=0
unskimmed=0
workflow="SUEP_CRprompt"
tag="test"
trigger="TripleMu"
region=""
extra_commands=0

while getopts 'sboducg:t:r:w:' flag; do
  case "${flag}" in
    s) all=0; signal=1 ;;
    b) all=0; background=1 ;;
    o) all=0; other_bkg=1 ;;
    d) all=0; data=1 ;;
    u) all=0; unskimmed=1 ;;
    c) extra_commands=1 ;;
    g) trigger="${OPTARG}" ;;
    t) tag="${OPTARG}" ;;
    r) region="${OPTARG}" ;;
    w) workflow="${OPTARG}" ;;
    *) echo "Unexpected option ${flag}" ;;
  esac
done

if [ $all -eq 1 ]; then
    signal=1
    background=1
    other_bkg=1
    data=1
fi

if [ $signal -eq 1 ]; then
    echo "Processing signal..."
    python dask/runner.py \
        --workflow "${workflow}" -o "${tag}" --region "${region}" \
        --json filelist/SUEP_signal_all_leptonic_2018.json \
        --executor futures -j 4 --chunk 10000 \
        --trigger "${trigger}" --era 2018 --isMC
fi

if [ $other_bkg -eq 1 ]; then
    echo "Processing other BKG..."
    python dask/runner.py \
        --workflow "${workflow}" -o "${tag}" \
        --json filelist/nanoaodsim_skimmed.json \
        --executor futures -j 8 --chunk 20000 \
        --skimmed --trigger "${trigger}" --region "${region}" \
        --era 2018 --isMC
fi

if [ $background -eq 1 ]; then
    echo "Processing BKG..."
    python dask/runner.py \
        --workflow "${workflow}" -o "${tag}" --memory 2GB \
        --json filelist/full_mc_skimmed_merged.json \
        --executor dask/lpc --chunk 20000 \
        --skimmed --trigger "${trigger}" --region "${region}" \
        --era 2018 --isMC
fi

if [ $data -eq 1 ]; then
    echo "Processing data..."
    python dask/runner.py \
        --workflow "${workflow}" -o "${tag}" --memory 2GB \
        --json filelist/data_Run2018A_1fb_unskimmed.json \
        --executor dask/lpc --chunk 15000 \
        --trigger "${trigger}" --region "${region}" \
        --era 2018
fi

if [ $unskimmed -eq 1 ]; then
    echo "Processing unskimmed MC..."
    python dask/runner.py \
        --workflow "${workflow}" -o "${tag}" --memory 4GB \
        --json filelist/mc_unskimmed_necessary_new_1.json \
        --executor dask/lpc --chunk 50000 \
        --max-scaleout 400 --region "${region}" \
        --trigger "${trigger}" \
        --era 2018 --isMC
fi

if [ $extra_commands -eq 1 ]; then
    ./move_files.sh -t "${tag}" -c -h
    for f in ${tag}*.hdf5; do 
        rm $f
    done
fi
