#!/bin/bash

# Input options
# -s : signal
# -b : background

# By default, run signal and background

all=1
signal=0
background=0
tag="kinematics"
extra_commands=0

while getopts 'sbt:c' flag; do
  case "${flag}" in
    s) all=0; signal=1 ;;
    b) all=0; background=1 ;;
    t) tag="${OPTARG}" ;;
    c) extra_commands=1 ;;
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
        --workflow SUEP_kinematics -o $tag \
        --json filelist/SUEP_signal_central_2018_from_mini.json \
        --executor futures -j 8 --chunk 5000 \
        --trigger TripleMu --era 2018 --isMC
fi

if [ $background -eq 1 ]; then
    echo "Processing BKG..."
    python dask/runner.py \
        --workflow SUEP_kinematics -o $tag \
        --json filelist/qcd_muenriched_jul2024.json \
        --executor futures -j 8 --chunk 10000 \
        --skimmed --trigger TripleMu \
        --era 2018 --isMC
fi

if [ $extra_commands -eq 1 ]; then
    ./move_files.sh -t "${tag}" -c -h
    # for f in ${tag}*.hdf5; do 
    #     rm $f
    # done
fi
