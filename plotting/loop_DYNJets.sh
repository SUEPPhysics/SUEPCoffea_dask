#!/bin/bash

xcache=''

for i in `seq 1 4`; do
    python make_plots.py -o test \
        --dataset DY${i}JetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8 \
        -f ../temp_output/condor_test_DY${i}JetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8.hdf5 \
        --era 2018 -s $(pwd) --isMC --PUreweight --Higgs_pt_reweight
done

mv *.pkl temp_output/
mv *.root temp_output/
