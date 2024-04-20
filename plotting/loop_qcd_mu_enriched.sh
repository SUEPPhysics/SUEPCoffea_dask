#!/bin/bash

xcache=""

ptbins=(
    15To20
    20To30
    30To50
    50To80
    80To120
    120To170
    170To300
    300To470
    470To600
    600To800
    800To1000
    1000
)

suffix="MuEnrichedPt5_TuneCP5_13TeV-pythia8"

for ptbin in ${ptbins[@]}; do
    python make_plots.py -o test \
        --dataset QCD_Pt-${ptbin}_${suffix} \
        -f ../temp_output/condor_test_QCD_Pt-${ptbin}_${suffix}.hdf5 \
        --era 2018 -s $(pwd) --isMC --PUreweight --Higgs_pt_reweight
done

mv *.pkl temp_output/
mv *.root temp_output/
