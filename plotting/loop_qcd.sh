#!/bin/bash

xcache=''

ptbins=( \
    15to30 \
    30to50 \
    50to80 \
    80to120 \
    120to170 \
    170to300 \
    300to470 \
    470to600 \
    600to800 \
    800to1000 \
    1000to1400 \
    1400to1800 \
    1800to2400 \
    2400to3200 \
    3200toInf \
)

suffix="TuneCP5_13TeV_pythia8+RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v1+MINIAODSIM"

for ptbin in ${ptbins[@]}; do
    python make_plots.py -o test \
        --dataset QCD_Pt_${ptbin}_${suffix} \
        -f ../temp_output/condor_test_QCD_Pt_${ptbin}\+RunIISummer20UL18.hdf5 \
        --era 2018 -s $(pwd) --isMC --PUreweight --Higgs_pt_reweight
done
