#!/bin/bash

xcache=''

masses=( 125 400 750 1000 )

decays=( darkPho darkPhoHad )

for mass in ${masses[@]}; do
for decay in ${decays[@]}; do
    signal=SUEP-m${mass}-${decay}+RunIIAutumn18
    python make_plots.py -o test --era 2018 \
        --dataset ${signal}-private+MINIAODSIM \
        -f ../temp_output/condor_test_${signal}.hdf5 \
        -s $(pwd) --isMC --isSignal  --PUreweight --Higgs_pt_reweight
done
done
