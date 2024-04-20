#!/bin/bash

xcache=''

samples=(
    ttZJets_TuneCP5_13TeV_madgraphMLM_pythia8
    WWZ_4F_TuneCP5_13TeV-amcatnlo-pythia8
    WWZJetsTo4L2Nu_4F_TuneCP5_13TeV-amcatnlo-pythia8
    ZZTo4L_TuneCP5_13TeV_powheg_pythia8
    ZZZ_TuneCP5_13TeV-amcatnlo-pythia8
)

for sample in ${samples[@]}; do
    python make_plots.py -o test \
        --dataset ${sample} \
        -f ../temp_output/condor_test_${sample}.hdf5 \
        --era 2018 -s $(pwd) --isMC --PUreweight --Higgs_pt_reweight
done
