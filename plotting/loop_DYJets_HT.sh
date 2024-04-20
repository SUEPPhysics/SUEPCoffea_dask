#!/bin/bash

xcache=''

htbins_low=(
  70to100
  100to200
  200to400
  400to600
  600toInf
)

for htbin in ${htbins_low[@]}; do
    python make_plots.py -o test \
        --dataset DYJetsToLL_M-4to50_HT-${htbin}_TuneCP5_13TeV-madgraphMLM-pythia8 \
        -f ../temp_output/condor_test_DYJetsToLL_M-4to50_HT-${htbin}_TuneCP5_13TeV-madgraphMLM-pythia8.hdf5 \
        --era 2018 -s $(pwd) --isMC --PUreweight --Higgs_pt_reweight
done

htbins_high=(
  70to100
  100to200
  200to400
  400to600
  600to800
  800to1200
  1200to2500
  2500toInf
)

for htbin in ${htbins_high[@]}; do
    python make_plots.py -o test \
        --dataset DYJetsToLL_M-50_HT-${htbin}_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8 \
        -f ../temp_output/condor_test_DYJetsToLL_M-50_HT-${htbin}_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8.hdf5 \
        --era 2018 -s $(pwd) --isMC --PUreweight --Higgs_pt_reweight
done

mv *.pkl temp_output/
mv *.root temp_output/
