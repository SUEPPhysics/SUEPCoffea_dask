#!/bin/bash

xcache=''

generator_tag=TuneCP5_PSWeights_13TeV-madgraph-pythia8+RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1
htbins=(
  1000to1500_${generator_tag}-v1
  100to200_${generator_tag}-v1
  1500to2000_${generator_tag}-v1
  2000toInf_${generator_tag}-v1
  200to300_${generator_tag}-v1
  300to500_${generator_tag}-v1
  500to700_${generator_tag}-v1
  50to100_${generator_tag}-v1
  700to1000_${generator_tag}-v2
)

for htbin in ${htbins[@]}; do
    dataset=QCD_HT${htbin}+MINIAODSIM
    python make_plots.py -o test \
        --dataset $dataset \
        -f ../temp_output/condor_test_${dataset}.hdf5 \
        --era 2018 -s $(pwd) --isMC --PUreweight \
        --prepare --doABCD --Higgs_pt_reweight
done

# move all to final destination
mv *.pkl temp_output/
mv *.root temp_output/

