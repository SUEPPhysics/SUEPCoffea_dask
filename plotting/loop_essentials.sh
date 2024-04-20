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
        -s $(pwd) --isMC --isSignal  --PUreweight \
        --Higgs_pt_reweight --doABCD --prepare 
done
done

ptbins=(
    15to30
    30to50
    50to80
    80to120
    120to170
    170to300
    300to470
    470to600
    600to800
    800to1000
    1000to1400
    1400to1800
    1800to2400
    2400to3200
    3200toInf
)

suffix="TuneCP5_13TeV_pythia8+RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v1+MINIAODSIM"

for ptbin in ${ptbins[@]}; do
    python make_plots.py -o test \
        --dataset QCD_Pt_${ptbin}_${suffix} \
        -f ../temp_output/condor_test_QCD_Pt_${ptbin}\+RunIISummer20UL18.hdf5 \
        --era 2018 -s $(pwd) --isMC --PUreweight \
        --Higgs_pt_reweight --prepare --doABCD
done

samples=(
    TTJets_TuneCP5_13TeV-amcatnloFXFX-pythia8
    DY1JetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8
    DY2JetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8
    DY3JetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8
    DY4JetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8
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
        --era 2018 -s $(pwd) --isMC --PUreweight \
        --Higgs_pt_reweight --prepare --doABCD
done

# move all to final destination
mv *.pkl temp_output/
mv *.root temp_output/

