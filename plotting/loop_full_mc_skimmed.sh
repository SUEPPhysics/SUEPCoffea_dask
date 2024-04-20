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
        --prepare --doABCD --Higgs_pt_reweight
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
    dataset=QCD_Pt_${ptbin}_${suffix}
    if [ $ptbin == 120to170 ]; then
        dataset="QCD_Pt_120to170_TuneCP5_13TeV_pythia8+RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2+MINIAODSIM"
    fi
    python make_plots.py -o test \
        --dataset $dataset \
        -f ../temp_output/condor_test_${dataset}.hdf5 \
        --era 2018 -s $(pwd) --isMC --PUreweight \
        --prepare --doABCD --Higgs_pt_reweight
done

samples=(
    TTJets_TuneCP5_13TeV-amcatnloFXFX-pythia8+RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2+MINIAODSIM
    DY1JetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8+RunIISummer19UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v1+MINIAODSIM
    DY2JetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8+RunIISummer19UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v1+MINIAODSIM
    DY3JetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8+RunIISummer19UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v1+MINIAODSIM
    DY4JetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8+RunIISummer19UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v1+MINIAODSIM
    ttZJets_TuneCP5_13TeV_madgraphMLM_pythia8+RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2+MINIAODSIM
    WWZ_4F_TuneCP5_13TeV-amcatnlo-pythia8+RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1_ext1-v2+MINIAODSIM
    ZZTo4L_TuneCP5_13TeV_powheg_pythia8+RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2+MINIAODSIM
    ZZZ_TuneCP5_13TeV-amcatnlo-pythia8+RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1_ext1-v2+MINIAODSIM
)

for sample in ${samples[@]}; do
    python make_plots.py -o test \
        --dataset ${sample} \
        -f ../temp_output/condor_test_${sample}.hdf5 \
        --era 2018 -s $(pwd) --isMC --PUreweight \
        --prepare --doABCD --Higgs_pt_reweight
done

# move all to final destination
mv *.pkl temp_output/
mv *.root temp_output/

