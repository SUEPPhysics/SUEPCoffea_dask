#!/bin/bash

function mv_files () {
  tag="$1"
  mode=$2

  # Output section
  outdir=plotting/temp_output_${mode}
  if [ "${tag}" != "test" ]; then
    outdir="plotting/${tag}_output_${mode}"
  fi
  if [ ! -d "${outdir}" ]; then
    mkdir -pv "${outdir}"
  fi

  # Signal section
  masses=( 125 400 750 1000 )
  decays=( darkPho darkPhoHad )

  for mass in ${masses[@]}; do
    for decay in ${decays[@]}
    do
      name=SUEP-m${mass}-${decay}+RunIIAutumn18
      if [ ! -f "${tag}_${name}_${mode}.pkl" ]; then
        echo "Warning: ${tag}_${name}_${mode}.pkl does not exist!"
        continue
      fi
      mv "${tag}_${name}_${mode}.pkl" "${outdir}/${name}-private+MINIAODSIM_${mode}.pkl"
    done
  done

  # More signal section
  temps=( 8 16 32 )
  mDark=( 2 4 8 )
  decays=( darkPho darkPhoHad )

  for temp in ${temps[@]}; do
    for mass in ${mDark[@]}; do
      for decay in ${decays[@]}
      do
        name=SUEP_mMed-125_mDark-${mass}_temp-${temp}_decay-${decay}
        if [ ! -f "${tag}_${name}_${mode}.pkl" ]; then
          echo "Warning: ${tag}_${name}_${mode}.pkl does not exist!"
          continue
        fi
        mv "${tag}_${name}_${mode}.pkl" "${outdir}/${name}_${mode}.pkl"
      done
    done
  done

  # BKG section
  primary_1=TuneCP5_13TeV_pythia8
  secondary_1=RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1
  primary_2=TuneCP5_13TeV-madgraphMLM-pythia8
  secondary_2=RunIISummer19UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1
  secondary_3=RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1
  datasets=(
    QCD_Pt_15to30_${primary_1}+${secondary_1}-v1+MINIAODSIM
    QCD_Pt_30to50_${primary_1}+${secondary_1}-v1+MINIAODSIM
    QCD_Pt_50to80_${primary_1}+${secondary_1}-v1+MINIAODSIM
    QCD_Pt_80to120_${primary_1}+${secondary_1}-v1+MINIAODSIM
    QCD_Pt_120to170_${primary_1}+${secondary_1}-v2+MINIAODSIM
    QCD_Pt_170to300_${primary_1}+${secondary_1}-v1+MINIAODSIM
    QCD_Pt_300to470_${primary_1}+${secondary_1}-v1+MINIAODSIM
    QCD_Pt_470to600_${primary_1}+${secondary_1}-v1+MINIAODSIM
    QCD_Pt_600to800_${primary_1}+${secondary_1}-v1+MINIAODSIM
    QCD_Pt_800to1000_${primary_1}+${secondary_1}-v1+MINIAODSIM
    QCD_Pt_1000to1400_${primary_1}+${secondary_1}-v1+MINIAODSIM
    QCD_Pt_1400to1800_${primary_1}+${secondary_1}-v1+MINIAODSIM
    QCD_Pt_1800to2400_${primary_1}+${secondary_1}-v1+MINIAODSIM
    QCD_Pt_2400to3200_${primary_1}+${secondary_1}-v1+MINIAODSIM
    QCD_Pt_3200toInf_${primary_1}+${secondary_1}-v1+MINIAODSIM
    TTJets_TuneCP5_13TeV-amcatnloFXFX-pythia8+${secondary_1}-v2+MINIAODSIM
    ttZJets_TuneCP5_13TeV_madgraphMLM_pythia8+${secondary_1}-v2+MINIAODSIM
    WWZ_4F_TuneCP5_13TeV-amcatnlo-pythia8+${secondary_1}_ext1-v2+MINIAODSIM
    ZZTo4L_TuneCP5_13TeV_powheg_pythia8+${secondary_1}-v2+MINIAODSIM
    ZZZ_TuneCP5_13TeV-amcatnlo-pythia8+${secondary_1}_ext1-v2+MINIAODSIM
    DY1JetsToLL_M-50_${primary_2}+${secondary_2}-v1+MINIAODSIM
    DY2JetsToLL_M-50_${primary_2}+${secondary_2}-v1+MINIAODSIM
    DY3JetsToLL_M-50_${primary_2}+${secondary_2}-v1+MINIAODSIM
    DY4JetsToLL_M-50_${primary_2}+${secondary_2}-v1+MINIAODSIM
    DYJetsToLL_M-10to50_TuneCP5_13TeV-amcatnloFXFX-pythia8+${secondary_1}-v2+MINIAODSIM
    DYJetsToLL_M-10to50_TuneCP5_13TeV-amcatnloFXFX-pythia8+${secondary_3}-v1+NANOAODSIM
    DYJetsToLL_M-10to50_${primary_2}+${secondary_1}-v1+MINIAODSIM
    ST_t-channel_antitop_5f_InclusiveDecays_TuneCP5_13TeV-powheg-pythia8+${secondary_3}-v1+NANOAODSIM
    ST_t-channel_top_5f_InclusiveDecays_TuneCP5_13TeV-powheg-pythia8+${secondary_3}-v1+NANOAODSIM
    ST_tW_Dilept_5f_DR_TuneCP5_13TeV-amcatnlo-pythia8+${secondary_3}-v2+NANOAODSIM
    WJetsToLNu_HT-100To200_${primary_2}+${secondary_3}-v1+NANOAODSIM
    WJetsToLNu_HT-1200To2500_${primary_2}+${secondary_3}-v1+NANOAODSIM
    WJetsToLNu_HT-200To400_${primary_2}+${secondary_3}-v1+NANOAODSIM
    WJetsToLNu_HT-2500ToInf_${primary_2}+${secondary_3}-v2+NANOAODSIM
    WJetsToLNu_HT-400To600_${primary_2}+${secondary_3}-v1+NANOAODSIM
    WJetsToLNu_HT-600To800_${primary_2}+${secondary_3}-v1+NANOAODSIM
    WJetsToLNu_HT-70To100_${primary_2}+${secondary_3}-v1+NANOAODSIM
    WJetsToLNu_HT-800To1200_${primary_2}+${secondary_3}-v1+NANOAODSIM
    WJetsToLNu_TuneCP5_13TeV-amcatnloFXFX-pythia8+${secondary_3}-v2+NANOAODSIM
    WWTo1L1Nu2Q_4f_TuneCP5_13TeV-amcatnloFXFX-pythia8+${secondary_3}-v1+NANOAODSIM
    WWTo2L2Nu_TuneCP5_13TeV-powheg-pythia8+${secondary_3}-v2+NANOAODSIM
    WZTo1L1Nu2Q_4f_TuneCP5_13TeV-amcatnloFXFX-pythia8+${secondary_3}-v1+NANOAODSIM
    WZTo1L3Nu_4f_TuneCP5_13TeV-amcatnloFXFX-pythia8+${secondary_3}-v1+NANOAODSIM
    WZTo2Q2L_mllmin4p0_TuneCP5_13TeV-amcatnloFXFX-pythia8+${secondary_3}-v1+NANOAODSIM
    WZTo3LNu_mllmin4p0_TuneCP5_13TeV-powheg-pythia8+${secondary_3}-v2+NANOAODSIM
    WZTo3LNu_5f_TuneCP5_13TeV-madgraphMLM-pythia8+${secondary_3}-v1+NANOAODSIM
    DY0JetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8+${secondary_3}-v2+NANOAODSIM
    DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8+${secondary_3}-v2+NANOAODSIM
    QCD_Pt-1000_MuEnrichedPt5_TuneCP5_13TeV-pythia8+${secondary_3}-v2+NANOAODSIM
    QCD_Pt-120To170_MuEnrichedPt5_TuneCP5_13TeV-pythia8+${secondary_3}-v2+NANOAODSIM
    QCD_Pt-15To20_MuEnrichedPt5_TuneCP5_13TeV-pythia8+${secondary_3}-v2+NANOAODSIM
    QCD_Pt-170To300_MuEnrichedPt5_TuneCP5_13TeV-pythia8+${secondary_3}-v2+NANOAODSIM
    QCD_Pt-20To30_MuEnrichedPt5_TuneCP5_13TeV-pythia8+${secondary_3}-v2+NANOAODSIM
    QCD_Pt-300To470_MuEnrichedPt5_TuneCP5_13TeV-pythia8+${secondary_3}-v2+NANOAODSIM
    QCD_Pt-30To50_MuEnrichedPt5_TuneCP5_13TeV-pythia8+${secondary_3}-v2+NANOAODSIM
    QCD_Pt-470To600_MuEnrichedPt5_TuneCP5_13TeV-pythia8+${secondary_3}-v2+NANOAODSIM
    QCD_Pt-50To80_MuEnrichedPt5_TuneCP5_13TeV-pythia8+${secondary_3}-v2+NANOAODSIM
    QCD_Pt-600To800_MuEnrichedPt5_TuneCP5_13TeV-pythia8+${secondary_3}-v2+NANOAODSIM
    QCD_Pt-800To1000_MuEnrichedPt5_TuneCP5_13TeV-pythia8+${secondary_3}-v2+NANOAODSIM
    QCD_Pt-80To120_MuEnrichedPt5_TuneCP5_13TeV-pythia8+${secondary_3}-v2+NANOAODSIM
    TTJets_TuneCP5_13TeV-amcatnloFXFX-pythia8+${secondary_3}-v1+NANOAODSIM
    ttZJets_TuneCP5_13TeV_madgraphMLM_pythia8+${secondary_3}-v2+NANOAODSIM
    WWZ_4F_TuneCP5_13TeV-amcatnlo-pythia8+${secondary_3}_ext1-v2+NANOAODSIM
    ZZTo4L_TuneCP5_13TeV_powheg_pythia8+${secondary_3}-v2+NANOAODSIM
    ZZZ_TuneCP5_13TeV-amcatnlo-pythia8+${secondary_3}_ext1-v2+NANOAODSIM
  )

  for dataset in ${datasets[@]}; do
    name=${dataset}
    filein="${tag}_${name}_${mode}.pkl"
    fileout=${name}_${mode}.pkl
    if [ ! -f "${filein}" ]; then
      echo "Warning: "${filein}" does not exist!"
      continue
    fi
    mv "${filein}" "${outdir}/${fileout}"
  done

  # Data section
  dataset="DoubleMuon+Run2018A-UL2018_MiniAODv2-v1+MINIAOD"
  if [ -f "${tag}_${dataset}_${mode}.pkl" ]; then
    mv "${tag}_${dataset}_${mode}.pkl" "${outdir}/${dataset}_${mode}.pkl"
  else
    echo "Warning: "${tag}_${dataset}_${mode}.pkl" does not exist!"
  fi
}

# Input section
tag="test"
cutflow=false
histograms=false
while getopts cht: option; do
  case "${option}" in
    t) tag="${OPTARG}";;
    c) cutflow=true;;
    h) histograms=true;;
    *) exit 1;;
  esac
done

if [ ${cutflow} == true ] && [ ${histograms} == true ]; then
  mv_files "${tag}" cutflow
  mv_files "${tag}" histograms
elif [ ${cutflow} == true ]; then
  mv_files "${tag}" cutflow
elif [ ${histograms} == true ]; then
  mv_files "${tag}" histograms
else
  echo "Please specify mode with -c and/or -h"
  exit 1
fi

