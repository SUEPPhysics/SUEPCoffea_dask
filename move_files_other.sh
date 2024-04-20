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

  # BKG section
  secondary="RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1"
  datasets=(
    WJetsToLNu_HT-100To200_TuneCP5_13TeV-madgraphMLM-pythia8+${secondary}-v1
    WJetsToLNu_HT-1200To2500_TuneCP5_13TeV-madgraphMLM-pythia8+${secondary}-v1
    WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8+${secondary}-v1
    WJetsToLNu_HT-2500ToInf_TuneCP5_13TeV-madgraphMLM-pythia8+${secondary}-v2
    WJetsToLNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8+${secondary}-v1
    WJetsToLNu_HT-600To800_TuneCP5_13TeV-madgraphMLM-pythia8+${secondary}-v1
    WJetsToLNu_HT-70To100_TuneCP5_13TeV-madgraphMLM-pythia8+${secondary}-v1
    WJetsToLNu_HT-800To1200_TuneCP5_13TeV-madgraphMLM-pythia8+${secondary}-v1
    WJetsToLNu_TuneCP5_13TeV-amcatnloFXFX-pythia8+${secondary}-v2
    WWTo1L1Nu2Q_4f_TuneCP5_13TeV-amcatnloFXFX-pythia8+${secondary}-v1
    WWTo2L2Nu_TuneCP5_13TeV-powheg-pythia8+${secondary}-v2
    WZTo1L1Nu2Q_4f_TuneCP5_13TeV-amcatnloFXFX-pythia8+${secondary}-v1
    WZTo1L3Nu_4f_TuneCP5_13TeV-amcatnloFXFX-pythia8+${secondary}-v1
    WZTo2Q2L_mllmin4p0_TuneCP5_13TeV-amcatnloFXFX-pythia8+${secondary}-v1
    WZTo3LNu_mllmin4p0_TuneCP5_13TeV-powheg-pythia8+${secondary}-v2
    ST_tW_Dilept_5f_DR_TuneCP5_13TeV-amcatnlo-pythia8+${secondary}-v2
    ST_t-channel_antitop_5f_InclusiveDecays_TuneCP5_13TeV-powheg-pythia8+${secondary}-v1
    ST_t-channel_top_5f_InclusiveDecays_TuneCP5_13TeV-powheg-pythia8+${secondary}-v1
  )

  for dataset in ${datasets[@]}; do
    name="${dataset}+NANOAODSIM"
    filein="${tag}_${name}_${mode}.pkl"
    fileout=${name}_${mode}.pkl
    mv "${filein}" "${outdir}/${fileout}"
  done
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

