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
  primary_1=TuneCP5_13TeV_pythia8
  secondary_1=RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1
  primary_2=TuneCP5_PSWeights_13TeV-madgraph-pythia8
  datasets=(
    QCD_Pt_15to30_${primary_1}+${secondary_1}-v1
    QCD_Pt_30to50_${primary_1}+${secondary_1}-v1
    QCD_Pt_50to80_${primary_1}+${secondary_1}-v1
    QCD_Pt_80to120_${primary_1}+${secondary_1}-v1
    QCD_Pt_120to170_${primary_1}+${secondary_1}-v1
    QCD_Pt_170to300_${primary_1}+${secondary_1}-v1
    QCD_Pt_300to470_${primary_1}+${secondary_1}-v1
    QCD_Pt_470to600_${primary_1}+${secondary_1}-v1
    QCD_Pt_600to800_${primary_1}+${secondary_1}-v1
    QCD_Pt_800to1000_${primary_1}+${secondary_1}-v1
    QCD_Pt_1000to1400_${primary_1}+${secondary_1}-v1
    QCD_Pt_1400to1800_${primary_1}+${secondary_1}-v1
    QCD_Pt_1800to2400_${primary_1}+${secondary_1}-v1
    QCD_Pt_2400to3200_${primary_1}+${secondary_1}-v1
    QCD_Pt_3200toInf_${primary_1}+${secondary_1}-v1
    QCD_HT1000to1500_${primary_2}+${secondary_1}-v1
    QCD_HT100to200_${primary_2}+${secondary_1}-v1
    QCD_HT1500to2000_${primary_2}+${secondary_1}-v1
    QCD_HT2000toInf_${primary_2}+${secondary_1}-v1
    QCD_HT200to300_${primary_2}+${secondary_1}-v1
    QCD_HT300to500_${primary_2}+${secondary_1}-v1
    QCD_HT500to700_${primary_2}+${secondary_1}-v1
    QCD_HT50to100_${primary_2}+${secondary_1}-v1
    QCD_HT700to1000_${primary_2}+${secondary_1}-v2
  )

  for dataset in ${datasets[@]}; do
    name=${dataset}+NANOAODSIM
    filein="${tag}_${name}_${mode}.pkl"
    fileout=${name}_${mode}.pkl
    mv "${filein}" "${outdir}/${fileout}"
  done
}

# Input section
tag="nbjet_comparison"
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

