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
  primary=
  datasets=(
    QCD_Pt-15To20_MuEnrichedPt5_TuneCP5_13TeV-pythia8
    QCD_Pt-20To30_MuEnrichedPt5_TuneCP5_13TeV-pythia8
    QCD_Pt-30To50_MuEnrichedPt5_TuneCP5_13TeV-pythia8
    QCD_Pt-50To80_MuEnrichedPt5_TuneCP5_13TeV-pythia8
    QCD_Pt-80To120_MuEnrichedPt5_TuneCP5_13TeV-pythia8
    QCD_Pt-120To170_MuEnrichedPt5_TuneCP5_13TeV-pythia8
    QCD_Pt-170To300_MuEnrichedPt5_TuneCP5_13TeV-pythia8
    QCD_Pt-300To470_MuEnrichedPt5_TuneCP5_13TeV-pythia8
    QCD_Pt-470To600_MuEnrichedPt5_TuneCP5_13TeV-pythia8
    QCD_Pt-600To800_MuEnrichedPt5_TuneCP5_13TeV-pythia8
    QCD_Pt-800To1000_MuEnrichedPt5_TuneCP5_13TeV-pythia8
    QCD_Pt-1000_MuEnrichedPt5_TuneCP5_13TeV-pythia8
  )

  for dataset in ${datasets[@]}; do
    name=${dataset}
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

