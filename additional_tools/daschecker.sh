#!/bin/bash

# setting up environment
export VO_CMS_SW_DIR=/cvmfs/cms.cern.ch
source $VO_CMS_SW_DIR/cmsset_default.sh
echo "Checking proxy..."
timeleft=$(voms-proxy-info --actimeleft)
echo "$timeleft seconds left in proxy"
echo

datasets=("EGamma+Run2018A-UL2018_MiniAODv2_GT36-v1+MINIAOD" "EGamma+Run2018B-UL2018_MiniAODv2_GT36-v1+MINIAOD" "EGamma+Run2018C-UL2018_MiniAODv2_GT36-v1+MINIAOD" "EGamma+Run2018D-UL2018_MiniAODv2_GT36-v3+MINIAOD" "SingleMuon+Run2018A-UL2018_MiniAODv2_GT36-v2+MINIAOD" "SingleMuon+Run2018B-UL2018_MiniAODv2_GT36-v2+MINIAOD" "SingleMuon+Run2018C-UL2018_MiniAODv2_GT36-v3+MINIAOD" "SingleMuon+Run2018D-UL2018_MiniAODv2_GT36-v2+MINIAOD")  # Define your list of datasets here

for dataset in "${datasets[@]}"; do
    das_dataset=$(echo "/$dataset" | tr + /)
    echo "Analyzing $dataset for differences between DAS MINIAOD and our T2 NANOAOD"
    echo "Querying DAS..."
    das_output=$(dasgoclient --query "file dataset=$das_dataset")
    das_files=$(echo "$das_output" | grep -oE '[^/]+\.root$' | sort)
    echo "Querying T2..."
    t2_output=$(xrdfs root://xrootd.cmsaf.mit.edu/ ls /store/user/paus/nanosu/A02/$dataset/)
    t2_files=$(echo "$t2_output" | grep -oE '[^/]+\.root$' | sort)
    echo "Difference:"
    diff_output=$(diff <(echo "$das_files") <(echo "$t2_files"))
    echo "$diff_output" | awk '/^>/ || /^</ {print}'
    echo
done

