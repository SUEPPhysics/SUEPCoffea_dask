#!/bin/bash
export _THISDIR=$(dirname "${BASH_SOURCE[0]}")
export SUEP_BASE=$( cd "$_THISDIR" )" && pwd )
echo "Created env. variable SUEP_BASE=${SUEP_BASE}"

hostname=$(hostname)
if [[ "$hostname" == *"mit.edu"* ]]; then
  export APPTAINER_BIND="/scratch,/data,/cvmfs,/home,/work"
  export SUEP_LOGS="/work/submit/$USER/SUEP/logs/"
  echo "Created env. variable SUEP_LOGS=${SUEP_LOGS}"
  export SUEP_OUT="/data/submit/cms/store/user/$USER/SUEP/"
  echo "Created env. variable SUEP_OUT=${SUEP_OUT}"
fi

CONTAINER=/cvmfs/unpacked.cern.ch/registry.hub.docker.com/coffeateam/coffea-dask:latest

# Add function for script wrapper
function suepRun(){
    singularity run $CONTAINER "$@"
}
echo "Created function suepRun"
