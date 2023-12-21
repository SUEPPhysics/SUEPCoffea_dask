### SUEP Coffea

[![Actions Status](https://github.com/chrispap95/hcaloms/workflows/CI/badge.svg)](https://github.com/SUEPPhysics/SUEPCoffea_dask/actions)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This code runs the SUEP analysis using the coffea framework. We use fastjet with awkward array inputs for jet clustering.

Three separate analyses and their documentation are here:

**Offline:**

CADI: https://cms.cern.ch/iCMS/analysisadmin/cadilines?line=EXO-23-002&tp=an&id=2650&ancode=EXO-23-002

Note: https://gitlab.cern.ch/tdr/notes/AN-22-133

PAS: https://gitlab.cern.ch/tdr/notes/EXO-23-002

Paper: https://gitlab.cern.ch/tdr/papers/EXO-23-002

**Scouting:**

CADI: https://cms.cern.ch/iCMS/analysisadmin/cadilines?line=EXO-23-001&tp=an&id=2649&ancode=EXO-23-001

Note: https://gitlab.cern.ch/tdr/notes/AN-21-119

PAS: NA

Paper: https://gitlab.cern.ch/tdr/papers/EXO-23-001

**ZH:**

CADI: https://cms.cern.ch/iCMS/analysisadmin/cadilines?line=EXO-23-003&tp=an&id=2651&ancode=EXO-23-003

Note: NA

PAS: NA

Paper: https://gitlab.cern.ch/tdr/papers/EXO-23-003

## Environment

### Singularity

The NTuple maker and the histmaker run by default using the coffea singularity provided through `/cvmfs`. You can use this locally by,

```bash
singularity shell -B ${PWD}:/work /cvmfs/unpacked.cern.ch/registry.hub.docker.com/coffeateam/coffea-dask:latest
```

If there are files in other folders that are necessary (the folder with your NTuples for example) you can bind additional folders with the following, which will allow one to access the files in the `/mnt` directory:

```bash
export SINGULARITY_BIND="/mnt"
```

or by adding `--bind /path1,/path2/,...` to the `singularity shell` command.

### Python environment

A minimal python environment is provided in `environment.yml`.
Install this with

```bash
conda env create -f environment.yml
conda activate suep
```

This environment should be enough for all the important parts of the workflow: the NTuple maker, the histmaker, and the plotting.

## Overview

The workflow is as follows:
1. `workflows/`: Produce NTuples from NanoAOD files using the NTuple maker for your analysis. This is done using coffea producers treating events as awkward arrays, and clustering using FastJet. The NTuples are stored in hdf5 files in tabular format in pandas dataframes. This is usually ran through HTCondor or Dask. See the README in `workflows/` for more information for how to run this for each analysis.
2. `histmaker/`: Make histograms from the NTuples using the histmaker. The histograms are stored in root files hist histograms. You can run this locally or through Slurm. See the README in `histmaker/` for more information for how to run this.
3. `plotting.`: Plot the histograms using the plotting notebooks. See the README in `plotting/` for more information.
