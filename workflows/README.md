### Workflows

This directory contains the files needed to create the SUEP ntuples from NanoAOD files. There are 3 main analyses contained here:

| Channel               | Files                   |
| --------------------- | ----------------------- |
| ggF offline/scouting  | SUEP_Coffea, SUEP_utils |
| Graph Neural Net      | ML_Coffea, ML_utils     |
| Associated Production | ZH_Coffea, ZH_utils     |

## CMS Corrections

Additional corrections for CMS analyses are stored in the CMS_corrections directory and can be called directly in the main workflows. Some, but not all of these corrections are common between channels.

See table below:

| Correction        | Channels       | Status                                         |
| ----------------- | -------------- | ---------------------------------------------- |
| Golden JSON       | ggF, ML and ZH | - [x]                                          |
| JECs              | ggF, ML and ZH | - [x]                                          |
| Parton Shower     | ggF, ML and ZH | - [x]                                          |
| Track killing     | ggF, ML and ZH | - [x] (Percentages for scouting need updating) |
| Higgs pT reweight | ggF, ML and ZH | - [ ]                                          |
| Prefire           | ggF, ML and ZH | - [x] Need CMSSW_10_6_26 (A02)                 |
| Lepton SF         | ZH             | - [ ]                                          |

Other uncertainties are added on the fly while making histograms in the plotting directory:

| Correction      | Channels       | Status            |
| --------------- | -------------- | ----------------- |
| Pileup Weight   | ggF, ML and ZH | - [x]             |
| Trigger SF      | ggF, ML and ZH | - [ ] IN PROGRESS |
| GNN Uncertainty | ML             | - [x]             |

## Additional utils

Additional tools are used for various features:

1. root_rewrite: Fix naming issues with Scouting NTuples
2. pandas_utils: Tools to save pandas dataframes to hdf5 files
3. merger : Tool to merge output files together
