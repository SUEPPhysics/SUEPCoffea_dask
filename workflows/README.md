# NTuple maker

This directory contains the files needed to create the SUEP ntuples from NanoAOD files. There are 4 main analyses contained here:

| Channel                        | Files                   |
| ------------------------------ | ----------------------- |
| ggF offline                    | SUEP_coffea, SUEP_utils |
| ggF scouting                   | SUEP_coffea, SUEP_utils |
| Associated Production (WH)     | SUEP_coffea_WH, WH_utils, SUEP_utils     |
| Associated Production (ZH)     | SUEP_coffea_ZH, ZH_utils, SUEP_utils     |

## How to run it locally, for one file

Each `condor_SUEP_<analysis>.py` is used to execute the main processor for that analysis, `SUEP_coffea_<analysis>.py`, which in turn relies on the various `*_utils.py`.

Inside the singularity container or python environment, you can run the ntuplemaker with the following command:

```bash
python3 condor_SUEP_<analysis>.py --isMC=<isMC> --era=<era> --sample=<sample> --infile=XXX.root
```

There are many other options for each analysis, check out the scripts for more details.

## How to run it on HTCondor

The `kraken_run.py` file will submit Condor jobs for all the files in specified samples.
Depending on where these are stored, you might need to edit the script to generate a list of files to run over.
An example submission can be seen below:

```bash
python kraken_run.py --isMC=1 --era=2018 --tag=my_tag --input=filelist/list_2018_MC_A01.txt
```

The submission will create `/output/directory/TAG/SAMPLE` for each sample in the input list.
If the tag already exists use the `--force` option if you are trying to resubmit/overwrite.

It will also create a log directory `/log/directory/jobs_TAG_SAMPLE` for each sample in the input list, where the condor jobs will be redirected to, as well as the condor.sub and executable that is passed to each job. Also in this directory, is a list of input files for each sample.

*Nota bene*: if reading many samples (> few 1000) from MIT T2, use the `--wait` option to wait between submitting samples. If several thousand start concurrently, the xrootd reads might overload T2 and cause headaches for everyone. Typical wait times can be a couple of minutes for small samples, to something like an hour for larger ones.

### Monitoring and resubmitting jobs

To monitor and resubmit jobs we can use the `monitor.py` file.
This script expects that each input .root NanoAOD file will have a corresponding output .hdf5 ntuple file in the output directory.

```bash
python monitor.py --tag=<tag name> --input=filelist/list_2018_MC_A01.txt
```

To resubmit you would use the `-r=1` option like below:

```bash
python monitor.py --tag=<tag name> --input=filelist/list_2018_MC_A01.txt -r=1
```

This will check the input files stored in the log directory and resubmit any that have failed by looking for the output ntuple files that don't have a matching input.

To automatically resubmit your jobs multiple times, we can use the `resubmit.py` file.

```bash
python resubmit.py --tag=<tag name> --resubmits=10 --hours=1
```

This will call `monitor.py` 10 times, resubmitting the files each time (i.e.,`-r=1`) and waiting 1 hour between each call. N.B.: all jobs that are still running after the number of hours specified will be removed.

### SUEP Coffea Scouting

The setup for the scouting analysis is similar to the offline analysis. We must simply run the scouting uproot job through the following (Note you must be in the singularity).

```bash
python3 condor_Scouting.py --isMC=0/1 --era=201X --dataset=<dataset> --infile=XXX.root
```

Here the "PFcand" has been added to be recognized by the coffea NanoAODSchema. Additionally root_rewrite.py is used to rewrite branches with "m" to "mass" until vector is implemented in the methods of coffea. Like in the offline analysis, condor jobs can be run on the jobs stored through the Kraken system:

```
python kraken_run.py --isMC=1 --era=2018 --tag=<tag name> --scout=1 --input=filelist/list_2018_scout_MC.txt
```

Signal samples are produced privately. To process these samples:

```
python kraken_run.py --isMC=1 --era=2018/2017/2016/2016apv --tag=<tag name> --scout=1 --private=1 --input=filelist/list_noHighMS_signal_scout.txt
```

All other commands listed above in the offline section will work similarly.

## CMS Corrections

Additional corrections for CMS analyses are stored in the CMS_corrections directory and can be called directly in the main workflows. Some, but not all of these corrections are common between channels.

See table below:

| Correction    | Channels       | Status                                         |
| ------------- | -------------- | ---------------------------------------------- |
| Golden JSON   | ggF, ML and ZH | - [x]                                          |
| JECs          | ggF, ML and ZH | - [x]                                          |
| Parton Shower | ggF, ML and ZH | - [x]                                          |
| Track killing | ggF, ML and ZH | - [x] (Percentages for scouting need updating) |
| Prefire       | ggF, ML and ZH | - [x]                                          |
| Lepton SF     | ZH             | - [ ]                                          |

Other uncertainties are added on the fly while making histograms in the plotting directory:

| Correction        | Channels       | Status            |
| ----------------- | -------------- | ----------------- |
| Pileup Weight     | ggF, ML and ZH | - [x]             |
| Trigger SF        | ggF, ML and ZH | - [ ] IN PROGRESS |
| Higgs pT reweight | ggF, ML and ZH | - [x]             |
| GNN Uncertainty   | ML             | - [x]             |

## Additional utils

Additional tools are used for various features:

1. root_rewrite: Fix naming issues with Scouting NTuples
2. pandas_utils: Tools to save pandas dataframes to hdf5 files
3. merger : Tool to merge output files together. This is not needed if you are using the pandas_accumulator in the ntuplemaker.
