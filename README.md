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

## to run the producer

```bash
python3 condor_SUEP_WS.py --isMC=0/1 --era=201X --dataset=<dataset> --infile=XXX.root
```

If you do not have the requirements set up then you can also run this through the docker container that the coffea team provides. This is simple and easy to do. You just need to enter the Singularity and then issue the command above. To do this use:

```bash
singularity shell -B ${PWD}:/work /cvmfs/unpacked.cern.ch/registry.hub.docker.com/coffeateam/coffea-dask:latest
```

If there are files in other folders that are necessary (The folder with your NTuples for example) you can bind additional folders like with the following which will allow one to access the files in the `/mnt` directory:

```bash
export SINGULARITY_BIND="/mnt"
```

## Manually control condor jobs rather than Dask

The `kraken_run.py` file which will submit Condor jobs for all the files in specified datasets. This submission currently uses `xrdfs` to find the files stored on Kraken. An example submission can be seen below:

```bash
python kraken_run.py --isMC=1 --era=2018 --tag=<tag name> --input=filelist/list_2018_MC_A01.txt
```

The submission will name a directory in the output directory after the tag name you input. If the tag already exists use the `--force` option if you are trying to resubmit/overwrite.

Note that this submission will look for the dataset `xsec` in `xsections_<era>.yaml`.

To monitor and resubmit jobs we can use the `monitor.py` file.

```bash
python monitor.py --tag=<tag name> --input=filelist/list_2018_MC_A01.txt
```

To resubmit you must specify to resubmit like below:

```bash
python monitor.py --tag=<tag name> --input=filelist/list_2018_MC_A01.txt -r=1
```

To automatically resubmit your jobs multiple times, we can use the `resubmit.py` file.

```bash
python resubmit.py --tag=<tag name> --resubmits=10 --hours=1
```

This will call `monitor.py` 10 times, resubmitting the files each time (i.e., -r=1) and waiting 1 hour between each call. N.B.: all jobs that are still running after the number of hours specified will be removed.

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
python kraken_run.py --isMC=1 --era=2018/2017/2016/2016apv --tag=<tag name> --private=1 --scout=1 --input=filelist/list_noHighMS_signal_scout.txt
```

All other commands listed above in the offline section will work similarly.

### Example Workflow

Explained here is an example workflow. Each of these scripts should have more descriptions in the README's throughout this repo, but this guide should better explain how they fit together.

**Produce NTuples**

1. Find datasets to run (specified in a .txt file in `filelist/`), and lists of the .root files for the datasets (usually in `/home/tier3/cmsprod/catalog/t2mit/nanosc/E02/{}/RawFiles.00` as specified in `kraken_run.py`).
2. Run `kraken_run.py` to submit these jobs to HTCondor. Make sure to set the correct output and log directories in the python script.
3. These usually take a couple hours, which you can monitor using HTCondor. We don't expect perfect efficiency here, as normal in batch submission systems, but 80-90% is typical: if it's much less, the errors need to be investigated using the logs produced (found in `logdir`, specified in step 2). You can check how many of them have successfully finished using `python monitor.py -r=0`. Once a good amount of them have finished running (successfully or not), usually after a couple hours, kill the currently running jobs, and resubmit using `python monitor.py -r=1`.
   You can use the resubmit.py file to automatically resubmit the failed jobs for a few times. Example `nohup python resubmit.py  --tag=wj_test_PFHT --input=filelist/list_2018_scout_data.txt &`, which runs `python monitor.py --tag=wj_test_PFHT --input=filelist/list_2018_scout_data.txt` once an hour for 10 times (you can changes the default setups.)
4. Repeat step 3. until you have achieved desired completion rate (suggested: >95% for MC, >99% for data).

**(Optional) Merge and Move NTuples**

5. Merge the hdf5 files for faster plotting, see section above.
6. Depending the way you have set it up, the output is on a remote filesystem, so move the hdf5 files (and/or the merged ones if you went through step 5), to a local filesystem for faster reading.

**Plotting**

7. Run `make_plots.py` (if needed, with `multithread.py`) over all the desired datasets to produce histograms
8. Use plotting notebooks like `plot.ipynb` to display them.
