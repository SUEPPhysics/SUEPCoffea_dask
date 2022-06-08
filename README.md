### SUEP Coffea

This code runs the SUEP analysis using the coffea framework. We use fastjet with awkward array inputs for jet clustering.

## to run the producer

```bash
python3 condor_SUEP_WS.py --isMC=0/1 --era=201X --dataset=<dataset> --infile=XXX.root
```

If you do not have the requirements set up then you can also run this through the docker container that the coffea team provides. This is simple and easy to do. You just need to enter the Singularity and then issue the command above. To do this use:

```bash
singularity shell -B ${PWD}:/work /cvmfs/unpacked.cern.ch/registry.hub.docker.com/coffeateam/coffea-dask:latest
```

If there are files in other folders that are necessary (The folder with your NTuples for example) you can bind additional folders like with the following which will allow one to access the files in the /mnt directory:

```bash
export SINGULARITY_BIND="/mnt"
```
  
## Manually control condor jobs rather than Dask

The kraken_run.py file which will submit Condor jobs for all the files in specified datasets. This submission currenty uses xrdfs to find the files stored on Kraken. An example submission can be seen below:

```
python kraken_run.py --isMC=1 --era=2018 --tag=<tag name> --input=filelist/list_2018_MC_A01.txt 
```
The submission will name a directory in the output directory after the tage name you input. If the tag already exists use the ```--force``` option if you are trying to resubmit/overwrite.

Note that this submission will look for the dataset xsec in xsections_<era>.yaml.
  
To monitor and resubmit jobs we can use the monitor.py file. 
  
```
python monitor.py --tag=<tag name> --input=filelist/list_2018_MC_A01.txt
```
To resubmit you must specify to resubmit like below:

```
python monitor.py --tag=<tag name> --input=filelist/list_2018_MC_A01.txt -r=1 
```
To automatically resubmit your jobs multiple times, we can use the resubmit.py file.
```
python resubmit.py --tag=<tag name> --resubmits=10 --hours=1
```
This will call monitor.py 10 times, resubmitting the files each time (i.e., -r=1) and waiting 1 hour between each call. N.B.: all jobs that are still running after the number of hours specified will be removed.
  
  
### SUEP Coffea Scouting
 
The setup for the scouting analysis is similar to the offline analysis. We must simply run the scouting uproot job through the following (Note you must be in the singularity).
  
```bash
python3 condor_Scouting.py --isMC=0/1 --era=201X --dataset=<dataset> --infile=XXX.root
```
  
Here the "PFcand" has been added to be recognized by the coffea NanoAODSchema. Additionally root_rewrite.py is used to rewrite branches with "m" to "mass" until vector is implemented in the methods of coffea. Like in the offline analysis, condor jobs can be run on the jobs stored through the Kraken system:
  
```
python kraken_run.py --isMC=1 --era=2018 --tag=<tag name> --scout=1 --input=filelist/list_2018_scout_MC.txt 
```
 
All other commands listed above in the offline section will work similarly.
