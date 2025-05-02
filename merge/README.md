## (Optional) Merge hdf5 files

Once you have produced the ntuples, your next step it to head to `plotting/` and make plots. However, since there are many hdf5 files for each sample, reading in a large amount of files can be slow; we can thus merge these hdf5 files together into larger ones to reduce the amount of files to read in. This can be done using `merge_ntuples.py`, which is ran on one sample, and `submit.py`, a wrapper for `merge_ntuples.py` to run it over many samples over slurm or multithread. The syntax for this is,

```
python merge_ntuples.py --sample=<sample> --tag=<tag> --isMC=<isMC>
```

N.B.: this is only set up to grab files from remote using XRootD, for now.

And the wrapper can be ran with,

```
python submit.py --tag=<tag> --code=merge  --inputList=<filelist>
```
