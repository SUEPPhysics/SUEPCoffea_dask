## Plotting for SUEP Analysis

### (Optional) Merge hdf5 files
Once you have produced the ntuples, your next step it to head to `plotting/` and make plots. However, since there are many hdf5 files for each dataset, reading in a large amount of files can be slow; we can thus merge these hdf5 files together into larger ones to reduce the amount of files to read in. This can be done using `merge_plots.py`, which is ran on one dataset, and `merge_all.py`, a wrapper for `merge_plots.py` to run it over many datasets. The syntax for this is
```
python merge_plots.py --dataset=<dataset> --tag=<tag> --isMC=<isMC>
```
N.B.: this is only set up to grab files from remote using XRootD, for now.
And the wrapper,
```
python multithread.py --tag=<tag> --code=merge  --inputList=<filelist>
```
  
### Example Workflow
Explained here is an example workflow. Each of these scripts should have more descriptions in the README's throughout this repo, but this guide should better explain how they fit together. 
  
**Produce NTuples**

  1. Find datasets to run (specified in a .txt file in `filelist/`), and lists of the .root files for the datasets (usually in `/home/tier3/cmsprod/catalog/t2mit/nanosc/E02/{}/RawFiles.00` as specified in `kraken_run.py`).
2. Run `kraken_run.py` to submit these jobs to HTCondor. Make sure to set the correct output and log directories in the python script.
3. These usually take a couple hours, which you can monitor using HTCondor. We don't expect perfect efficiency here, as normal in batch submission systems, but 80-90% is typical: if it's much less, the errors need to be investigated using the logs produced (found in `logdir`, specified in step 2). You can check how many of them have successfully finished using `python monitor.py -r=0`. Once a good amount of them have finished running (succesfully or not), usually after a couple hours, kill the currently running jobs, and resubmit using `python monitor.py -r=1`.
4. Repeat step 3. until you have achieved desired completion rate (suggested: >95% for MC, >99% for data).
  
**(Optional) Merge and Move NTuples**

5. Merge the hdf5 files for faster plotting, see section above.
6. Depending the way you have set it up, the output is on a remote filesystem, so move the hdf5 files (and/or the merged ones if you went through step 5), to a local filesystem for faster reading.
  
**Plotting**

7. Run `plot_all.py` over all the desired datasets to produce histograms, and `plot.ipynb` to display them.


### Producing the Plots
The plotting is to be done over the hdf5 ntuples produced by workflows/SUEP_coffea.py. This is achieved using `make_plots.py`, using the options as follows,
```
python make_plots.py --dataset=<dataset> --tag=<tag> --era=2018 --isMC=0 --xrootd=0
```
The expected structure of your data is: `/path/<tag>/<dataset>/`. The xrootd option tells the script whether you need xrootd to access the files, or whether they are stored locally. You might need to change the dataDir in the script for it to point to the correct spot.

To automatically run make_plots.py over all the \<dataset\>s, use mutlithreading:
```
python multithread.py --tag=<tag> --xrootd=0 --code=plot --inputList=<filelist>
```
This will parallelize the plotting for each dataset, producing one pkl file for each dataset.

#### Weights, Cross sections, and More
The script now uses a couple different scalings which are applied to MC datasets, and which are briefly explained here:
1. **xsection**: These are defined in `../data/xsections_{}.json` for each HT or pT bin/dataset, based on the era. These work with `gensumweight`, which is obtained from each hdf5 file's metadata, to scale that entire dataset by `xsection/total_weight`, where `total_weight` here is the sum of all the files `gensumweights`.
2. **pileup**: The weights are applied to MC based on the era only, they are applied based on the variable `Pileup_nTrueInt` directly to the events, and are defined in `pileup_weight.py`.
3. **ABCD weights**: These are weights that are defined based on each ABDC region (with the variables x_var, y_var) to force a third variable (z_var) to match for MC and data. These are produced in `plot.ipynb` and are saved as `.npy` files which are read in the script using `--weights=<file.npy>`.
  
  
### Plotting
The outputs of make_plots.py are .pkl files containing boost histograms. You can open these in your own scripts and notebooks to view them,
but a notebook, `plot.ipynb`, is provided with many useful functionalities. By specifying which pkl files you want to import using a tag,
the notebook will automatically load all the pkl files into one nested dictionary (`plots`), with dimensions (sample x plots),
where sample is either one of the \<dataset\>s, as well as all the combined QCD bins. Functions to plot in 1d and 2d, ratios of histograms,
plots by QCD bin, 1d slices of 2d hists, and other useful plotting functions are defined within.
