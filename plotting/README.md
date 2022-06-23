## Plotting for SUEP Analysis

### Producing the Plots
The plotting is to be done over the hdf5 ntuples produced by workflows/SUEP_coffea.py. This is achieved using `make_plots.py`, using the options as follows,
```
python make_plots.py --dataset=<dataset> --tag=<tag> --era=2018 --isMC=0 --xrootd=0
```
The expected structure of your data is: `/path/<tag>/<dataset>/`. The xrootd option tells the script whether you need xrootd to access the files, or whether they are stored locally. You might need to change the dataDir in the script for it to point to the correct spot.

To automatically run make_plots.py over all the \<dataset\>s, use `plot_all.py`:
```
python plot_all.py --tag=<tag> --xrootd=0
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
