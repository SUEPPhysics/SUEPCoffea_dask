## Plotting for SUEP Analysis

### Producing the Plots
The plotting is to be done over the hdf5 ntuples produced by workflows/SUEPCoffea_dask.py. This is achieved using `make_plots.py`, using the options as follows,
```
python make_plots.py --dataset=<dataset> --tag=<tag> --era=2018 --isMC=0 --xrootd=0
```
The expected structure of your data is: `/path/<tag>/<dataset>/`. The xrootd option tells the script whether you need xrootd to access the files, or whether they are stored locally.

To automatically run make_plots.py over all the \<dataset\>s, use `plot_all.py`:
```
python plot_all.py --tag=<tag> --xrootd=0
```
This will parallelize the plotting for each dataset.
  
  
### Plotting
The outputs of make_plots.py are .pkl files containing boost histograms. You can open these in your own scripts and notebooks to view them,
but a notebook, `plot.ipynb`, is provided with many useful functionalities. By specifying which pkl files you want to import using a tag,
the notebook will automatically load all the pkl files into one nested dictionary (`plots`), with dimensions (sample x plots),
where sample is either one of the \<dataset\>s, as well as all the combined QCD bins. Functions to plot in 1d and 2d, ratios of histograms,
plots by QCD bin, 1d slices of 2d hists, and other useful plotting functions are defined within.
