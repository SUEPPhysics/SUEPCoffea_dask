# Plotting the histograms

The outputs of make_hists.py are .root files containing Hist histograms.
You can open these in your own scripts and notebooks to view them, but a notebook, `plot.ipynb`, is provided with many useful functionalities as an example.

It is encouraged that each analysis has its own plotting scripts, but common functions should be put in `plot_utils.py` as much as possible.

## plot.ipynb

By specifying which root files you want, the notebook will automatically load all the histograms, merging samples, into one nested dictionary (`plots`), with dimensions (sample x plots).
This 'magic' is done through `plot_utils.py/loader()`.
The key feature of this is that it will combine histograms based on the samples, e.g. it will combine each `QCD_HT123To456` bin into one `QCD_HT` (as well as load it separately, if requested).
This is mapping of which samples should be merged into which name is defined by `plot_utils.py/getSampleNameAndBin()`.

Functions to plot in 1d and 2d, ratios of histograms, plots by bins or years, 1d slices of 2d hists, ABCD methods, and other useful plotting functions are defined within.
