## Making histograms for SUEP analyses

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

## Producing the histograms: overview

The histogram making is to be done over the hdf5 ntuples produced by `workflows/SUEP_coffea.py`.
This is achieved using `make_hists.py`, for example,

```
python make_hists.py --sample <sample> --output <output_tag> --tag <tag> --era <year> --isMC <bool> --doSyst <bool> --channel <channel>
```

To automatically run make_hists.py over all the samples, use `submit.py`, which supports parallelizing using multithread or slurm:

```
python submit.py --code plot --inputList <list> --sample <sample> --output <output_tag> --tag <tag> --era <year> --isMC <bool> --doSyst <bool> --channel <channel>
```

This will parallelize the script, producing one .root file of histograms for each sample.

## Producing the histograms: how to configure it

### Over what to run it
Either provide:
1. one filepath with -f
2. ntuple --tag and --sample for something in dataDirLocal.format(tag, sample) (or dataDirXRootD with --xrootd 1). This is the structure expected from the ntuple makers.
3. a directory of files: dataDirLocal (or dataDirXRootD with --xrootd 1)

### Selections, blinding, ABCD method
All of these are controlled by the `config` dictionary:

```
config = {
    'Cluster' : {
        'input_method' : 'CL',
        'method_var': 'SUEP_nconst_CL',
        'xvar' :'SUEP_S1_CL',
        'xvar_regions' : [0.3, 0.4, 0.5, 1.0],
        'yvar' : 'SUEP_nconst_CL',
        'yvar_regions' : [30, 50, 70, 1000],
        'SR' : [['SUEP_S1_CL', '>=', 0.5], ['SUEP_nconst_CL', '>=', 70]],
        'selections' : [['ht_JEC', '>', 1200], ['ntracks','>', 0]]
    },
    ...
}
```

This will:
- Grab all ntuple variables from the input method `CL` and fill histograms in the output method `Cluster`.
- Select all events that pass the `CL` method, using `method_var`.
- Blind the SR for data.
- Apply the `selections` to the DataFrame before filling the histograms.
- Make each histogram for each ABCD region, and make an ABCD prediction for the SR.

### Defining histograms

This is done in `hist_defs.py`.
The script will call `initialize_histograms()`, in this you can define your own function which is to be called for a particular method or channel (e.g. `Cluster` above) to add the histograms you want.
**All 1D and 2D histograms correctly named will be automatically filled by the script.**

"Correctly named" means:  `2D_variable1_vs_variable2_<label>`, `variable1_<label>`, or `<region>_variable1_<label>`, where `variable1,2` are in the ntuple, `<region>` is a numeric region if you're doing ABCD, and `<label>` are the output labels (above, `Cluster`) which, if you are running systematics, will be modified to include, e.g. `Cluster_JEC_up`. See `initialize_histograms()` for some examples.

### Systematics

For now these are hardcoded in the script for each analysis in `plot_systematic()`.
Systematics are either weights, or different variables that will be used to make selections.
Each variable will be plotted in a different histogram for each systematic, the output method name will be modified to include the systematic name for those histograms.


### Making new variables

The `config` dictionary can take the argument `new_variables`, which will be read by `fill_utils.py/make_new_variable()` to combine the DataFrame columns to make new variables with arbitrary functions.

The syntax is as follows:

```
config = {
    ...
    `new_variables`: [
        ['new_variable_name', callable, [callable inputs]]
    ]
    ...
}
```

For example,

```
    "new_variables": [
        ["SUEP_ISR_deltaPhi_CL", lambda x,y : abs(x-y), ["SUEP_phi_CL", "ISR_phi_CL"]],
        ["SUEP_ISR_deltaEta_CL", lambda x,y : abs(x-y), ["SUEP_eta_CL", "ISR_eta_CL"]]
    ]
```

One could define some function in to generate a whole set of these, if putting them all in the configuration dictionary becomes cumbersome.


## Producing the histograms: technical details

The following scripts are used:

1. `make_hists.py`: the main script to fill the histograms, define ABCD regions, apply selections, run systematics, and more.
2. `CMS_corrections/*.py`: all the systematics, called from the main script
3. `fill_utils.py`: a set of general helper functions for the main script
4. `hist_defs.py`: defining histograms

### make_hists.py

The DataFrame generated by the ntuple makers has the form:

| event variables (ht, ...) | CL vars (SUEP_S1_CL, ...) | GNN vars (SUEP_S1_GNN, ...) | Other methods |
| ------------------------- | ------------------------- | --------------------------- | ------------- |
| 0.3                       | 0.73                      | 0.22                        | ...           |
| 1.1                       | 1.0                       | Nan                         | ...           |
| 0.8                       | Nan                       | 0.12                        | ...           |
| ...                       | ...                       | ...                         | ...           |

where we have different input methods for our SUEP selection (CL, GNN, etc.), as well as event variables.
(The event variables are always filled, while the variables for each method are filled only if the event passes the method's selections, hence the `NaN`s).

The idea of this script is to map each of these methods to one or more set of histograms and fill them. Thus, for any input method, we can define:

1. Output tag: histograms will be named using this, and the systematics will extend each output tag (e.g. input method: CL --> output tag Cluster, with systematics Cluster_sys_up and Cluster_sys_down).
2. `SR`: signal region definition (e.g. `[['SUEP_S1_CL', '>=', 0.5], ['SUEP_nconst_CL', '>=', 80]]`), used to blind if needed.
3. [Semi-optional] `method_var`: variable to use to select events for this method.
4. [Optional] `selections`: a set of selections, syntax: either a list of `['variable', 'operator', value]` or a single string `"variable operator value"`. e.g. `[['ht', '>', 1200], ['ntracks > 0']]`.
5. [Optional] `xvar/yvar`: x/y variables for ABCD method
6. [Optional] `xvar_ragions/yvar_regions`: regions for ABCD method. N.B.: Include lower and upper bounds for all ABCD regions (e.g. `[0.0, 0.5, 1.0]`).
7. [Optional] `new_variables`: new variables to be defined as functions of existing variables in the DataFrame, syntax: `[['new_variable_name', callable, [callable inputs]]]`.

Each own input methods have their own selections, ABCD regions, and signal region. Multiple output tags can be defined for the same input method: i.e. different selections, ABCD methods, and SRs can be defined. Thus, each input method has a dictionary defined for it which is in turn stores in the `config` dictionary, with the key being the output label, e.g.

```
config = {
    'Cluster' : {
        'input_method' : 'CL',
        'method_var': 'SUEP_nconst_CL',
        'xvar' :'SUEP_S1_CL',
        'xvar_regions' : [0.35, 0.4, 0.5, 1.0],
        'yvar' : 'SUEP_nconst_CL',
        'yvar_regions' : [20, 40, 80, 1000],
        'SR' : [['SUEP_S1_CL', '>=', 0.5], ['SUEP_nconst_CL', '>=', 80]],
        'selections' : [['ht', '>', 1200], ['ntracks','>', 0]]
    },
    ...
}
```

This script will fill histograms, for each output method:

1. All event variables, e.g. ht will be put in ht_Cluster
2. All DataFrame columns from 'input_method', e.g. SUEP_S1_CL column will be plotted to histogram SUEP_S1_Cluster, as well as combinations of them in 2D histograms.
3. For each of the above, for each systematic, the up and down variation of the histograms
4. For each of the above, if doing ABCD, each histogram when in one particular ABCD region

**However, histograms are filled only if they are initialized in the output dictionary!!**

Thus, if for some reason you don't want to see a particular histogram for all ABCD regions, or all systematics, you can just define the nominal variation, and not the others, e.g. `ht_Cluster` and not `{region}ht_Cluster_{label}`, see `hist_defs.py`.

The main script relies from many functions in the helper script, `fill_utils.py`. A couple of the more important ones are explained below:

### fill_utils.py: prepareDataFrame()

    1. Grab only events that don't have NaN for the input method variables.
    2. Blind for data! Use SR to define signal regions and cut it out of df.
    3. Apply selections as defined in the 'selections' in the config dict.
    4. Define new variables as defined in 'new_variables' in the config dict.

### fill_utils.py: auto_fill()

    1. Plot variables from the DataFrame.
       1a. Event wide variables
       1b. Input method variables
    2. Plot 2D variables.
    3. Plot variables from the different ABCD regions as defined in the config dict.
       3a. Event wide variables
       3b. Input method variables

### Weights, Cross sections, and Systematics

The cross section and reweighting by the gen weight is done in `make_hists.py` directly, with some helper functions in `fill_utils.py`.
The systematics can be found in `CMS_corrections/*.py`, and are applied in `make_hists.py` on MC and signal samples.

1. **xsection**: These are defined in `../data/xsections_{}.json` for each HT or pT bin/sample, based on the era. These work with `gensumweight`, which is obtained from each hdf5 file's metadata, to scale that entire sample by `xsection/total_weight`, where `total_weight` here is the sum of all the files `gensumweights`. Cross sections are not applied for SUEP signal samples because of how we set up the limit code.

2. **pileup**: The weights are applied to MC based on the era only, they are applied based on the variable `Pileup_nTrueInt` directly to the events, and are defined in `pileup_weight.py`.

3. **track killing**: for each of the methods coming out of SUEPCoffea.py, we run through the whole method a second time with a certain % of the tracks in the event having been removed. The new method is then called `originalMethod_track_down` (e.g. `CL_track_down`). However, you don't need to add this method to the config dictionary in `make_hists.py`, the script will, for each method in config, automatically look for its track down variation, and make histograms for both. The track up variation is a symmetric variation of the difference betweeh the nominal and track_down histograms, and is also carried out in this script. Defined in `track_killing.py`.

4. **Pre shower weights**: `PS_weight` column found in the DataFrame, applied to event weight.

5. **GNN syst**: A systematic applied to GNN output on SUEP MC, defnied in `GNN_syst.py`. Obtained from the data vs. QCD difference when running the GNN on ISR jets instead of SUEPs. Practically, in the config for method `GNN`, need to add:

   a) `fGNNsyst`: path to a json file containing the bin corrections. Expected form is a nested dictionary of dimensions: (year x GNN_model x bin_syst_list). The bin_syst_list is a list of floats that correspond to a certain % correction on a predefined set of bins.

   b) `GNNsyst_bins`: set of bins corresponding to systematics as found in the file `fGNNsyst`.

6. **Higgs reweight**: applied only to event weight for SUEP's with mass of 125 GeV (Higgs case), as a function of the gen pT of the SUEP. Defined in `higgs_rewight.py`

7. **Trigger scale factor**: weight applied to event weight based on era only, defined in `triggerSF.py`.

8. **Jet energy corrections**: applied by cutting on different variations of the `ht` variations, and using the variables that have `<input_method>_track_down`.

9. [Optional] **weights**: These are weights that are defined based on each ABDC region (with the variables x_var, y_var) to force a third variable (z_var) to match for MC and data. These are produced in `plot.ipynb` and are saved as `.npy` files which are read in the script using `--weights=<file.npy>`.

## Next steps

Head to `plotting/` to find some exampls for how to plot the histograms you've just produced.
