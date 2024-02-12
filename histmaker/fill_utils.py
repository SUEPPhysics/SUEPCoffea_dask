import json
import logging
import os
import subprocess
import sys
from collections import defaultdict
from copy import deepcopy

import numpy as np
import pandas as pd


def h5load(ifile: str, label: str):
    """
    Load a pandas DataFrame from a HDF5 file, including metadata.
    Nota bene: metadata is unstable, we have found that using pandas==1.4.1 and pytables==3.7.0 works.
    """
    try:
        with pd.HDFStore(ifile, "r") as store:
            try:
                data = store[label]
                metadata = store.get_storer(label).attrs.metadata
                return data, metadata

            except KeyError:
                logging.warning("No key", label, ifile)
                return 0, 0
    except BaseException:
        logging.warning("Some error occurred", ifile)
        return 0, 0


def open_ntuple(
    ifile: str, redirector: str = "root://submit50.mit.edu/", xrootd: bool = False
):
    """
    Open a ntuple, either locally or on xrootd.
    """
    if not xrootd and "root://" not in ifile:
        return h5load(ifile, "vars")
    else:
        if "root://" in ifile:
            xrd_file = ifile
        else:
            xrd_file = redirector + ifile
        just_file = ifile.split("/")[-1].split(".")[0]
        os.system(f"xrdcp -s {xrd_file} {just_file}.hdf5")
        return h5load(just_file + ".hdf5", "vars")


def close_ntuple(ifile: str) -> None:
    """
    Delete the ntuple after it has been copied over via xrootd (see open_ntuple).
    """
    just_file = ifile.split("/")[-1].split(".")[0]
    os.system(f"rm {just_file}.hdf5")


def get_git_info(path="."):
    """
    Get the current commit and git diff.
    """

    # Change directory to the git repo
    os.chdir(path)

    # Get the current commit and diff
    commit = (
        subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode("utf-8")
    )
    diff = subprocess.check_output(["git", "diff"]).strip().decode("utf-8")

    logging.debug(f"Current Commit: {commit}")
    logging.debug(f"Git Diff:\n{diff}")

    return commit, diff


def getXSection(dataset: str, year=None, path="../data/") -> float:
    xsection = 1

    xsec_file = f"{path}/xsections_{year}.json"
    with open(xsec_file) as file:
        MC_xsecs = json.load(file)
        try:
            xsection *= MC_xsecs[dataset]["xsec"]
            xsection *= MC_xsecs[dataset]["kr"]
            xsection *= MC_xsecs[dataset]["br"]
        except KeyError:
            logging.warning(
                f"WARNING: I did not find the xsection for {dataset} in {xsec_file}. Check the dataset name and the relevant yaml file."
            )
            return 1

    return xsection


def make_selection(
    df: pd.DataFrame, variable: str, operator: str, value, apply: bool = True
) -> pd.DataFrame:
    """
    Apply a selection on DataFrame df based on on the df column'variable'
    using the 'operator' and 'value' passed as arguments to the function.
    Returns the resulting DataFrame after the operation is applied.

    df: input dataframe.
    variable: df column.
    operator: see code below.
    value: value to cut variable on using operator.
    apply: toggles whether the selection is applied to the dataframe, or
    whether a list of booleans is returned matching the indices that
    passed and failed the selection.
    """

    if operator in ["greater than", "gt", ">"]:
        if apply:
            return df.loc[(df[variable] > value)]
        else:
            return df[variable] > value
    if operator in ["greater than or equal to", ">="]:
        if apply:
            return df.loc[(df[variable] >= value)]
        else:
            return df[variable] >= value
    elif operator in ["less than", "lt", "<"]:
        if apply:
            return df.loc[(df[variable] < value)]
        else:
            return df[variable] < value
    elif operator in ["less than or equal to", "<="]:
        if apply:
            return df.loc[(df[variable] <= value)]
        else:
            return df[variable] <= value
    elif operator in ["equal to", "eq", "=="]:
        if apply:
            return df.loc[(df[variable] == value)]
        else:
            return df[variable] == value
    else:
        raise Exception("Couldn't find operator requested " + operator)


def apply_scaling_weights(
    df,
    scaling_weights,
    abcd,
    regions="ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    x_var="SUEP_S1_CL",
    y_var="SUEP_nconst_CL",
    z_var="ht",
):
    """
    df: input DataFrame to scale
    abcd: dictionary of options, as in make_plots.py
    scaling_weights: nested dictionary, region x (bins or ratios)
    regions: string of ordered regions, used to apply corrections
    *_var: x/y are of the ABCD plane, z of the scaling histogram
    """

    x_var_regions = abcd["x_var_regions"]
    y_var_regions = abcd["y_var_regions"]
    iRegion = 0

    # S1 regions
    for i in range(len(x_var_regions) - 1):
        x_val_lo = x_var_regions[i]
        x_val_hi = x_var_regions[i + 1]

        # nconst regions
        for j in range(len(y_var_regions) - 1):
            y_val_lo = y_var_regions[j]
            y_val_hi = y_var_regions[j + 1]

            r = regions[iRegion]

            # from the weights
            bins = scaling_weights[r]["bins"]
            ratios = scaling_weights[r]["ratios"]

            # ht bins
            for k in range(len(bins) - 1):
                z_val_lo = bins[k]
                z_val_hi = bins[k + 1]
                ratio = ratios[k]

                zslice = (df[z_var] >= z_val_lo) & (df[z_var] < z_val_hi)
                yslice = (df[y_var] >= y_val_lo) & (df[y_var] < y_val_hi)
                xslice = (df[x_var] >= x_val_lo) & (df[x_var] < x_val_hi)

                df.loc[xslice & yslice & zslice, "event_weight"] *= ratio

            iRegion += 1
    return df


def prepare_DataFrame(
    df: pd.DataFrame,
    config: dict,
    label_out: str,
    blind: bool = True,
    isMC: bool = False,
    cutflow: dict = {},
) -> pd.DataFrame:
    """
    Applies blinding, selections, and makes new variables. See README.md for more details.

    INPUTS:
        df: input file DataFrame.
        config:  dictionary of definitions of ABCD regions, signal region, event selections.
        label_out: label associated with the output (e.g. "ISRRemoval"), as keys in
                   the config dictionary.
        sys: str of systematic being applied.

    OUTPUT: df: input DataFrame prepared for plotting
    """

    # 1. keep only events that passed this method, if any defined
    if config.get("xvar"):
        if config["xvar"] not in df.columns:
            return None
        df = df[~df[config["xvar"]].isnull()]

    # 2. blind
    if blind and not isMC:
        df = blind_DataFrame(df, label_out, config["SR"])
        if "SR2" in config.keys():
            df = blind_DataFrame(df, label_out, config["SR2"])

    # 3. make new variables
    if "new_variables" in config.keys():
        for var in config["new_variables"]:
            df = make_new_variable(df, var[0], var[1], *var[2])

    # 4. apply selections
    if "selections" in config.keys():
        for sel in config["selections"]:
            if (
                type(sel) is str
            ):  # converts "attribute operator value" to ["attribute", "operator", "value"] to pass to make_selection()
                sel = sel.split(" ")
            if (
                sel[0] not in df.keys()
            ):  # error out if variable doesn't exist in the DataFrame
                raise Exception(
                    f"Trying to apply a cut on a variable {sel[0]} that does not exist in the DataFrame"
                )
            if type(sel[2]) is str and sel[2].isdigit():
                sel[2] = float(
                    sel[2]
                )  # convert to float if the value to cut on is a number

            # make the selection
            df = make_selection(df, sel[0], sel[1], sel[2], apply=True)

            # store number of events passing using the event weights into the cutflow dict
            cutflow_label = (
                "cutflow_" + sel[0] + "_" + sel[1] + "_" + str(sel[2]) + "_" + label_out
            )
            if cutflow_label in cutflow.keys():
                cutflow[cutflow_label] += np.sum(df["event_weight"])
            else:
                cutflow[cutflow_label] = np.sum(df["event_weight"])

    return df


def make_new_variable(
    df: pd.DataFrame, name: str, function: callable, *columns: list
) -> pd.DataFrame:
    """
    Make a new column in the DataFrame df by applying the function to the columns
    passed as *columns. The new column will be named 'name'.
    """
    df[name] = function(*[df[col] for col in columns])
    return df


def fill_2d_distributions(df, output, label_out, input_method):
    keys = list(output.keys())
    keys_2Dhists = [k for k in keys if "2D" in k]
    df_keys = list(df.keys())

    for key in keys_2Dhists:
        if not key.endswith(label_out):
            continue
        string = key[
            len("2D") + 1 : -(len(label_out) + 1)
        ]  # cut out "2D_" and output label
        var1 = string.split("_vs_")[0]
        var2 = string.split("_vs_")[1]
        if var1 not in df_keys:
            var1 += "_" + input_method
            if var1 not in df_keys:
                continue
        if var2 not in df_keys:
            var2 += "_" + input_method
            if var2 not in df_keys:
                continue
        output[key].fill(df[var1], df[var2], weight=df["event_weight"])


def auto_fill(
    df: pd.DataFrame,
    output: dict,
    config: dict,
    label_out: str,
    isMC: bool = False,
    do_abcd: bool = False,
) -> None:
    input_method = config["input_method"]

    #####################################################################################
    # ---- Fill Histograms
    # Automatically fills all histograms that are declared in the output dict.
    #####################################################################################

    # 1. fill the distributions as they are saved in the dataframes
    # 1a. fill event wide variables
    event_plot_labels = [
        key for key in df.keys() if key + "_" + label_out in list(output.keys())
    ]
    for plot in event_plot_labels:
        output[plot + "_" + label_out].fill(df[plot], weight=df["event_weight"])

    # 1b. fill method variables
    method_plot_labels = [
        key
        for key in df.keys()
        if key.replace(input_method, label_out) in list(output.keys())
        and key.endswith(input_method)
    ]
    for plot in method_plot_labels:
        output[plot.replace(input_method, label_out)].fill(
            df[plot], weight=df["event_weight"]
        )

    # 2. fill some 2D distributions
    fill_2d_distributions(df, output, label_out, input_method)

    # 3. divide the dfs by region
    if do_abcd:
        regions = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        xvar = config["xvar"]
        yvar = config["yvar"]
        xvar_regions = config["xvar_regions"]
        yvar_regions = config["yvar_regions"]
        iRegion = 0
        for i in range(len(xvar_regions) - 1):
            x_val_lo = xvar_regions[i]
            x_val_hi = xvar_regions[i + 1]

            for j in range(len(yvar_regions) - 1):
                r = regions[iRegion] + "_"

                y_val_lo = yvar_regions[j]
                y_val_hi = yvar_regions[j + 1]

                x_cut = make_selection(
                    df, xvar, ">=", x_val_lo, False
                ) & make_selection(df, xvar, "<", x_val_hi, False)
                y_cut = make_selection(
                    df, yvar, ">=", y_val_lo, False
                ) & make_selection(df, yvar, "<", y_val_hi, False)
                df_r = df.loc[(x_cut & y_cut)]

                # double check the region is defined correctly
                assert (
                    df_r[
                        (df_r[xvar] > float(x_val_hi)) | (df_r[xvar] < float(x_val_lo))
                    ].shape[0]
                    == 0
                )

                # double check blinding
                if (
                    iRegion == (len(xvar_regions) - 1) * (len(yvar_regions) - 1)
                    and not isMC
                ):
                    if df_r.shape[0] > 0:
                        sys.exit(
                            label_out + ": You are not blinding correctly! Exiting."
                        )

                # by default, we only plot the ABCD variables in each region, to reduce the size of the output
                # the option do_abcd created a histogram of each variable for each region

                # 3a. fill event wide variables
                for plot in event_plot_labels:
                    if r + plot + "_" + label_out not in list(output.keys()):
                        continue
                    output[r + plot + "_" + label_out].fill(
                        df_r[plot], weight=df_r["event_weight"]
                    )

                # 3b. fill method variables
                for plot in method_plot_labels:
                    if r + plot.replace(input_method, label_out) not in list(
                        output.keys()
                    ):
                        continue
                    output[r + plot.replace(input_method, label_out)].fill(
                        df_r[plot], weight=df_r["event_weight"]
                    )

                iRegion += 1


def apply_normalization(plots: dict, norm: float) -> dict:
    if norm > 0.0:
        for plot in list(plots.keys()):
            plots[plot] = plots[plot] * norm
    else:
        logging.warning("Norm is 0, not applying normalization.")
    return plots


def get_track_killing_config(config: dict) -> dict:
    new_config = {}
    for label_out, _config_out in config.items():
        label_out_new = label_out
        new_config[label_out_new] = deepcopy(config[label_out])
        new_config[label_out_new]["input_method"] += "_track_down"
        new_config[label_out_new]["xvar"] += "_track_down"
        new_config[label_out_new]["yvar"] += "_track_down"
        for iSel in range(len(new_config[label_out_new]["SR"])):
            new_config[label_out_new]["SR"][iSel][0] += "_track_down"
        for iSel in range(len(new_config[label_out_new]["selections"])):
            # handle some exceptions by hand, for now
            if new_config[label_out_new]["selections"][iSel][0] in [
                "ht",
                "ngood_ak4jets",
                "ht_JEC",
            ]:
                continue
            new_config[label_out_new]["selections"][iSel][0] += "_track_down"

    return new_config


def get_jet_correction_config(config: dict, jet_correction: str) -> dict:
    new_config = {}
    for label_out, _config_out in config.items():
        label_out_new = label_out
        new_config[label_out_new] = deepcopy(config[label_out])
        for iSel in range(len(new_config[label_out_new]["selections"])):
            if "ht" == new_config[label_out_new]["selections"][iSel][0]:
                new_config[label_out_new]["selections"][iSel][0] += "_" + jet_correction
            elif "ht_JEC" == new_config[label_out_new]["selections"][iSel][0]:
                new_config[label_out_new]["selections"][iSel][0] += "_" + jet_correction
    return new_config


def read_in_weights(fweights):
    w = np.load(fweights, allow_pickle=True)
    scaling_weights = defaultdict(lambda: np.zeros(2))
    scaling_weights.update(w.item())
    return scaling_weights


def blind_DataFrame(df: pd.DataFrame, label_out: str, SR: list) -> pd.DataFrame:
    """
    Blind a DataFrame df by removing events that pass the signal region SR definition.
    Expects a SR defined as a list of lists,
    e.g. SR = [["SUEP_S1_CL", ">=", 0.5], ["SUEP_nconst_CL", ">=", 70]],
    """
    if len(SR) != 2:
        sys.exit(
            label_out
            + """: Make sure you have correctly defined your signal region.
            For now we only support a two-variable SR, because of the way
            this function was written. Exiting."""
        )
    df = df.loc[
        ~(
            make_selection(df, SR[0][0], SR[0][1], SR[0][2], apply=False)
            & make_selection(df, SR[1][0], SR[1][1], SR[1][2], apply=False)
        )
    ]
    return df
