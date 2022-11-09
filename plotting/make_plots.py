# Make plots for SUEP analysis. Reads in hdf5 files and outputs to pickle and root files
import argparse
import getpass
import json
import os
import pickle
import subprocess
import sys
from collections import defaultdict
from copy import deepcopy

import higgs_reweight
import numpy as np
import pandas as pd

# Import our own functions
import pileup_weight
import triggerSF
import uproot
from hist import Hist
from plot_utils import *
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Famous Submitter")
# Name of the output file tag
parser.add_argument("-o", "--output", type=str, help="output tag", required=True)
# Where the files are coming from. Either provide:
# 1. filepath with -f
# 2. tag and dataset for something in dataDirLocal.format(tag, dataset)
# 3. tag, dataset and xrootd = 1 for something in dataDirXRootD.format(tag, dataset)
parser.add_argument(
    "-dataset",
    "--dataset",
    type=str,
    default="QCD",
    help="dataset name",
    required=False,
)
parser.add_argument(
    "-t", "--tag", type=str, default="IronMan", help="production tag", required=False
)
parser.add_argument(
    "-f", "--file", type=str, default="", help="Use specific input file"
)
parser.add_argument(
    "--xrootd",
    type=int,
    default=0,
    help="Local data or xrdcp from hadoop (default=False)",
)
# optional: call it with --merged = 1 to append a /merged/ to the paths in options 2 and 3
parser.add_argument("--merged", type=int, default=1, help="Use merged files")
# some info about the files, highly encouraged to specify every time
parser.add_argument("-e", "--era", type=int, help="era", required=True)
parser.add_argument("--isMC", type=int, help="Is this MC or data", required=True)
parser.add_argument("--scouting", type=int, default=0, help="Is this scouting or no")
# some parameters you can toggle freely
parser.add_argument("--doSyst", type=int, default=0, help="make systematic plots")
parser.add_argument("--doInf", type=int, default=0, help="")
parser.add_argument(
    "--blind", type=int, default=1, help="Blind the data (default=True)"
)
parser.add_argument(
    "--weights",
    type=str,
    default="None",
    help="Pass the filename of the weights, e.g. --weights weights.npy",
)
options = parser.parse_args()

###################################################################################################################
# Script Parameters
###################################################################################################################

outDir = f"/work/submit/{getpass.getuser()}/SUEP/outputs/"
# define these if --xrootd 0
dataDirLocal = "/data/submit//cms/store/user/{}/SUEP/{}/{}/".format(
    getpass.getuser(), options.tag, options.dataset
)
# and these is --xrootd 1
redirector = "root://submit50.mit.edu/"
dataDirXRootD = "/cms/store/user/{}/SUEP/{}/{}/".format(
    getpass.getuser(), options.tag, options.dataset
)

"""
Define output plotting methods, each draws from an input_method (outputs of SUEPCoffea),
and can have its own selections, ABCD regions, and signal region.
Multiple plotting methods can be defined for the same input method, as different
selections and ABCD methods can be applied.
N.B.: Include lower and upper bounds for all ABCD regions.
"""
config = {
    "Cluster": {
        "input_method": "CL",
        "xvar": "SUEP_S1_CL",
        "xvar_regions": [0.35, 0.4, 0.5, 1.0],
        "yvar": "SUEP_nconst_CL",
        "yvar_regions": [20, 40, 80, 1000],
        "SR": [["SUEP_S1_CL", ">=", 0.5], ["SUEP_nconst_CL", ">=", 80]],
        "selections": [["ht", ">", 1200], ["ntracks", ">", 0]],
    },
    "ClusterInverted": {
        "input_method": "CL",
        "xvar": "ISR_S1_CL",
        "xvar_regions": [0.35, 0.4, 0.5, 1.0],
        "yvar": "ISR_nconst_CL",
        "yvar_regions": [20, 40, 80, 1000],
        "SR": [["ISR_S1_CL", ">=", 0.5], ["ISR_nconst_CL", ">=", 80]],
        "selections": [["ht", ">", 1200], ["ntracks", ">", 0]],
    },
    "GNN": {
        "input_method": "GNN",
        "xvar": "SUEP_S1_GNN",
        "xvar_regions": [0.3, 0.4, 0.5, 1.0],
        "yvar": "single_l5_bPfcand_S1_SUEPtracks_GNN",
        "yvar_regions": [0.0, 0.5, 1.0],
        "SR": [
            ["SUEP_S1_GNN", ">=", 0.5],
            ["single_l5_bPfcand_S1_SUEPtracks_GNN", ">=", 0.5],
        ],
        "SR2": [
            ["SUEP_S1_CL", ">=", 0.5],
            ["SUEP_nconst_CL", ">=", 80],
        ],  # both are blinded
        "selections": [["ht", ">", 1200], ["ntracks", ">", 40]],
        "models": ["single_l5_bPfcand_S1_SUEPtracks"],
    },
    "GNNInverted": {
        "input_method": "GNNInverted",
        "xvar": "ISR_S1_GNNInverted",
        "xvar_regions": [0.0, 1.5, 2.0],
        "yvar": "single_l5_bPfcand_S1_SUEPtracks_GNNInverted",
        "yvar_regions": [0.0, 1.5, 2.0],
        "SR": [
            ["ISR_S1_GNNInverted", ">=", 10.0],
            ["single_l5_bPfcand_S1_SUEPtracks_GNNInverted", ">=", 10.0],
        ],
        #'SR2': [['ISR_S1_CL', '>=', 0.5], ['ISR_nconst_CL', '>=', 80]], # both are blinded
        "selections": [["ht", ">", 1200], ["ntracks", ">", 40]],
        "models": ["single_l5_bPfcand_S1_SUEPtracks"],
    },
    # 'ISRRemoval' : {
    #     'input_method' : 'IRM',
    #     'xvar' : 'SUEP_S1_IRM',
    #     'xvar_regions' : [0.35, 0.4, 0.5, 1.0],
    #     'yvar' : 'SUEP_nconst_IRM',
    #     'yvar_regions' : [10, 20, 40, 1000],
    #     'SR' : [['SUEP_S1_IRM', '>=', 0.5], ['SUEP_nconst_IRM', '>=', 40]],
    #     'selections' : [['ht', '>', 1200], ['ntracks','>', 0], ["SUEP_S1_IRM", ">=", 0.0]]
    # },
}

# output histos
def create_output_file(label, abcd, sys):

    # don't recreate histograms if called multiple times with the same output label
    if len(sys) > 0:
        label += "_" + sys
    if label in output["labels"]:
        return output
    else:
        output["labels"].append(label)

    # ABCD histogram
    xvar = abcd["xvar"]
    yvar = abcd["yvar"]
    xvar_regions = abcd["xvar_regions"]
    yvar_regions = abcd["yvar_regions"]
    output.update(
        {
            "ABCDvars_"
            + label: Hist.new.Reg(100, 0, yvar_regions[-1], name=xvar)
            .Reg(100, 0, xvar_regions[-1], name=yvar)
            .Weight()
        }
    )

    # defnie all the regions, will be used to make historgams for each region
    regions = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    n_regions = (len(xvar_regions) - 1) * (len(yvar_regions) - 1)
    regions_list = [""] + [regions[i] + "_" for i in range(n_regions)]

    ###########################################################################################################################
    # variables from the dataframe for all the events, and those in A, B, C regions
    for r in regions_list:
        output.update(
            {
                r
                + "ht_"
                + label: Hist.new.Reg(
                    100, 0, 10000, name=r + "ht_" + label, label="HT"
                ).Weight(),
                r
                + "ht_JEC_"
                + label: Hist.new.Reg(
                    100, 0, 10000, name=r + "ht_JEC_" + label, label="HT JEC"
                ).Weight(),
                r
                + "ht_JEC_JER_up_"
                + label: Hist.new.Reg(
                    100, 0, 10000, name=r + "ht_JEC_JER_up_" + label, label="HT JEC up"
                ).Weight(),
                r
                + "ht_JEC_JER_down_"
                + label: Hist.new.Reg(
                    100,
                    0,
                    10000,
                    name=r + "ht_JEC_JER_down_" + label,
                    label="HT JEC JER down",
                ).Weight(),
                r
                + "ht_JEC_JES_up_"
                + label: Hist.new.Reg(
                    100,
                    0,
                    10000,
                    name=r + "ht_JEC_JES_up_" + label,
                    label="HT JEC JES up",
                ).Weight(),
                r
                + "ht_JEC_JES_down_"
                + label: Hist.new.Reg(
                    100,
                    0,
                    10000,
                    name=r + "ht_JEC_JES_down_" + label,
                    label="HT JEC JES down",
                ).Weight(),
                r
                + "ntracks_"
                + label: Hist.new.Reg(
                    101, 0, 500, name=r + "ntracks_" + label, label="# Tracks in Event"
                ).Weight(),
                r
                + "ngood_fastjets_"
                + label: Hist.new.Reg(
                    9,
                    0,
                    10,
                    name=r + "ngood_fastjets_" + label,
                    label="# FastJets in Event",
                ).Weight(),
                r
                + "PV_npvs_"
                + label: Hist.new.Reg(
                    199, 0, 200, name=r + "PV_npvs_" + label, label="# PVs in Event "
                ).Weight(),
                r
                + "Pileup_nTrueInt_"
                + label: Hist.new.Reg(
                    199,
                    0,
                    200,
                    name=r + "Pileup_nTrueInt_" + label,
                    label="# True Interactions in Event ",
                ).Weight(),
                r
                + "ngood_ak4jets_"
                + label: Hist.new.Reg(
                    19,
                    0,
                    20,
                    name=r + "ngood_ak4jets_" + label,
                    label="# ak4jets in Event",
                ).Weight(),
                r
                + "ngood_tracker_ak4jets_"
                + label: Hist.new.Reg(
                    19,
                    0,
                    20,
                    name=r + "ngood_tracker_ak4jets_" + label,
                    label=r"# ak4jets in Event ($|\eta| < 2.4$)",
                ).Weight(),
                r
                + "FNR_"
                + label: Hist.new.Reg(
                    50,
                    0,
                    1,
                    name=r + "FNR_" + label,
                    label=r"# SUEP Tracks in ISR / # SUEP Tracks",
                ).Weight(),
                r
                + "ISR_contamination_"
                + label: Hist.new.Reg(
                    50,
                    0,
                    1,
                    name=r + "ISR_contamination_" + label,
                    label=r"# SUEP Tracks in ISR / # ISR Tracks",
                ).Weight(),
            }
        )

    ###########################################################################################################################
    if any([l in label for l in ["ISRRemoval", "Cluster", "Cone"]]):
        # 2D histograms
        output.update(
            {
                "2D_SUEP_S1_vs_ntracks_"
                + label: Hist.new.Reg(
                    100, 0, 1.0, name="SUEP_S1_" + label, label="$Sph_1$"
                )
                .Reg(100, 0, 500, name="ntracks_" + label, label="# Tracks")
                .Weight(),
                "2D_SUEP_S1_vs_SUEP_nconst_"
                + label: Hist.new.Reg(
                    100, 0, 1.0, name="SUEP_S1_" + label, label="$Sph_1$"
                )
                .Reg(200, 0, 500, name="nconst_" + label, label="# Constituents")
                .Weight(),
                "2D_SUEP_nconst_vs_SUEP_pt_avg_"
                + label: Hist.new.Reg(
                    200, 0, 500, name="SUEP_nconst_" + label, label="# Const"
                )
                .Reg(200, 0, 500, name="SUEP_pt_avg_" + label, label="$p_T Avg$")
                .Weight(),
                "2D_SUEP_nconst_vs_SUEP_pt_avg_b_"
                + label: Hist.new.Reg(
                    200, 0, 500, name="SUEP_nconst_" + label, label="# Const"
                )
                .Reg(
                    50,
                    0,
                    50,
                    name="SUEP_pt_avg_b_" + label,
                    label="$p_T Avg (Boosted frame)$",
                )
                .Weight(),
            }
        )

        # variables from the dataframe for all the events, and those in A, B, C regions
        for r in regions_list:
            output.update(
                {
                    r
                    + "SUEP_nconst_"
                    + label: Hist.new.Reg(
                        199,
                        0,
                        500,
                        name=r + "SUEP_nconst_" + label,
                        label="# Tracks in SUEP",
                    ).Weight(),
                    r
                    + "SUEP_pt_"
                    + label: Hist.new.Reg(
                        100,
                        0,
                        2000,
                        name=r + "SUEP_pt_" + label,
                        label=r"SUEP $p_T$ [GeV]",
                    ).Weight(),
                    r
                    + "SUEP_delta_pt_genPt_"
                    + label: Hist.new.Reg(
                        400,
                        -2000,
                        2000,
                        name=r + "SUEP_delta_pt_genPt_" + label,
                        label="SUEP $p_T$ - genSUEP $p_T$ [GeV]",
                    ).Weight(),
                    r
                    + "SUEP_pt_avg_"
                    + label: Hist.new.Reg(
                        200,
                        0,
                        500,
                        name=r + "SUEP_pt_avg_" + label,
                        label=r"SUEP Components $p_T$ Avg.",
                    ).Weight(),
                    r
                    + "SUEP_eta_"
                    + label: Hist.new.Reg(
                        100, -5, 5, name=r + "SUEP_eta_" + label, label=r"SUEP $\eta$"
                    ).Weight(),
                    r
                    + "SUEP_phi_"
                    + label: Hist.new.Reg(
                        100,
                        -6.5,
                        6.5,
                        name=r + "SUEP_phi_" + label,
                        label=r"SUEP $\phi$",
                    ).Weight(),
                    r
                    + "SUEP_mass_"
                    + label: Hist.new.Reg(
                        150,
                        0,
                        2000,
                        name=r + "SUEP_mass_" + label,
                        label="SUEP Mass [GeV]",
                    ).Weight(),
                    r
                    + "SUEP_delta_mass_genMass_"
                    + label: Hist.new.Reg(
                        400,
                        -2000,
                        2000,
                        name=r + "SUEP_delta_mass_genMass_" + label,
                        label="SUEP Mass - genSUEP Mass [GeV]",
                    ).Weight(),
                    r
                    + "SUEP_S1_"
                    + label: Hist.new.Reg(
                        100, 0, 1, name=r + "SUEP_S1_" + label, label="$Sph_1$"
                    ).Weight(),
                }
            )

    ###########################################################################################################################
    if "ClusterInverted" in label:
        output.update(
            {
                # 2D histograms
                "2D_ISR_S1_vs_ntracks_"
                + label: Hist.new.Reg(
                    100, 0, 1.0, name="ISR_S1_" + label, label="$Sph_1$"
                )
                .Reg(200, 0, 500, name="ntracks_" + label, label="# Tracks")
                .Weight(),
                "2D_ISR_S1_vs_ISR_nconst_"
                + label: Hist.new.Reg(
                    100, 0, 1.0, name="ISR_S1_" + label, label="$Sph_1$"
                )
                .Reg(200, 0, 500, name="nconst_" + label, label="# Constituents")
                .Weight(),
                "2D_ISR_nconst_vs_ISR_pt_avg_"
                + label: Hist.new.Reg(200, 0, 500, name="ISR_nconst_" + label)
                .Reg(500, 0, 500, name="ISR_pt_avg_" + label)
                .Weight(),
                "2D_ISR_nconst_vs_ISR_pt_avg_b_"
                + label: Hist.new.Reg(200, 0, 500, name="ISR_nconst_" + label)
                .Reg(100, 0, 100, name="ISR_pt_avg_" + label)
                .Weight(),
            }
        )
        # variables from the dataframe for all the events, and those in A, B, C regions
        for r in regions_list:
            output.update(
                {
                    r
                    + "ISR_nconst_"
                    + label: Hist.new.Reg(
                        199,
                        0,
                        500,
                        name=r + "ISR_nconst_" + label,
                        label="# Tracks in ISR",
                    ).Weight(),
                    r
                    + "ISR_pt_"
                    + label: Hist.new.Reg(
                        100,
                        0,
                        2000,
                        name=r + "ISR_pt_" + label,
                        label=r"ISR $p_T$ [GeV]",
                    ).Weight(),
                    r
                    + "ISR_pt_avg_"
                    + label: Hist.new.Reg(
                        500,
                        0,
                        500,
                        name=r + "ISR_pt_avg_" + label,
                        label=r"ISR Components $p_T$ Avg.",
                    ).Weight(),
                    r
                    + "ISR_eta_"
                    + label: Hist.new.Reg(
                        100, -5, 5, name=r + "ISR_eta_" + label, label=r"ISR $\eta$"
                    ).Weight(),
                    r
                    + "ISR_phi_"
                    + label: Hist.new.Reg(
                        100, -6.5, 6.5, name=r + "ISR_phi_" + label, label=r"ISR $\phi$"
                    ).Weight(),
                    r
                    + "ISR_mass_"
                    + label: Hist.new.Reg(
                        150,
                        0,
                        4000,
                        name=r + "ISR_mass_" + label,
                        label="ISR Mass [GeV]",
                    ).Weight(),
                    r
                    + "ISR_S1_"
                    + label: Hist.new.Reg(
                        100, 0, 1, name=r + "ISR_S1_" + label, label="$Sph_1$"
                    ).Weight(),
                }
            )

    ###########################################################################################################################
    if label == "GNN" and options.doInf:

        # 2D histograms
        for model in abcd["models"]:
            output.update(
                {
                    "2D_SUEP_S1_vs_"
                    + model
                    + "_"
                    + label: Hist.new.Reg(
                        100, 0, 1.0, name="SUEP_S1_" + label, label="$Sph_1$"
                    )
                    .Reg(100, 0, 1, name=model + "_" + label, label="GNN Output")
                    .Weight(),
                    "2D_SUEP_nconst_vs_"
                    + model
                    + "_"
                    + label: Hist.new.Reg(
                        200, 0, 500, name="SUEP_nconst_" + label, label="# Const"
                    )
                    .Reg(100, 0, 1, name=model + "_" + label, label="GNN Output")
                    .Weight(),
                }
            )

        output.update(
            {
                "2D_SUEP_nconst_vs_SUEP_S1_"
                + label: Hist.new.Reg(
                    200, 0, 500, name="SUEP_nconst_" + label, label="# Const"
                )
                .Reg(100, 0, 1, name="SUEP_S1_" + label, label="$Sph_1$")
                .Weight(),
            }
        )

        for r in regions_list:
            output.update(
                {
                    r
                    + "SUEP_nconst_"
                    + label: Hist.new.Reg(
                        199,
                        0,
                        500,
                        name=r + "SUEP_nconst" + label,
                        label="# Tracks in SUEP",
                    ).Weight(),
                    r
                    + "SUEP_S1_"
                    + label: Hist.new.Reg(
                        100, -1, 2, name=r + "SUEP_S1_" + label, label="$Sph_1$"
                    ).Weight(),
                }
            )
            for model in abcd["models"]:
                output.update(
                    {
                        r
                        + model
                        + "_"
                        + label: Hist.new.Reg(
                            100, 0, 1, name=r + model + "_" + label, label="GNN Output"
                        ).Weight()
                    }
                )

    ###########################################################################################################################
    if label == "GNNInverted" and options.doInf:

        # 2D histograms
        for model in abcd["models"]:
            output.update(
                {
                    "2D_ISR_S1_vs_"
                    + model
                    + "_"
                    + label: Hist.new.Reg(
                        100, 0, 1.0, name="ISR_S1_" + label, label="$Sph_1$"
                    )
                    .Reg(100, 0, 1, name=model + "_" + label, label="GNN Output")
                    .Weight(),
                    "2D_ISR_nconst_vs_"
                    + model
                    + "_"
                    + label: Hist.new.Reg(
                        200, 0, 500, name="ISR_nconst_" + label, label="# Const"
                    )
                    .Reg(100, 0, 1, name=model + "_" + label, label="GNN Output")
                    .Weight(),
                }
            )
        output.update(
            {
                "2D_ISR_nconst_vs_ISR_S1_"
                + label: Hist.new.Reg(
                    200, 0, 500, name="ISR_nconst_" + label, label="# Const"
                )
                .Reg(100, 0, 1, name="ISR_S1_" + label, label="$Sph_1$")
                .Weight()
            }
        )

        for r in regions_list:
            output.update(
                {
                    r
                    + "ISR_nconst_"
                    + label: Hist.new.Reg(
                        199,
                        0,
                        500,
                        name=r + "ISR_nconst" + label,
                        label="# Tracks in ISR",
                    ).Weight(),
                    r
                    + "ISR_S1_"
                    + label: Hist.new.Reg(
                        100, -1, 2, name=r + "ISR_S1_" + label, label="$Sph_1$"
                    ).Weight(),
                }
            )
            for model in abcd["models"]:
                output.update(
                    {
                        r
                        + model
                        + "_"
                        + label: Hist.new.Reg(
                            100, 0, 1, name=r + model + "_" + label, label="GNN Output"
                        ).Weight()
                    }
                )

    ###########################################################################################################################

    return output


#############################################################################################################


def plotter(df, output, abcd, label_out, sys, blind=True, isMC=False):
    """
    INPUTS:
        df: input file DataFrame.
        output: dictionary of histograms to be filled.
        abcd: definitions of ABCD regions, signal region, event selections.
        label_out: label associated with the output (e.g. "ISRRemoval"), as keys in
                   the config dictionary.

    OUTPUTS:
        output: dict, now with updated histograms.

    EXPLANATION:
    The DataFrame generated by ../workflows/SUEP_coffea.py has the form:
    event variables (ht, ...)   CL vars (SUEP_S1_CL, ...)  ML vars  Other Methods
          0                                 0                   0          ...
          1                                 NaN                 1          ...
          2                                 NaN                 NaN        ...
          3                                 1                   2          ...
    (The event vars are always filled, while the vars for each method are filled only
    if the event passes the method's selections, hence the NaNs).

    This function will plot, for each 'label_out':
        1. All event variables, e.g. ht_label_out
        2. All columns from 'input_method', e.g. SUEP_S1_IRM column will be
           plotted to histogram SUEP_S1_ISRRemoval.
        3. 2D variables are automatically plotted, as long as hstogram is
           initialized in the output dict as "2D_var1_vs_var2"

    N.B.: Histograms are filled only if they are initialized in the output dictionary.

    e.g. We want to plot CL.
    Event Selection:
        1. Grab only events that don't have NaN for CL variables.
        2. Blind for data! Use SR to define signal regions and cut it out of df.
        3. Apply selections as defined in the 'selections' in the dict.

    Fill Histograms:
        1. Plot variables from the DataFrame.
           1a. Event wide variables
           1b. Cluster method (CL) variables
        2. Plot 2D variables.
        3. Plot variables from the different ABCD regions as defined in the abcd dict.
           3a. Event wide variables
           3b. Cluster method (CL) variables
    """

    input_method = abcd["input_method"]
    if len(sys) > 0:
        label_out = label_out + "_" + sys

    # 1. keep only events that passed this method
    df = df[~df[abcd["xvar"]].isnull()]

    # 2. blind
    if blind and not isMC:
        SR = abcd["SR"]
        if len(SR) != 2:
            sys.exit(
                label_out
                + ": Make sure you have correctly defined your signal region. Exiting."
            )
        df = df.loc[
            ~(
                make_selection(df, SR[0][0], SR[0][1], SR[0][2], apply=False)
                & make_selection(df, SR[1][0], SR[1][1], SR[1][2], apply=False)
            )
        ]

        if "SR2" in abcd.keys():
            SR2 = abcd["SR2"]
            if len(SR2) != 2:
                sys.exit(
                    label_out
                    + ": Make sure you have correctly defined your signal region. Exiting."
                )
            df = df.loc[
                ~(
                    make_selection(df, SR2[0][0], SR2[0][1], SR2[0][2], apply=False)
                    & make_selection(df, SR2[1][0], SR2[1][1], SR2[1][2], apply=False)
                )
            ]

    # 3. apply selections
    for sel in abcd["selections"]:
        df = make_selection(df, sel[0], sel[1], sel[2], apply=True)

    # auto fill all histograms in the output dictionary
    auto_fill(df, output, abcd, label_out, isMC=isMC, do_abcd=True)

    return output


#############################################################################################################

# get list of files
username = getpass.getuser()
if options.file:
    files = [options.file]
elif options.xrootd:
    dataDir = dataDirXRootD
    if options.merged:
        dataDir += "merged/"
    result = subprocess.check_output(["xrdfs", redirector, "ls", dataDir])
    result = result.decode("utf-8")
    files = result.split("\n")
    files = [f for f in files if len(f) > 0]
else:
    dataDir = dataDirLocal
    if options.merged:
        dataDir += "merged/"
    files = [dataDir + f for f in os.listdir(dataDir)]

# get cross section
xsection = 1.0
if options.isMC:
    xsection = getXSection(options.dataset, options.era, SUEP=False)

# event weights
puweights, puweights_up, puweights_down = pileup_weight.pileup_weight(options.era)
trig_bins, trig_weights, trig_weights_up, trig_weights_down = triggerSF.triggerSF(
    options.era
)
# Higgs pT reweighting is implemented later when df is defined

# custom per region weights
scaling_weights = None
if options.weights != "None":
    w = np.load(options.weights, allow_pickle=True)
    scaling_weights = defaultdict(lambda: np.zeros(2))
    scaling_weights.update(w.item())

# fill ABCD hists with dfs from hdf5 files
nfailed = 0
total_gensumweight = 0

if options.dataset:
    outFile = outDir + "/" + options.dataset + "_" + options.output
else:
    outFile = os.path.join(outDir, options.output)
output = {"labels": []}

# systematics
if options.isMC and options.doSyst:

    new_config = {}

    # track systematics
    # we need to use the track_down version of the data,
    # which has the randomly deleted tracks (see SUEPCoffea.py)
    # so we need to modify the config to use the _track_down vars
    for label_out, config_out in config.items():
        label_out_new = label_out + "_track_down"
        new_config[label_out_new] = deepcopy(config[label_out])
        new_config[label_out_new]["input_method"] += "_track_down"
        new_config[label_out_new]["xvar"] += "_track_down"
        new_config[label_out_new]["yvar"] += "_track_down"
        for iSel in range(len(new_config[label_out_new]["SR"])):
            new_config[label_out_new]["SR"][iSel][0] += "_track_down"
        for iSel in range(len(new_config[label_out_new]["selections"])):
            if new_config[label_out_new]["selections"][iSel][0] in [
                "ht",
                "ngood_ak4jets",
            ]:
                continue
            new_config[label_out_new]["selections"][iSel][0] += "_track_down"

    # jet systematics
    # here, we just change ht to ht_SYS (e.g. ht -> ht_JEC_JES_up)
    jet_corrections = [
        "JEC",
        "JEC_JER_up",
        "JEC_JER_down",
        "JEC_JES_up",
        "JEC_JES_down",
    ]
    for sys in jet_corrections:
        for label_out, config_out in config.items():
            label_out_new = label_out + "_" + sys
            new_config[label_out_new] = deepcopy(config[label_out])
            for iSel in range(len(new_config[label_out_new]["selections"])):
                if "ht" == new_config[label_out_new]["selections"][iSel][0]:
                    new_config[label_out_new]["selections"][iSel][0] += "_" + sys

    config = new_config | config

### Plotting loop #######################################################################
for ifile in tqdm(files):

    #####################################################################################
    # ---- Load file
    #####################################################################################

    if options.xrootd:
        if os.path.exists(options.dataset + ".hdf5"):
            os.system("rm " + options.dataset + ".hdf5")
        xrd_file = redirector + ifile
        os.system(f"xrdcp -s {xrd_file} {options.dataset}.hdf5")
        df, metadata = h5load(options.dataset + ".hdf5", "vars")
    else:
        df, metadata = h5load(ifile, "vars")

    # check if file is corrupted
    if type(df) == int:
        nfailed += 1
        continue

    # update the gensumweight
    if options.isMC and metadata != 0:
        total_gensumweight += metadata["gensumweight"]

    # check if file is empty
    if "empty" in list(df.keys()):
        continue
    if df.shape[0] == 0:
        continue

    #####################################################################################
    # ---- Additional weights
    # Currently applies pileup weights through nTrueInt
    # and optionally (options.weights) scaling weights that are derived to force
    # MC to agree with data in one variable. Usage:
    # df['event_weight'] *= another event weight, etc
    # ---- Make plots
    #####################################################################################
    event_weight = np.ones(df.shape[0])
    (
        higgs_bins,
        higgs_weights,
        higgs_weights_up,
        higgs_weights_down,
    ) = higgs_reweight.higgs_reweight(df["SUEP_genPt"])
    if options.doSyst:
        sys_loop = [
            "",
            "puweights_up",
            "puweights_down",
            "trigSF_up",
            "trigSF_down",
            "PSWeight_ISR_up",
            "PSWeight_ISR_down",
            "PSWeight_FSR_up",
            "PSWeight_FSR_down",
            "higgs_weights_up",
            "higgs_weights_down",
        ]
    else:
        sys_loop = [""]
    for sys in sys_loop:
        # prepare new event weight
        df["event_weight"] = event_weight

        # 1) pileup weights
        if options.isMC == 1 and options.scouting != 1:
            Pileup_nTrueInt = np.array(df["Pileup_nTrueInt"]).astype(int)
            if "puweights_up" in sys:
                pu = puweights_up[Pileup_nTrueInt]
            elif "puweights_down" in sys:
                pu = puweights_down[Pileup_nTrueInt]
            else:
                pu = puweights[Pileup_nTrueInt]
            df["event_weight"] *= pu

        # 2) TriggerSF weights
        if options.isMC == 1 and options.scouting != 1:
            ht = np.array(df["ht"]).astype(int)
            ht_bin = np.digitize(ht, trig_bins) - 1  # digitize the values to bins
            ht_bin = np.clip(ht_bin, 0, 49)  # Set overflow to last SF
            if "trigSF_up" in sys:
                trigSF = trig_weights_up[ht_bin]
            elif "trigSF_down" in sys:
                trigSF = trig_weights_down[ht_bin]
            else:
                trigSF = trig_weights[ht_bin]
            df["event_weight"] *= trigSF

        # 3) PS weights
        if options.isMC == 1 and options.scouting != 1 and ("PSWeight" in sys):
            if sys in df.keys():
                df["event_weight"] *= df[sys]

        # 4) Higgs_pt weights
        if options.isMC == 1 and "SUEP-m125" in options.dataset:
            gen_pt = np.array(df["SUEP_genPt"]).astype(int)
            gen_bin = np.digitize(gen_pt, higgs_bins) - 1
            if "higgs_weights_up" in sys:
                higgs_weight = higgs_weights_up[gen_bin]
            elif "higgs_weights_down" in sys:
                higgs_weight = higgs_weights_up[gen_bin]
            else:
                higgs_weight = higgs_weights[gen_bin]
            df["event_weight"] *= higgs_weight

        # 5) scaling weights
        # N.B.: these aren't part of the systematics, just an optional scaling
        if options.isMC == 1 and scaling_weights is not None:
            df = apply_scaling_weights(
                df.copy(),
                scaling_weights,
                config["Cluster"]["x_var_regions"],
                config["Cluster"]["x_var_regions"],
                regions="ABCDEFGHI",
                x_var="SUEP_S1_CL",
                y_var="SUEP_nconst_CL",
                z_var="ht",
            )

        for label_out, config_out in config.items():
            if "track_down" in label_out and sys != "":
                continue
            if options.isMC and sys != "":
                if any([j in label_out for j in jet_corrections]):
                    continue
            output.update(create_output_file(label_out, config_out, sys))
            output = plotter(
                df.copy(),
                output,
                config_out,
                label_out,
                sys,
                isMC=options.isMC,
                blind=options.blind,
            )

    #####################################################################################
    # ---- End
    #####################################################################################

    # remove file at the end of loop
    if options.xrootd:
        os.system("rm " + options.dataset + ".hdf5")

print("Number of files that failed to be read:", nfailed)
### End plotting loop ###################################################################

# do the tracks UP systematic
if options.doSyst:
    sys = "track_up"
    for label_out, config_out in config.items():
        if "track_down" in label_out:
            continue

        new_output = {}
        for hist_name in output.keys():
            if not hist_name.endswith("_track_down"):
                continue
            hDown = output[hist_name].copy()
            hNom = output[hist_name.replace("_track_down", "")].copy()
            hUp = get_tracks_up(hNom, hDown)
            new_output.update({hist_name.replace("_track_down", "_track_up"): hUp})
        output = new_output | output

# do the GNN systematic
if options.doSyst and options.doInf:

    # load in the json file containing the corrections for each year/model
    fGNNsyst = "../data/GNN/GNNsyst.json"
    with open(fGNNsyst) as f:
        GNNsyst = json.load(f)

    # complex numbers for hist
    bins = [0.0j, 0.25j, 0.5j, 0.75j, 1.0j]
    for model in config["GNN"]["models"]:

        # load the correct model for each year
        yearSyst = GNNsyst.get(str(options.era))
        if yearSyst is None:
            logging.warning(
                "--- {} was not found in file {}; systematic has not been applied".format(
                    options.era, fGNNsyst
                )
            )
            continue
        scales = yearSyst.get(model)
        if scales is None:
            logging.warning(
                "--- {} was not found in file {}; systematic has not been applied".format(
                    model, fGNNsyst
                )
            )
            continue

        # scale them
        GNN_syst_plots = {}
        for plot in output.keys():
            # apply only to GNN
            if not plot.endswith("GNN"):
                continue

            if model in plot and "2D" not in plot:
                GNN_syst_plots[plot + "_GNN_down_GNN"] = apply_binwise_scaling(
                    output[plot].copy(), bins, [1 - s for s in scales]
                )
                GNN_syst_plots[plot + "_GNN_up_GNN"] = apply_binwise_scaling(
                    output[plot].copy(), bins, [1 + s for s in scales]
                )
            if model in plot and "2D" in plot:
                var1 = plot.split("_vs_")[0]
                var2 = plot.split("_vs_")[1]
                if model in var1:
                    dim = "x"
                elif model in var2:
                    dim = "y"
                GNN_syst_plots[plot + "_GNN_down_GNN"] = apply_binwise_scaling(
                    output[plot].copy(), bins, [1 - s for s in scales], dim=dim
                )
                GNN_syst_plots[plot + "_GNN_up_GNN"] = apply_binwise_scaling(
                    output[plot].copy(), bins, [1 + s for s in scales], dim=dim
                )
        output.update(GNN_syst_plots)

# apply normalization
output.pop("labels")
if options.isMC:
    if total_gensumweight > 0.0:
        for plot in list(output.keys()):
            output[plot] = output[plot] * xsection / total_gensumweight
    else:
        print("Total gensumweight is 0")

# Save to pickle
pickle.dump(output, open(outFile + ".pkl", "wb"))

# save to root
with uproot.recreate(outFile + ".root") as froot:
    for h, hist in output.items():
        froot[h] = hist
