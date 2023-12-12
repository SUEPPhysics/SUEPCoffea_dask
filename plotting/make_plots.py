# Make plots for SUEP analysis. Reads in hdf5 files and outputs to root files
import argparse
import getpass
import logging
import os
import subprocess

import fill_utils
import numpy as np
import plot_utils
import uproot

# Import our own functions
from CMS_corrections import (
    GNN_syst,
    higgs_reweight,
    pileup_weight,
    track_killing,
    triggerSF,
)
from hist import Hist
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)

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
    "-s",
    "--save",
    type=str,
    help="Use specific output directory. Overrides MIT-specific paths.",
    required=False,
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
parser.add_argument("-e", "--era", type=str, help="era", required=True)
parser.add_argument("--isMC", type=int, help="Is this MC or data", required=True)
parser.add_argument(
    "--isSignal", type=int, help="Is this signal sample or not", default=0
)
parser.add_argument(
    "--channel", type=str, help="Analysis channel: ggF, WH", required=True
)
parser.add_argument("--scouting", type=int, default=0, help="Is this scouting or no")
# some parameters you can toggle freely
parser.add_argument("--doInf", type=int, default=0, help="make GNN plots")
parser.add_argument(
    "--doABCD", type=int, default=0, help="make plots for each ABCD+ region"
)
parser.add_argument("--doSyst", type=int, default=0, help="make systematic plots")
parser.add_argument(
    "--predictSR", type=int, default=0, help="Predict SR using ABCD method."
)
parser.add_argument(
    "--blind", type=int, default=1, help="Blind the data (default=True)"
)
parser.add_argument(
    "--weights",
    default="None",
    help="Pass the filename of the weights, e.g. --weights weights.npy",
)
options = parser.parse_args()

###################################################################################################################
# Script Parameters
###################################################################################################################

outDir = f"/data/submit/{getpass.getuser()}/SUEP/outputs/"
if options.save is not None and options.save != "None" and options != "none":
    outDir = options.save
redirector = "root://submit50.mit.edu/"
username = getpass.getuser()
if os.path.isdir("/data/submit/cms/store/user/" + username):
    # define these if --xrootd 0
    dataDirLocal = "/data/submit//cms/store/user/{}/SUEP/{}/{}/".format(
        username, options.tag, options.dataset
    )
    # and these if --xrootd 1
    dataDirXRootD = "/cms/store/user/{}/SUEP/{}/{}/".format(
        username, options.tag, options.dataset
    )
elif os.path.isdir("/data/submit/" + username):
    # define these if --xrootd 0
    dataDirLocal = "/data/submit/{}/SUEP/{}/{}/".format(
        username, options.tag, options.dataset
    )
    # and these if --xrootd 1
    dataDirXRootD = f"/{username}/SUEP/{options.tag}/{options.dataset}/"
"""
Define output plotting methods, each draws from an input_method (outputs of SUEPCoffea),
and can have its own selections, ABCD regions, and signal region.
Multiple plotting methods can be defined for the same input method, as different
selections and ABCD methods can be applied.
N.B.: Include lower and upper bounds for all ABCD regions.
"""
if options.channel == "WH":
    config = {
        "TopPT": {
            "input_method": "TopPT",
            "selections": [],
        },
    }
if options.channel == "ggF":
    if options.scouting:
        config = {
            "Cluster": {
                "input_method": "CL",
                "xvar": "SUEP_S1_CL",
                "xvar_regions": [0.3, 0.34, 0.5, 2.0],
                "yvar": "SUEP_nconst_CL",
                "yvar_regions": [0, 18, 50, 1000],
                "SR": [["SUEP_S1_CL", ">=", 0.5], ["SUEP_nconst_CL", ">=", 70]],
                "selections": [["ht_JEC", ">", 560], ["ntracks", ">", 0]],
            },
            "ClusterInverted": {
                "input_method": "CL",
                "xvar": "ISR_S1_CL",
                "xvar_regions": [0.3, 0.34, 0.5, 2.0],
                "yvar": "ISR_nconst_CL",
                "yvar_regions": [0, 18, 35, 1000],
                "SR": [["SUEP_S1_CL", ">=", 0.5], ["SUEP_nconst_CL", ">=", 50]],
                "selections": [["ht_JEC", ">", 560], ["ntracks", ">", 0]],
            },
        }
    else:
        config = {
            "Cluster70": {
                "input_method": "CL",
                "xvar": "SUEP_S1_CL",
                "xvar_regions": [0.3, 0.4, 0.5, 2.0],
                "yvar": "SUEP_nconst_CL",
                "yvar_regions": [30, 50, 70, 1000],
                "SR": [["SUEP_S1_CL", ">=", 0.5], ["SUEP_nconst_CL", ">=", 70]],
                "selections": [["ht_JEC", ">", 1200], ["ntracks", ">", 0]],
            },
            "ClusterInverted": {
                "input_method": "CL",
                "xvar": "ISR_S1_CL",
                "xvar_regions": [0.3, 0.4, 0.5, 2.0],
                "yvar": "ISR_nconst_CL",
                "yvar_regions": [30, 50, 70, 1000],
                # "SR": [["SUEP_S1_CL", ">=", 0.5], ["SUEP_nconst_CL", ">=", 75]],
                "SR": [["SUEP_S1_CL", ">=", 0.5], ["SUEP_nconst_CL", ">=", 70]],
                "selections": [["ht_JEC", ">", 1200], ["ntracks", ">", 0]],
            },
        }

    if options.doInf:
        config.update(
            {
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
                    "selections": [["ht_JEC", ">", 1200], ["ntracks", ">", 40]],
                    "models": ["single_l5_bPfcand_S1_SUEPtracks"],
                    "fGNNsyst": "../data/GNN/GNNsyst.json",
                    "GNNsyst_bins": [0.0j, 0.25j, 0.5j, 0.75j, 1.0j],
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
                    # 'SR2': [['ISR_S1_CL', '>=', 0.5], ['ISR_nconst_CL', '>=', 80]], # both are blinded
                    "selections": [["ht_JEC", ">", 1200], ["ntracks", ">", 40]],
                    "models": ["single_l5_bPfcand_S1_SUEPtracks"],
                },
            }
        )


def open_file(options, redirector, ifile):
    if not options.xrootd:
        return fill_utils.h5load(ifile, "vars")
    if os.path.exists(options.dataset + ".hdf5"):
        os.system("rm " + options.dataset + ".hdf5")
    xrd_file = redirector + ifile
    os.system(f"xrdcp -s {xrd_file} {options.dataset}.hdf5")
    return fill_utils.h5load(options.dataset + ".hdf5", "vars")


def create_output_clusterInverted(output, label, regions_list):
    output.update(
        {
            # 2D histograms
            f"2D_ISR_S1_vs_ntracks_{label}": Hist.new.Reg(
                100, 0, 1.0, name=f"ISR_S1_{label}", label="$Sph_1$"
            )
            .Reg(200, 0, 500, name=f"ntracks_{label}", label="# Tracks")
            .Weight(),
            f"2D_ISR_S1_vs_ISR_nconst_{label}": Hist.new.Reg(
                100, 0, 1.0, name=f"ISR_S1_{label}", label="$Sph_1$"
            )
            .Reg(501, 0, 500, name=f"nconst_{label}", label="# Constituents")
            .Weight(),
            f"2D_ISR_nconst_vs_ISR_pt_avg_{label}": Hist.new.Reg(
                501, 0, 500, name=f"ISR_nconst_{label}"
            )
            .Reg(500, 0, 500, name=f"ISR_pt_avg_{label}")
            .Weight(),
        }
    )
    # variables from the dataframe for all the events, and those in A, B, C regions
    for r in regions_list:
        output.update(
            {
                f"{r}ISR_nconst_{label}": Hist.new.Reg(
                    501,
                    0,
                    500,
                    name=f"{r}ISR_nconst_{label}",
                    label="# Tracks in ISR",
                ).Weight(),
                f"{r}ISR_S1_{label}": Hist.new.Reg(
                    100, 0, 1, name=f"{r}ISR_S1_{label}", label="$Sph_1$"
                ).Weight(),
            }
        )
        if (
            "up" not in label and "down" not in label and r == ""
        ):  # don't care for systematics for these
            output.update(
                {
                    f"ISR_pt_{label}": Hist.new.Reg(
                        100,
                        0,
                        2000,
                        name=f"ISR_pt_{label}",
                        label=r"ISR $p_T$ [GeV]",
                    ).Weight(),
                    f"ISR_pt_avg_{label}": Hist.new.Reg(
                        500,
                        0,
                        500,
                        name=f"ISR_pt_avg_{label}",
                        label=r"ISR Components $p_T$ Avg.",
                    ).Weight(),
                    f"ISR_eta_{label}": Hist.new.Reg(
                        100,
                        -5,
                        5,
                        name=f"ISR_eta_{label}",
                        label=r"ISR $\eta$",
                    ).Weight(),
                    f"ISR_phi_{label}": Hist.new.Reg(
                        100,
                        -6.5,
                        6.5,
                        name=f"ISR_phi_{label}",
                        label=r"ISR $\phi$",
                    ).Weight(),
                    f"ISR_mass_{label}": Hist.new.Reg(
                        150,
                        0,
                        4000,
                        name=f"ISR_mass_{label}",
                        label="ISR Mass [GeV]",
                    ).Weight(),
                }
            )


def create_output_GNN(abcd, output, label, regions_list):
    # 2D histograms
    for model in abcd["models"]:
        output.update(
            {
                f"2D_SUEP_S1_vs_{model}_{label}": Hist.new.Reg(
                    100, 0, 1.0, name=f"SUEP_S1_{label}", label="$Sph_1$"
                )
                .Reg(100, 0, 1, name=f"{model}_{label}", label="GNN Output")
                .Weight(),
                f"2D_SUEP_nconst_vs_{model}_{label}": Hist.new.Reg(
                    501,
                    0,
                    500,
                    name=f"SUEP_nconst_{label}",
                    label="# Const",
                )
                .Reg(100, 0, 1, name=f"{model}_{label}", label="GNN Output")
                .Weight(),
            }
        )

    output.update(
        {
            f"2D_SUEP_nconst_vs_SUEP_S1_{label}": Hist.new.Reg(
                501, 0, 500, name=f"SUEP_nconst_{label}", label="# Const"
            )
            .Reg(100, 0, 1, name=f"SUEP_S1_{label}", label="$Sph_1$")
            .Weight(),
        }
    )

    for r in regions_list:
        output.update(
            {
                f"{r}SUEP_nconst_{label}": Hist.new.Reg(
                    501,
                    0,
                    500,
                    name=f"{r}SUEP_nconst{label}",
                    label="# Constituents",
                ).Weight(),
                f"{r}SUEP_S1_{label}": Hist.new.Reg(
                    100,
                    -1,
                    2,
                    name=f"{r}SUEP_S1_{label}",
                    label="$Sph_1$",
                ).Weight(),
            }
        )
        for model in abcd["models"]:
            output.update(
                {
                    f"{r}{model}_{label}": Hist.new.Reg(
                        100,
                        0,
                        1,
                        name=f"{r}{model}_{label}",
                        label="GNN Output",
                    ).Weight()
                }
            )


def create_output_GNNInverted(abcd, output, label, regions_list):
    # 2D histograms
    for model in abcd["models"]:
        output.update(
            {
                f"2D_ISR_S1_vs_{model}_{label}": Hist.new.Reg(
                    100, 0, 1.0, name=f"ISR_S1_{label}", label="$Sph_1$"
                )
                .Reg(100, 0, 1, name=f"{model}_{label}", label="GNN Output")
                .Weight(),
                f"2D_ISR_nconst_vs_{model}_{label}": Hist.new.Reg(
                    501, 0, 500, name=f"ISR_nconst_{label}", label="# Const"
                )
                .Reg(100, 0, 1, name=f"{model}_{label}", label="GNN Output")
                .Weight(),
            }
        )
    output.update(
        {
            f"2D_ISR_nconst_vs_ISR_S1_{label}": Hist.new.Reg(
                501, 0, 500, name=f"ISR_nconst_{label}", label="# Const"
            )
            .Reg(100, 0, 1, name=f"ISR_S1_{label}", label="$Sph_1$")
            .Weight()
        }
    )

    for r in regions_list:
        output.update(
            {
                f"{r}ISR_nconst_{label}": Hist.new.Reg(
                    501,
                    0,
                    500,
                    name=f"{r}ISR_nconst{label}",
                    label="# Tracks in ISR",
                ).Weight(),
                f"{r}ISR_S1_{label}": Hist.new.Reg(
                    100, -1, 2, name=f"{r}ISR_S1_{label}", label="$Sph_1$"
                ).Weight(),
            }
        )
        for model in abcd["models"]:
            output.update(
                {
                    f"{r}{model}_{label}": Hist.new.Reg(
                        100,
                        0,
                        1,
                        name=f"{r}{model}_{label}",
                        label="GNN Output",
                    ).Weight()
                }
            )


# output histos
def create_output_file(label, abcd, options):
    # don't recreate histograms if called multiple times with the same output label
    if label in output["labels"]:
        return output
    else:
        output["labels"].append(label)

    if options.doABCD:
        # ABCD histogram
        xvar = abcd["xvar"]
        yvar = abcd["yvar"]
        xvar_regions = abcd["xvar_regions"]
        yvar_regions = abcd["yvar_regions"]
        output.update(
            {
                f"ABCDvars_{label}": Hist.new.Reg(
                    100, yvar_regions[0], yvar_regions[-1], name=xvar
                )
                .Reg(100, xvar_regions[0], xvar_regions[-1], name=yvar)
                .Weight()
            }
        )

    # define all the regions, will be used to make historgams for each region
    regions_list = [""]
    if options.doABCD:
        regions = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        n_regions = (len(xvar_regions) - 1) * (len(yvar_regions) - 1)
        regions_list += [regions[i] + "_" for i in range(n_regions)]

    ###########################################################################################################################
    # variables from the dataframe for all the events, and those in A, B, C regions
    output.update(
        {
            f"ht_{label}": Hist.new.Reg(
                100, 0, 10000, name=f"ht_{label}", label="HT"
            ).Weight(),
            f"ht_JEC_{label}": Hist.new.Reg(
                100, 0, 10000, name=f"ht_JEC_{label}", label="HT JEC"
            ).Weight(),
            f"ht_JEC_JER_up_{label}": Hist.new.Reg(
                100,
                0,
                10000,
                name=f"ht_JEC_JER_up_{label}",
                label="HT JEC up",
            ).Weight(),
            f"ht_JEC_JER_down_{label}": Hist.new.Reg(
                100,
                0,
                10000,
                name=f"ht_JEC_JER_down_{label}",
                label="HT JEC JER down",
            ).Weight(),
            f"ht_JEC_JES_up_{label}": Hist.new.Reg(
                100,
                0,
                10000,
                name=f"ht_JEC_JES_up_{label}",
                label="HT JEC JES up",
            ).Weight(),
            f"ht_JEC_JES_down_{label}": Hist.new.Reg(
                100,
                0,
                10000,
                name=f"ht_JEC_JES_down_{label}",
                label="HT JEC JES down",
            ).Weight(),
            f"ntracks_{label}": Hist.new.Reg(
                101,
                0,
                500,
                name=f"ntracks_{label}",
                label="# Tracks in Event",
            ).Weight(),
            f"ngood_fastjets_{label}": Hist.new.Reg(
                9,
                0,
                10,
                name=f"ngood_fastjets_{label}",
                label="# FastJets in Event",
            ).Weight(),
            f"PV_npvs_{label}": Hist.new.Reg(
                199,
                0,
                200,
                name=f"PV_npvs_{label}",
                label="# PVs in Event ",
            ).Weight(),
            f"Pileup_nTrueInt_{label}": Hist.new.Reg(
                199,
                0,
                200,
                name=f"Pileup_nTrueInt_{label}",
                label="# True Interactions in Event ",
            ).Weight(),
            f"ngood_ak4jets_{label}": Hist.new.Reg(
                19,
                0,
                20,
                name=f"ngood_ak4jets_{label}",
                label="# ak4jets in Event",
            ).Weight(),
            f"CaloMET_pt_{label}": Hist.new.Reg(
                100,
                0,
                1000,
                name=f"CaloMET_pt_{label}",
                label="CaloMET pT",
            ).Weight(),
            f"CaloMET_phi_{label}": Hist.new.Reg(
                100,
                -4,
                4,
                name=f"CaloMET_phi_{label}",
                label="CaloMET phi",
            ).Weight(),
            f"CaloMET_sumEt_{label}": Hist.new.Reg(
                100,
                0,
                5000,
                name=f"CaloMET_sumEt_{label}",
                label="CaloMET sumEt",
            ).Weight(),
            f"ChsMET_pt_{label}": Hist.new.Reg(
                100,
                0,
                1000,
                name=f"ChsMET_pt_{label}",
                label="ChsMET pT",
            ).Weight(),
            f"ChsMET_phi_{label}": Hist.new.Reg(
                100,
                -4,
                4,
                name=f"ChsMET_phi_{label}",
                label="ChsMET phi",
            ).Weight(),
            f"ChsMET_sumEt_{label}": Hist.new.Reg(
                100,
                0,
                5000,
                name=f"ChsMET_sumEt_{label}",
                label="ChsMET sumEt",
            ).Weight(),
            f"TkMET_pt_{label}": Hist.new.Reg(
                100,
                0,
                1000,
                name=f"TkMET_pt_{label}",
                label="TkMET pt",
            ).Weight(),
            f"TkMET_phi_{label}": Hist.new.Reg(
                100,
                -4,
                4,
                name=f"TkMET_phi_{label}",
                label="TkMET phi",
            ).Weight(),
            f"TkMET_sumEt_{label}": Hist.new.Reg(
                100,
                0,
                5000,
                name=f"TkMET_sumEt_{label}",
                label="TkMET sumEt",
            ).Weight(),
            f"RawMET_pt_{label}": Hist.new.Reg(
                100,
                0,
                1000,
                name=f"RawMET_pt_{label}",
                label="RawMET pt",
            ).Weight(),
            f"RawMET_phi_{label}": Hist.new.Reg(
                100,
                -4,
                4,
                name=f"RawMET_phi_{label}",
                label="RawMET phi",
            ).Weight(),
            f"RawMET_sumEt_{label}": Hist.new.Reg(
                100,
                0,
                5000,
                name=f"RawMET_sumEt_{label}",
                label="RawMET sumEt",
            ).Weight(),
            f"PuppiMET_pt_{label}": Hist.new.Reg(
                100,
                0,
                1000,
                name=f"PuppiMET_pt_{label}",
                label="PuppiMET pt",
            ).Weight(),
            f"PuppiMET_pt_JER_up_{label}": Hist.new.Reg(
                100,
                0,
                1000,
                name=f"PuppiMET_pt_JER_up_{label}",
                label="PuppiMET pt",
            ).Weight(),
            f"PuppiMET_pt_JER_down_{label}": Hist.new.Reg(
                100,
                0,
                1000,
                name=f"PuppiMET_pt_JER_down_{label}",
                label="PuppiMET pt",
            ).Weight(),
            f"PuppiMET_pt_JES_up_{label}": Hist.new.Reg(
                100,
                0,
                1000,
                name=f"PuppiMET_pt_JES_up{label}",
                label="PuppiMET pt",
            ).Weight(),
            f"PuppiMET_pt_JES_down_{label}": Hist.new.Reg(
                100,
                0,
                1000,
                name=f"PuppiMET_pt_JES_down{label}",
                label="PuppiMET pt",
            ).Weight(),
            f"PuppiMET_phi_{label}": Hist.new.Reg(
                100,
                -4,
                4,
                name=f"PuppiMET_phi_{label}",
                label="PuppiMET phi",
            ).Weight(),
            f"PuppiMET_phi_JER_up_{label}": Hist.new.Reg(
                100,
                -4,
                4,
                name=f"PuppiMET_phi_JER_up_{label}",
                label="PuppiMET phi",
            ).Weight(),
            f"PuppiMET_phi_JER_down_{label}": Hist.new.Reg(
                100,
                -4,
                4,
                name=f"PuppiMET_phi_JER_down_{label}",
                label="PuppiMET phi",
            ).Weight(),
            f"PuppiMET_phi_JES_up_{label}": Hist.new.Reg(
                100,
                -4,
                4,
                name=f"PuppiMET_phi_JES_up_{label}",
                label="PuppiMET phi",
            ).Weight(),
            f"PuppiMET_phi_JES_down_{label}": Hist.new.Reg(
                100,
                -4,
                4,
                name=f"PuppiMET_phi_JES_down_{label}",
                label="PuppiMET phi",
            ).Weight(),
            f"PuppiMET_sumEt_{label}": Hist.new.Reg(
                100,
                0,
                5000,
                name=f"PuppiMET_sumEt_{label}",
                label="PuppiMET sumEt",
            ).Weight(),
            f"RawPuppiMET_pt_{label}": Hist.new.Reg(
                100,
                0,
                1000,
                name=f"RawPuppiMET_pt_{label}",
                label="RawPuppiMET pt",
            ).Weight(),
            f"RawPuppiMET_phi_{label}": Hist.new.Reg(
                100,
                -4,
                4,
                name=f"RawPuppiMET_phi_{label}",
                label="RawPuppiMET phi",
            ).Weight(),
            f"RawPuppiMET_sumEt_{label}": Hist.new.Reg(
                100,
                0,
                5000,
                name=f"RawPuppiMET_sumEt_{label}",
                label="RawPuppiMET sumEt",
            ).Weight(),
            f"MET_pt_{label}": Hist.new.Reg(
                100,
                0,
                1000,
                name=f"MET_pt_{label}",
                label="MET pt",
            ).Weight(),
            f"MET_phi_{label}": Hist.new.Reg(
                100,
                -4,
                4,
                name=f"MET_phi_{label}",
                label="MET phi",
            ).Weight(),
            f"MET_sumEt_{label}": Hist.new.Reg(
                100,
                0,
                5000,
                name=f"MET_sumEt_{label}",
                label="MET sumEt",
            ).Weight(),
            f"MET_JEC_pt_{label}": Hist.new.Reg(
                100,
                0,
                1000,
                name=f"MET_JEC_pt_{label}",
                label="MET JEC pt",
            ).Weight(),
            f"MET_JEC_pt_JER_up_{label}": Hist.new.Reg(
                100,
                0,
                1000,
                name=f"MET_JEC_pt_JER_up_{label}",
                label="MET JEC pt",
            ).Weight(),
            f"MET_JEC_pt_JER_down_{label}": Hist.new.Reg(
                100,
                0,
                1000,
                name=f"MET_JEC_pt_JER_down_{label}",
                label="MET JEC pt",
            ).Weight(),
            f"MET_JEC_pt_JES_up_{label}": Hist.new.Reg(
                100,
                0,
                1000,
                name=f"MET_JEC_pt_JES_up_{label}",
                label="MET JEC pt",
            ).Weight(),
            f"MET_JEC_pt_JES_down_{label}": Hist.new.Reg(
                100,
                0,
                1000,
                name=f"MET_JEC_pt_JES_down_{label}",
                label="MET JEC pt",
            ).Weight(),
            f"MET_JEC_pt_UnclusteredEnergy_up_{label}": Hist.new.Reg(
                100,
                0,
                1000,
                name=f"MET_JEC_pt_UnclusteredEnergy_up_{label}",
                label="MET JEC pt",
            ).Weight(),
            f"MET_JEC_pt_UnclusteredEnergy_down_{label}": Hist.new.Reg(
                100,
                0,
                1000,
                name=f"MET_JEC_pt_UnclusteredEnergy_down_{label}",
                label="MET JCE pt",
            ).Weight(),
            f"MET_JEC_phi_{label}": Hist.new.Reg(
                100,
                -4,
                4,
                name=f"MET_JEC_phi_{label}",
                label="MET JEC phi",
            ).Weight(),
            f"MET_JEC_phi_JER_up_{label}": Hist.new.Reg(
                100,
                -4,
                4,
                name=f"MET_JEC_phi_JER_up_{label}",
                label="MET JEC phi",
            ).Weight(),
            f"MET_JEC_phi_JER_down_{label}": Hist.new.Reg(
                100,
                -4,
                4,
                name=f"MET_JEC_phi_JER_down_{label}",
                label="MET JEC phi",
            ).Weight(),
            f"MET_JEC_phi_JES_up_{label}": Hist.new.Reg(
                100,
                -4,
                4,
                name=f"MET_JEC_phi_JES_up_{label}",
                label="MET JEC phi",
            ).Weight(),
            f"MET_JEC_phi_JES_down_{label}": Hist.new.Reg(
                100,
                -4,
                4,
                name=f"MET_JEC_phi_JES_down_{label}",
                label="MET JEC phi",
            ).Weight(),
            f"MET_JEC_phi_UnclusteredEnergy_up_{label}": Hist.new.Reg(
                100,
                -4,
                4,
                name=f"MET_JEC_phi_UnclusteredEnergy_up_{label}",
                label="MET JEC phi",
            ).Weight(),
            f"MET_JEC_phi_UnclusteredEnergy_down_{label}": Hist.new.Reg(
                100,
                -4,
                4,
                name=f"MET_JEC_phi_UnclusteredEnergy_down_{label}",
                label="MET JEC phi",
            ).Weight(),
            f"MET_JEC_sumEt_{label}": Hist.new.Reg(
                100,
                0,
                5000,
                name=f"MET_JEC_sumEt_{label}",
                label="MET JEC sumEt",
            ).Weight(),
        }
    )

    ###########################################################################################################################
    if any([lbl in label for lbl in ["ISRRemoval", "Cluster", "Cone", "TopPT"]]):
        # 2D histograms
        output.update(
            {
                f"2D_SUEP_S1_vs_SUEP_nconst_{label}": Hist.new.Reg(
                    100, 0, 1.0, name=f"SUEP_S1_{label}", label="$Sph_1$"
                )
                .Reg(501, 0, 500, name=f"nconst_{label}", label="# Constituents")
                .Weight(),
            }
        )
        if (
            "up" not in label and "down" not in label
        ):  # don't care for systematics for these
            output.update(
                {
                    f"2D_SUEP_S1_vs_ntracks_{label}": Hist.new.Reg(
                        100, 0, 1.0, name=f"SUEP_S1_{label}", label="$Sph_1$"
                    )
                    .Reg(100, 0, 500, name=f"ntracks_{label}", label="# Tracks")
                    .Weight(),
                    f"2D_SUEP_nconst_vs_SUEP_pt_avg_{label}": Hist.new.Reg(
                        501, 0, 500, name=f"SUEP_nconst_{label}", label="# Const"
                    )
                    .Reg(200, 0, 500, name=f"SUEP_pt_avg_{label}", label="$p_T Avg$")
                    .Weight(),
                    f"2D_SUEP_eta_vs_SUEP_nconst_{label}": Hist.new.Reg(
                        100, -5, 5, name=f"SUEP_eta_{label}", label=r"$\eta$"
                    )
                    .Reg(501, 0, 500, name=f"nconst_{label}", label="# Constituents")
                    .Weight(),
                    f"2D_SUEP_pt_vs_SUEP_nconst_{label}": Hist.new.Reg(
                        100, 0, 2000, name=f"SUEP_pt_{label}", label="SUEP $p_T$"
                    )
                    .Reg(501, 0, 500, name=f"nconst_{label}", label="# Constituents")
                    .Weight(),
                }
            )

        # variables from the dataframe for all the events, and those in A, B, C regions
        for r in regions_list:
            output.update(
                {
                    f"{r}SUEP_nconst_{label}": Hist.new.Reg(
                        501,
                        0,
                        500,
                        name=f"{r}SUEP_nconst_{label}",
                        label="# Constituents",
                    ).Weight(),
                    f"{r}SUEP_S1_{label}": Hist.new.Reg(
                        100, 0, 1, name=f"{r}SUEP_S1_{label}", label="$Sph_1$"
                    ).Weight(),
                }
            )
            if (
                r == "" and "up" not in label and "down" not in label
            ):  # don't care for systematics for these
                output.update(
                    {
                        f"{r}SUEP_genMass_{label}": Hist.new.Reg(
                            100,
                            0,
                            1200,
                            name=f"{r}SUEP_genMass_{label}",
                            label="Gen Mass of SUEP ($m_S$) [GeV]",
                        ).Weight(),
                        f"{r}SUEP_pt_{label}": Hist.new.Reg(
                            100,
                            0,
                            2000,
                            name=f"{r}SUEP_pt_{label}",
                            label=r"SUEP $p_T$ [GeV]",
                        ).Weight(),
                        f"{r}SUEP_delta_pt_genPt_{label}": Hist.new.Reg(
                            400,
                            -2000,
                            2000,
                            name=f"{r}SUEP_delta_pt_genPt_{label}",
                            label="SUEP $p_T$ - genSUEP $p_T$ [GeV]",
                        ).Weight(),
                        f"{r}SUEP_pt_avg_{label}": Hist.new.Reg(
                            200,
                            0,
                            500,
                            name=f"{r}SUEP_pt_avg_{label}",
                            label=r"SUEP Components $p_T$ Avg.",
                        ).Weight(),
                        f"{r}SUEP_eta_{label}": Hist.new.Reg(
                            100,
                            -5,
                            5,
                            name=f"{r}SUEP_eta_{label}",
                            label=r"SUEP $\eta$",
                        ).Weight(),
                        f"{r}SUEP_phi_{label}": Hist.new.Reg(
                            100,
                            -6.5,
                            6.5,
                            name=f"{r}SUEP_phi_{label}",
                            label=r"SUEP $\phi$",
                        ).Weight(),
                        f"{r}SUEP_mass_{label}": Hist.new.Reg(
                            150,
                            0,
                            2000,
                            name=f"{r}SUEP_mass_{label}",
                            label="SUEP Mass [GeV]",
                        ).Weight(),
                        f"{r}SUEP_delta_mass_genMass_{label}": Hist.new.Reg(
                            400,
                            -2000,
                            2000,
                            name=f"{r}SUEP_delta_mass_genMass_{label}",
                            label="SUEP Mass - genSUEP Mass [GeV]",
                        ).Weight(),
                    }
                )
    ###########################################################################################################################
    if "ClusterInverted" in label:
        create_output_clusterInverted(output, label, regions_list)

    ###########################################################################################################################
    if label == "GNN" and options.doInf:
        create_output_GNN(abcd, output, label, regions_list)

    ###########################################################################################################################
    if label == "GNNInverted" and options.doInf:
        create_output_GNNInverted(abcd, output, label, regions_list)

    ###########################################################################################################################

    return output


def calculate_systematic(
    df,
    config,
    syst,
    options,
):
    # prepare new event weight
    if options.isMC:
        df["event_weight"] = df["genweight"].to_numpy()
    else:
        df["event_weight"] = np.ones(df.shape[0])

    if options.isMC == 1:
        if options.scouting != 1:
            # 1) pileup weights
            puweights, puweights_up, puweights_down = pileup_weight.pileup_weight(
                options.era
            )
            pu = pileup_weight.get_pileup_weights(
                df, syst, puweights, puweights_up, puweights_down
            )
            df["event_weight"] *= pu

            # 2) TriggerSF weights
            (
                trig_bins,
                trig_weights,
                trig_weights_up,
                trig_weights_down,
            ) = triggerSF.triggerSF(options.era)
            trigSF = triggerSF.get_trigSF_weight(
                df,
                syst,
                trig_bins,
                trig_weights,
                trig_weights_up,
                trig_weights_down,
            )
            df["event_weight"] *= trigSF

            # 3) PS weights
            if "PSWeight" in syst and syst in df.keys():
                df["event_weight"] *= df[syst]

            # 3) prefire weights
            if options.era == "2016" or options.era == "2017":
                if "prefire" in syst and syst in df.keys():
                    df["event_weight"] *= df[syst]
                else:
                    df["event_weight"] *= df["prefire_nom"]

        else:
            # 1) pileup weights
            puweights, puweights_up, puweights_down = pileup_weight.pileup_weight(
                options.era
            )
            pu = pileup_weight.get_pileup_weights(
                df, syst, puweights, puweights_up, puweights_down
            )
            df["event_weight"] *= pu

            # 2) TriggerSF weights
            trigSF = triggerSF.get_scout_trigSF_weight(
                np.array(df["ht"]).astype(int), syst, options.era
            )
            df["event_weight"] *= trigSF

            # 3) PS weights
            if "PSWeight" in syst and syst in df.keys():
                df["event_weight"] *= df[syst]

        # 5) Higgs_pt weights
        if "mS125" in options.dataset:
            (
                higgs_bins,
                higgs_weights,
                higgs_weights_up,
                higgs_weights_down,
            ) = higgs_reweight.higgs_reweight(df["SUEP_genPt"])
            higgs_weight = higgs_reweight.get_higgs_weight(
                df,
                syst,
                higgs_bins,
                higgs_weights,
                higgs_weights_up,
                higgs_weights_down,
            )
            df["event_weight"] *= higgs_weight

    # 6) scaling weights
    # N.B.: these aren't part of the systematics, just an optional scaling
    if scaling_weights is not None:
        df = fill_utils.apply_scaling_weights(
            df.copy(),
            scaling_weights,
            config["Cluster"],
            regions="ABCDEFGHI",
            x_var="SUEP_S1_CL",
            y_var="SUEP_nconst_CL",
            z_var="ht",
        )

    for label_out, config_out in config.items():
        if "track_down" in label_out and syst != "":
            continue  # don't run other systematics when doing track killing systematic
        if options.isMC and syst != "":
            if any([j in label_out for j in jet_corrections]):
                continue  # don't run other systematics when doing jet systematics

        # rename if we have applied a systematic
        if len(syst) > 0:
            label_out = label_out + "_" + syst

        # initialize new hists, if needed
        output.update(create_output_file(label_out, config_out, options))

        # prepare the DataFrame for plotting: blind, selections
        df_plot = fill_utils.prepareDataFrame(
            df.copy(), config_out, label_out, isMC=options.isMC, blind=options.blind
        )

        # auto fill all histograms
        fill_utils.auto_fill(
            df_plot,
            output,
            config_out,
            label_out,
            isMC=options.isMC,
            do_abcd=options.doABCD,
        )


#############################################################################################################

# variables that will be filled
nfailed = 0
total_gensumweight = 0
output = {"labels": []}

# get list of files
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
    xsection = fill_utils.getXSection(
        options.dataset, options.era, SUEP=bool(options.isSignal)
    )

# custom per region weights
scaling_weights = None
if options.weights is not None and options.weights != "None":
    scaling_weights = fill_utils.read_in_weights(options.weights)

# add new output methods for track killing and jet energy systematics
if options.isMC and options.doSyst:
    # track systematics: need to use the track_down version of the variables
    new_config_track_killing = fill_utils.get_track_killing_config(config)

    # jet systematics: just change ht to ht_SYS (e.g. ht -> ht_JEC_JES_up)
    jet_corrections = [
        "JER_up",
        "JER_down",
        "JES_up",
        "JES_down",
    ]
    new_config_jet_corrections = fill_utils.get_jet_corrections_config(
        config, jet_corrections
    )

    config = config | new_config_jet_corrections
    config = config | new_config_track_killing

logging.info("Setup ready, filling histograms now.")

# Plotting loop #######################################################################
for ifile in tqdm(files):
    #####################################################################################
    # ---- Load file
    #####################################################################################

    # get the file
    df, metadata = open_file(options, redirector, ifile)

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

    if options.isMC and options.doSyst:
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
            "prefire_up",
            "prefire_down",
        ]
        if "mS125" in options.dataset:
            sys_loop += [
                "higgs_weights_up",
                "higgs_weights_down",
            ]
    else:
        sys_loop = [""]

    for syst in sys_loop:
        # prepare new event weight
        calculate_systematic(df, config, syst, options)

    #####################################################################################
    # ---- End
    #####################################################################################

    # remove file at the end of loop
    if options.xrootd:
        os.system("rm " + options.dataset + ".hdf5")

logging.warning("Number of files that failed to be read: " + str(nfailed))
# End plotting loop ###################################################################

# not needed anymore
output.pop("labels")

logging.info("Applying symmetric systematics and normalization.")

# do the systematics
if options.isMC and options.doSyst:
    # do the tracks UP systematic
    output = track_killing.generate_up_histograms(config.keys(), output)

    if options.doInf:
        # do the GNN systematic
        GNN_syst.apply_GNN_syst(
            output,
            config["GNN"]["fGNNsyst"],
            config["GNN"]["models"],
            config["GNN"]["GNNsyst_bins"],
            options.era,
            out_label="GNN",
        )

# apply normalization
if options.isMC:
    print("xsection", xsection, "total_gensumweight", total_gensumweight)
    output = fill_utils.apply_normalization(output, xsection / total_gensumweight)

# Make ABCD expected histogram for signal region

if options.doABCD and options.blind and options.predictSR:
    logging.info("Predicting SR using ABCD method.")

    # Loop through every configuration
    for label_out, config_out in config.items():
        xregions = np.array(config_out["xvar_regions"]) * 1.0j
        yregions = np.array(config_out["yvar_regions"]) * 1.0j
        xvar = config_out["xvar"].replace("_" + config_out["input_method"], "")
        yvar = config_out["yvar"].replace("_" + config_out["input_method"], "")

        hist_name = f"2D_{xvar}_vs_{yvar}_{label_out}"

        # Check if histogram exists
        if hist_name not in output.keys():
            logging.warning(f"{hist_name} has not been created.")
            continue

        # Only calculate predicted for 9 region ABCD
        if (len(xregions) != 4) or (len(yregions) != 4):
            logging.warning(
                f"Can only calculate SR for 9 region ABCD, skipping {label_out}"
            )
            continue

        # Calculate SR from ABCD method
        # sum_var = 'x' corresponds to scaling F histogram
        SR, SR_exp = plot_utils.ABCD_9regions_errorProp(
            output[hist_name], xregions, yregions, sum_var="x"
        )

        output[f"I_{yvar}_{label_out}_exp"] = SR_exp

if options.dataset:
    outFile = outDir + "/" + options.dataset + "_" + options.output
else:
    outFile = os.path.join(outDir, options.output)
logging.info("Saving outputs to " + outFile + ".")
with uproot.recreate(outFile + ".root") as froot:
    for h, hist in output.items():
        froot[h] = hist
