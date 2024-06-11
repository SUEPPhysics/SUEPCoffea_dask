"""
Automatic histogram maker from the ntuples.
This script can make histograms for either a particular file, or a whole directory/sample of files (files here are intended as the ntuple hdf5 files).
It will fill all histograms that are part of the output dictionary (initialized in hist_defs.py), if it finds a matching variable in the ntuple dataframe.
(If doing an ABCD method, it will also fill each variable for each region.)
It can apply selections, blind, apply corrections and systematics, do ABCD method, make cutflows, and more.
The output histograms, cutflows, and metadata will be saved in a .root file.

e.g.
python make_hists.py --sample <sample> --output <output_tag> --tag <tag> --era <year> --isMC <bool> --doSyst <bool> --channel <channel>
"""

import argparse
import getpass
import logging
import os
import pickle
import subprocess
import sys

import numpy as np
import uproot
from tqdm import tqdm

sys.path.append("..")
import fill_utils
import hist_defs
from CMS_corrections import (
    GNN_syst,
    higgs_reweight,
    pileup_weight,
    track_killing,
    triggerSF,
)

import plotting.plot_utils as plot_utils


### Parser #######################################################################################################
def makeParser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser(description="Famous Submitter")

    # tag for output histograms
    parser.add_argument("-o", "--output", type=str, help="output tag", required=True)
    # Where the files are coming from. Either provide:
    # 1. one filepath with -f
    # 2. ntuple --tag and --sample for something in dataDirLocal.format(tag, sample) (or dataDirXRootD with --xrootd 1)
    # 3. a directory of files: dataDirLocal (or dataDirXRootD with --xrootd 1)
    parser.add_argument(
        "-sample",
        "--sample",
        type=str,
        default=None,
        help="sample name.",
        required=False,
    )
    parser.add_argument(
        "-t",
        "--tag",
        type=str,
        default="IronMan",
        help="ntuples production tag",
        required=False,
    )
    parser.add_argument("--file", type=str, default="", help="Use specific input file")
    parser.add_argument(
        "--xrootd",
        type=int,
        default=0,
        help="Local data or xrdcp from hadoop (default=False)",
    )
    parser.add_argument(
        "--dataDirLocal",
        type=str,
        default="/data/submit//cms/store/user/" + getpass.getuser() + "/SUEP/{}/{}/",
        help="Local data directory",
    )
    parser.add_argument(
        "--dataDirXRootD",
        type=str,
        default=f"/cms/store/user/" + getpass.getuser() + "/SUEP/{}/{}/",
        help="XRootD data directory",
    )
    ## optional: call it with --merged = 1 to append a /merged/ to the paths in options 2 and 3
    parser.add_argument("--merged", type=int, default=0, help="Use merged files")
    # some required info about the files
    parser.add_argument("-e", "--era", type=str, help="era", required=True)
    parser.add_argument("--isMC", type=int, help="Is this MC or data", required=True)
    parser.add_argument(
        "--channel",
        type=str,
        help="Analysis channel: ggF, WH",
        required=True,
        choices=["ggF", "WH"],
    )
    # some analysis parameters you can toggle freely
    parser.add_argument(
        "--scouting", type=int, default=0, help="Is this scouting or no"
    )
    parser.add_argument("--doInf", type=int, default=0, help="make GNN plots")
    parser.add_argument(
        "--doABCD", type=int, default=0, help="make plots for each ABCD+ region"
    )
    parser.add_argument(
        "--doSyst",
        type=int,
        default=0,
        help="Run systematic up and down variations in additional to the nominal.",
    )
    parser.add_argument(
        "--blind", type=int, default=1, help="Blind the data (default=True)"
    )
    parser.add_argument(
        "--weights",
        default="None",
        help="Pass the filename of the weights, e.g. --weights weights.npy",
    )
    parser.add_argument(
        "-p",
        "--printEvents",
        action="store_true",
        help="Print out events that pass the selections, used in particular for eventDisplay.py.",
        required=False,
    )
    # other arguments
    parser.add_argument(
        "--saveDir",
        type=str,
        default=f"/data/submit/{getpass.getuser()}/SUEP/outputs/",
        help="Use specific output directory. Overrides default MIT-specific path.",
        required=False,
    )
    parser.add_argument(
        "--redirector",
        type=str,
        default="root://submit50.mit.edu/",
        help="xrootd redirector (default: root://submit50.mit.edu/)",
        required=False,
    )
    parser.add_argument(
        "--maxFiles",
        type=int,
        default=-1,
        help="Maximum number of files to process (default=None, all files)",
        required=False,
    )
    parser.add_argument(
        "--pkl",
        type=int,
        default=0,
        help="Use pickle files instead of root files (default=False)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Run with verbose logging.",
        required=False,
    )
    return parser


### Main plotting function  ######################################################################################


def plot_systematic(df, metadata, config, syst, options, output, cutflow={}):
    # we might modify this for systematics, so make a copy
    config = config.copy()

    # prepare new event weight
    if options.isMC:
        df["event_weight"] = df["genweight"].to_numpy()
    else:
        df["event_weight"] = np.ones(df.shape[0])

    # apply systematics and weights
    if options.isMC:

        # 1) pileup weights
        puweights, puweights_up, puweights_down = pileup_weight.pileup_weight(
            options.era
        )
        pu = pileup_weight.get_pileup_weights(
            df, syst, puweights, puweights_up, puweights_down
        )
        df["event_weight"] *= pu

        # 2) PS weights
        if "PSWeight" in syst and syst in df.keys():
            df["event_weight"] *= df[syst]

        # 3) prefire weights
        if options.era == "2016" or options.era == "2017":
            if "prefire" in syst and syst in df.keys():
                df["event_weight"] *= df[syst]
            else:
                df["event_weight"] *= df["prefire_nom"]

        if options.channel == "ggF":
            if options.scouting != 1:
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

            else:
                # 2) TriggerSF weights
                trigSF = triggerSF.get_scout_trigSF_weight(
                    np.array(df["ht"]).astype(int), syst, options.era
                )
                df["event_weight"] *= trigSF

                # 4) prefire weights
                # no prefire weights for scouting

            # 5) Higgs_pt weights
            if "mS125" in metadata["sample"]:
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

            # 6) track killing
            if "track_down" in syst:
                # update configuration to cut on track_down variables
                config = fill_utils.get_track_killing_config(config)

            # 7) jet energy corrections
            if any([j in syst for j in ["JER", "JES"]]):
                # update configuration to cut on jet energy correction variables
                config = fill_utils.get_jet_correction_config(config, syst)

        elif options.channel == "WH":
            pass
            # FILL IN
            # should we keep these separate or try to, as much as possible, use the same code for systematics for both channels?
            # which systematics are applied and which aren't should be defined outside IMO, as is now
            # and in here we should just apply them as much as possible in the same way
            # with flags for the differences

    # scaling weights
    # N.B.: these are just an optional, arbitrary scaling of weights you're passing in
    if options.weights is not None and options.weights != "None":
        scaling_weights = fill_utils.read_in_weights(options.weights)
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
        # rename output method if we have applied a systematic
        if len(syst) > 0:
            label_out = label_out + "_" + syst

        # initialize new hists for this output tag, if we haven't already
        hist_defs.initialize_histograms(output, label_out, options, config_out)

        # prepare the DataFrame for plotting: blind, selections, new variables
        df_plot = fill_utils.prepare_DataFrame(
            df.copy(),
            config_out,
            label_out,
            isMC=options.isMC,
            blind=options.blind,
            cutflow=cutflow,
            output=output,
        )

        # if there are no events left after selections, no need to fill histograms
        if df_plot is None:
            continue

        # print out events that pass the selections, if requested
        if options.printEvents:
            print("Events passing selections for", label_out)
            for index, row in df_plot.iterrows():
                print(
                    f"{int(row['event'])}, {int(row['run'])}, {int(row['luminosityBlock'])}"
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


def main():
    parser = makeParser()
    options = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    if options.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    ##################################################################################################################
    # Script Parameters
    ##################################################################################################################
    """
    Define output plotting methods, each draws from an input_method (outputs of SUEPCoffea),
    and can have its own selections, ABCD regions, and signal region.
    Multiple plotting methods can be defined for the same input method, as different
    selections and ABCD methods can be applied.
    """

    if options.channel == "WH":
        new_variables_WH = [
            [
                "bjetSel",
                lambda x, y: ((x == 0) & (y < 2)),
                ["nBTight", "nBLoose"],
            ],
            [
                "W_SUEP_BV",
                fill_utils.balancing_var,
                ["W_pt", "SUEP_pt_HighestPT"],
            ],
            [
                "W_jet1_BV",
                fill_utils.balancing_var,
                ["W_pt", "jet1_pt"],
            ],
            [
                "ak4SUEP1_SUEP_BV",
                fill_utils.balancing_var,
                ["ak4jet1_inSUEPcluster_pt_HighestPT", "SUEP_pt_HighestPT"],
            ],
            [
                "W_SUEP_vBV",
                fill_utils.vector_balancing_var,
                [
                    "W_phi",
                    "SUEP_phi_HighestPT",
                    "W_pt",
                    "SUEP_pt_HighestPT",
                ],
            ],
            [
                "W_SUEP_vBV2",
                fill_utils.vector_balancing_var2,
                [
                    "W_phi",
                    "SUEP_phi_HighestPT",
                    "W_pt",
                    "SUEP_pt_HighestPT",
                ],
            ],
            [
                "W_jet1_vBV",
                fill_utils.vector_balancing_var,
                ["W_phi", "jet1_phi", "W_pt", "jet1_pt"],
            ],
            [
                "deltaPhi_SUEP_W",
                fill_utils.deltaPhi_x_y,
                [
                    "SUEP_phi_HighestPT",
                    "W_phi",
                ],
            ],
            [
                "deltaPhi_SUEP_MET",
                fill_utils.deltaPhi_x_y,
                [
                    "SUEP_phi_HighestPT",
                    "MET_phi",
                ],
            ],
            [
                "deltaPhi_lepton_MET",
                fill_utils.deltaPhi_x_y,
                ["lepton_phi", "MET_phi"],
            ],
            [
                "deltaPhi_lepton_SUEP",
                fill_utils.deltaPhi_x_y,
                [
                    "lepton_phi",
                    "SUEP_phi_HighestPT",
                ],
            ],
            [
                "deltaPhi_minDeltaPhiMETJet_SUEP",
                fill_utils.deltaPhi_x_y,
                [
                    "minDeltaPhiMETJet_phi",
                    "SUEP_phi_HighestPT",
                ],
            ],
            [
                "deltaPhi_minDeltaPhiMETJet_MET",
                fill_utils.deltaPhi_x_y,
                [
                    "minDeltaPhiMETJet_phi",
                    "MET_phi",
                ],
            ],
            [
                "deltaPhi_SUEP_jet1",
                fill_utils.deltaPhi_x_y,
                [
                    "SUEP_phi_HighestPT",
                    "jet1_phi",
                ],
            ],
            [
                "deltaPhi_SUEP_bjet",
                fill_utils.deltaPhi_x_y,
                [
                    "SUEP_phi_HighestPT",
                    "bjet_phi",
                ],
            ],
            [
                "deltaPhi_jet1_bjet",
                fill_utils.deltaPhi_x_y,
                ["jet1_phi", "bjet_phi"],
            ],
            [
                "deltaPhi_lepton_bjet",
                fill_utils.deltaPhi_x_y,
                ["lepton_phi", "bjet_phi"],
            ],
            [
                "nak4jets_outsideSUEP",
                lambda x, y: (x - y),
                ["ngood_ak4jets", "ak4jets_inSUEPcluster_n_HighestPT"],
            ],
            [
                "nonSUEP_S1",
                lambda x, y: 1.5 * (x + y),
                ["nonSUEP_eig0_HighestPT", "nonSUEP_eig1_HighestPT"],
            ],
            [
                "ntracks_outsideSUEP",
                lambda x, y: (x - y),
                ["ntracks", "SUEP_nconst_HighestPT"],
            ],
            [
                "BV_highestSUEPTrack_SUEP",
                fill_utils.balancing_var,
                ["SUEP_highestPTtrack_HighestPT", "SUEP_pt_HighestPT"],
            ],
            [
                "SUEP_nconst_minus_otherAK15_maxConst",
                lambda x, y: (x - y),
                ["SUEP_nconst_HighestPT", "otherAK15_maxConst_nconst_HighestPT"],
            ],
            [
                "jetsInSameHemisphere",
                lambda x, y: ((x == 1) | (y < 1.5)),
                ["ngood_ak4jets", "maxDeltaPhiJets"],
            ],
            [
                "deltaPhi_genSUEP_SUEP",
                fill_utils.deltaPhi_x_y,
                ["SUEP_genPhi", "SUEP_phi_HighestPT"],
            ],
            [
                "deltaR_genSUEP_SUEP",
                fill_utils.deltaR,
                [
                    "SUEP_genEta",
                    "SUEP_eta_HighestPT",
                    "SUEP_genPhi",
                    "SUEP_phi_HighestPT",
                ],
            ],
            [
                "percent_darkphis_inTracker",
                lambda x, y: x / y,
                ["n_darkphis_inTracker", "n_darkphis"],
            ],
            [
                "percent_tracks_dPhiW0p2",
                lambda x, y: x / y,
                ["ntracks_dPhiW0p2", "ntracks"],
            ],
            [
                "SUEPMostNumerous",
                lambda x, y: x > y,
                ["SUEP_nconst_HighestPT", "otherAK15_maxConst_nconst_HighestPT"],
            ],
            [
                "MaxConstAK15_phi",
                lambda x_nconst, y_nconst, x_phi, y_phi: np.where(
                    x_nconst > y_nconst, x_phi, y_phi
                ),
                [
                    "SUEP_nconst_HighestPT",
                    "otherAK15_maxConst_nconst_HighestPT",
                    "SUEP_phi_HighestPT",
                    "otherAK15_maxConst_phi_HighestPT",
                ],
            ],
            [
                "MaxConstAK15_eta",
                lambda x_nconst, y_nconst, x_eta, y_eta: np.where(
                    x_nconst > y_nconst, x_eta, y_eta
                ),
                [
                    "SUEP_nconst_HighestPT",
                    "otherAK15_maxConst_nconst_HighestPT",
                    "SUEP_eta_HighestPT",
                    "otherAK15_maxConst_eta_HighestPT",
                ],
            ],
            [
                "deltaPhi_SUEPgen_MaxConstAK15",
                fill_utils.deltaPhi_x_y,
                ["SUEP_genPhi", "MaxConstAK15_phi"],
            ],
            [
                "deltaR_SUEPgen_MaxConstAK15",
                fill_utils.deltaR,
                ["SUEP_genEta", "MaxConstAK15_eta", "SUEP_genPhi", "MaxConstAK15_phi"],
            ],
            [
                "highestPTtrack_pt_norm",
                lambda x, y: x / y,
                ["SUEP_highestPTtrack_HighestPT", "SUEP_pt_HighestPT"],
            ],
            [
                "highestPTtrack_pt_norm2",
                lambda x, y: x / y,
                ["SUEP_highestPTtrack_HighestPT", "SUEP_pt_avg_HighestPT"],
            ],
            ["isMuon", lambda x: abs(x) == 13, ["lepton_flavor"]],
            ["isElectron", lambda x: abs(x) == 11, ["lepton_flavor"]],
        ]
        if options.isMC:
            new_variables_WH += [
                ["deltaPhi_W_genW", fill_utils.deltaPhi_x_y, ["genW_phi", "W_phi"]],
                ["deltaPt_W_genW", lambda x, y: x - y, ["genW_pt", "W_pt"]],
            ]
        config = {
            "SR": {
                "input_method": "HighestPT",
                "method_var": "SUEP_nconst_HighestPT",
                "SR": [
                    ["SUEP_S1_HighestPT", ">=", 0.3],
                    ["SUEP_nconst_HighestPT", ">=", 50],
                ],
                "selections": [
                    "MET_pt > 30",
                    "W_pt > 40",
                    "W_mt < 130",
                    "W_mt > 30",
                    "bjetSel == 1",
                    "deltaPhi_SUEP_W > 1.5",
                    "deltaPhi_SUEP_MET > 1.5",
                    "deltaPhi_lepton_SUEP > 1.5",
                    "ak4jets_inSUEPcluster_n_HighestPT >= 1",
                    "deltaPhi_minDeltaPhiMETJet_MET > 0.4",
                    "W_SUEP_BV < 2",
                    "deltaPhi_minDeltaPhiMETJet_MET > 1.5",
                ],
                "new_variables": new_variables_WH,
            },
            "CRWJ": {
                "input_method": "HighestPT",
                "method_var": "SUEP_nconst_HighestPT",
                "SR": [
                    ["SUEP_S1_HighestPT", ">=", 0.3],
                    ["SUEP_nconst_HighestPT", ">=", 50],
                ],
                "selections": [
                    "MET_pt > 30",
                    "W_pt > 40",
                    "W_mt < 130",
                    "W_mt > 30",
                    "bjetSel == 1",
                    "deltaPhi_SUEP_W > 1.5",
                    "deltaPhi_SUEP_MET > 1.5",
                    "deltaPhi_lepton_SUEP > 1.5",
                    "ak4jets_inSUEPcluster_n_HighestPT >= 1",
                    "W_SUEP_BV < 2",
                    "deltaPhi_minDeltaPhiMETJet_MET > 1.5",
                    "SUEP_S1_HighestPT < 0.3",
                    "SUEP_nconst_HighestPT < 50",
                ],
                "new_variables": new_variables_WH,
            },
            "CRTT": {
                "input_method": "HighestPT",
                "method_var": "SUEP_nconst_HighestPT",
                "SR": [
                    ["SUEP_S1_HighestPT", ">=", 0.3],
                    ["SUEP_nconst_HighestPT", ">=", 40],
                ],
                "selections": [
                    "MET_pt > 30",
                    "W_pt > 40",
                    "W_mt < 130",
                    "W_mt > 30",
                    "bjetSel == 0",
                    "deltaPhi_SUEP_W > 1.5",
                    "deltaPhi_SUEP_MET > 1.5",
                    "deltaPhi_lepton_SUEP > 1.5",
                    "ak4jets_inSUEPcluster_n_HighestPT >= 1",
                    "W_SUEP_BV < 2",
                    "deltaPhi_minDeltaPhiMETJet_MET > 1.5",
                    "SUEP_S1_HighestPT < 0.3",
                    "SUEP_nconst_HighestPT < 50",
                ],
                "new_variables": new_variables_WH,
            },
        }

    if options.channel == "ggF":
        if options.scouting:
            config = {
                "Cluster": {
                    "input_method": "CL",
                    "method_var": "SUEP_S1_CL",
                    "xvar": "SUEP_S1_CL",
                    "xvar_regions": [0.3, 0.34, 0.5, 2.0],
                    "yvar": "SUEP_nconst_CL",
                    "yvar_regions": [0, 18, 50, 1000],
                    "SR": [["SUEP_S1_CL", ">=", 0.5], ["SUEP_nconst_CL", ">=", 70]],
                    "selections": [["ht_JEC", ">", 560], ["ntracks", ">", 0]],
                },
                "ClusterInverted": {
                    "input_method": "CL",
                    "method_var": "ISR_S1_CL",
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
                    "method_var": "SUEP_S1_CL",
                    "xvar": "SUEP_S1_CL",
                    "xvar_regions": [0.3, 0.4, 0.5, 2.0],
                    "yvar": "SUEP_nconst_CL",
                    "yvar_regions": [30, 50, 70, 1000],
                    "SR": [["SUEP_S1_CL", ">=", 0.5], ["SUEP_nconst_CL", ">=", 70]],
                    "selections": [
                        ["ht_JEC", ">", 1200],
                        ["ntracks", ">", 0],
                        "SUEP_nconst_CL > 30",
                        "SUEP_S1_CL > 0.3",
                    ],
                    "new_variables": [
                        [
                            "SUEP_ISR_deltaPhi_CL",
                            lambda x, y: abs(x - y),
                            ["SUEP_phi_CL", "ISR_phi_CL"],
                        ]
                    ],
                },
                "ClusterInverted": {
                    "input_method": "CL",
                    "method_var": "ISR_S1_CL",
                    "xvar": "ISR_S1_CL",
                    "xvar_regions": [0.3, 0.4, 0.5, 2.0],
                    "yvar": "ISR_nconst_CL",
                    "yvar_regions": [30, 50, 70, 1000],
                    "SR": [["SUEP_S1_CL", ">=", 0.5], ["SUEP_nconst_CL", ">=", 70]],
                    "selections": [["ht_JEC", ">", 1200], ["ntracks", ">", 0]],
                },
            }

        if options.doInf:
            config.update(
                {
                    "GNN": {
                        "input_method": "GNN",
                        "method_var": "SUEP_S1_GNN",
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
                        "method_var": "ISR_S1_GNNInverted",
                        "xvar": "ISR_S1_GNNInverted",
                        "xvar_regions": [0.0, 1.5, 2.0],
                        "yvar": "single_l5_bPfcand_S1_SUEPtracks_GNNInverted",
                        "yvar_regions": [0.0, 1.5, 2.0],
                        "SR": [
                            ["ISR_S1_GNNInverted", ">=", 10.0],
                            ["single_l5_bPfcand_S1_SUEPtracks_GNNInverted", ">=", 10.0],
                        ],
                        "selections": [["ht_JEC", ">", 1200], ["ntracks", ">", 40]],
                        "models": ["single_l5_bPfcand_S1_SUEPtracks"],
                    },
                }
            )

    ### Script preamble and set up ###################################################################################

    # variables that will be filled
    nfailed = 0
    ntotal = 0
    total_gensumweight = 0
    xsection = 1
    lumi = 1
    output = {"labels": []}
    cutflow = {}

    # get list of files
    if options.file:
        files = [options.file]
    elif options.xrootd:
        dataDir = (
            options.dataDirXRootD.format(options.tag, options.sample)
            if options.dataDirXRootD.count("{}") == 2
            else options.dataDirXRootD
        )
        if options.merged:
            dataDir += "/merged/"
        result = subprocess.check_output(["xrdfs", options.redirector, "ls", dataDir])
        result = result.decode("utf-8")
        files = result.split("\n")
        files = [f for f in files if len(f) > 0]
    else:
        dataDir = (
            options.dataDirLocal.format(options.tag, options.sample)
            if options.dataDirLocal.count("{}") == 2
            else options.dataDir
        )
        if options.merged:
            dataDir += "merged/"
        files = [dataDir + f for f in os.listdir(dataDir)]
    if options.maxFiles > 0:
        files = files[: options.maxFiles]
    files = [f for f in files if ".hdf5" in f]
    ntotal = len(files)

    if ntotal == 0:
        logging.error("No files found, exiting.")
        sys.exit(1)

    ### Plotting loop ################################################################################################

    logging.info("Setup ready, filling histograms now.")

    sample = options.sample
    for ifile in tqdm(files):
        # get the file
        df, metadata = fill_utils.open_ntuple(
            ifile, redirector=options.redirector, xrootd=options.xrootd
        )
        logging.debug(f"Opened file {ifile}")
        if options.printEvents:
            print(f"Opened file {ifile}")

        # check if file is corrupted
        if type(df) == int:
            nfailed += 1
            logging.debug(f"File {ifile} is corrupted, skipping.")
            continue

        # check sample consistency
        if metadata != 0 and "sample" in metadata.keys():
            if (
                sample is None
            ):  # we did not pass in any sample, and this is the first file
                sample = metadata["sample"]
            elif (
                metadata["sample"] == "X"
            ):  # default option for ntuplemaker, when not run properly specifying which sample. Ignore this.
                pass
            else:  # if we already have a sample, check it matches the metadata of the first file or what we passed in
                assert (
                    sample == metadata["sample"]
                ), "This script should only run on one sample at a time. Found {} in metadata, and passed sample {}".format(
                    metadata["sample"], sample
                )

        # update the gensumweight
        if options.isMC and metadata != 0:
            logging.debug("Updating gensumweight.")
            total_gensumweight += metadata["gensumweight"]

        # update the cutflows
        if metadata != 0 and any(["cutflow" in k for k in metadata.keys()]):
            logging.debug("Updating cutflows.")
            for k, v in metadata.items():
                if "cutflow" in k:
                    if k not in cutflow.keys():
                        cutflow[k] = v
                    else:
                        cutflow[k] += v

        # check if any events passed the selections
        if "empty" in list(df.keys()):
            logging.debug("No events passed the selections, skipping.")
            continue
        if df.shape[0] == 0:
            logging.debug("No events in file, skipping.")
            continue

        # define which systematics to loop over
        sys_loop = []
        if options.isMC and options.doSyst:
            if options.channel == "ggF":
                sys_loop = [
                    "puweights_up",
                    "puweights_down",
                    "trigSF_up",
                    "trigSF_down",
                    "PSWeight_ISR_up",
                    "PSWeight_ISR_down",
                    "PSWeight_FSR_up",
                    "PSWeight_FSR_down",
                    "track_down",
                    "JER_up",
                    "JER_down",
                    "JES_up",
                    "JES_down",
                ]
                if "mS125" in metadata["sample"]:
                    sys_loop += [
                        "higgs_weights_up",
                        "higgs_weights_down",
                    ]
                if options.scouting == 0:
                    sys_loop += [
                        "prefire_up",
                        "prefire_down",
                    ]
            elif options.channel == "WH":
                sys_loop = [
                    "puweights_up",
                    "puweights_down",
                    "PSWeight_ISR_up",
                    "PSWeight_ISR_down",
                    "PSWeight_FSR_up",
                    "PSWeight_FSR_down",
                    "prefire_up",
                    "prefire_down",
                    "track_down",
                    "JER_up",
                    "JER_down",
                    "JES_up",
                    "JES_down",
                ]
                if "mS125" in metadata["sample"]:
                    sys_loop += [
                        "higgs_weights_up",
                        "higgs_weights_down",
                    ]

        logging.debug("Running nominal histograms.")
        plot_systematic(df, metadata, config, "", options, output, cutflow)

        for syst in sys_loop:
            logging.debug(f"Running systematic {syst}")
            plot_systematic(df, metadata, config, syst, options, output, cutflow)

        # remove file at the end of loop
        if options.xrootd:
            logging.debug(f"Removing file {ifile}")
            fill_utils.close_ntuple(ifile)

    if nfailed > 0:
        logging.warning("Number of files that failed to be read: " + str(nfailed))

    ### Post-processing stuff ########################################################################################

    # not needed anymore
    output.pop("labels")

    logging.info("Applying symmetric systematics and normalization.")

    # do some systematics that you need the full histograms for
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

    # store whether sample is signal
    isSignal = (options.isMC) and (fill_utils.isSampleSignal(sample, options.era))
    logging.debug("Is signal: " + str(isSignal))

    # apply normalization to samples
    if options.isMC:
        logging.debug(f"Found total_gensumweight {total_gensumweight}.")
        xsection = fill_utils.getXSection(sample, options.era, failOnKeyError=True)
        logging.debug(f"Found cross section x kr x br: {xsection}.")
        lumi = plot_utils.getLumi(options.era, options.scouting)
        logging.debug(f"Found lumi: {lumi}.")

        if isSignal:
            normalization = 1 / total_gensumweight
        else:
            normalization = xsection * lumi / total_gensumweight

        logging.info(f"Applying normalization: {normalization}.")
        output = fill_utils.apply_normalization(output, normalization)
        cutflow = fill_utils.apply_normalization(cutflow, normalization)

    # form metadata
    metadata = {
        "ntuple_tag": options.tag,
        "analysis": options.channel,
        "scouting": int(options.scouting),
        "isMC": int(options.isMC),
        "signal": int(isSignal),
        "era": options.era,
        "sample": sample,
        "xsec": float(xsection),
        "gensumweight": float(total_gensumweight),
        "lumi": float(lumi),
        "nfiles": ntotal,
        "nfailed": nfailed,
    }
    if cutflow is not {}:
        for k, v in cutflow.items():
            metadata[k] = v
    commit, diff = fill_utils.get_git_info()
    metadata["git_commit"] = commit
    metadata["git_diff"] = diff

    ### Write output #################################################################################################

    # write histograms and metadata to a root or pkl file
    outFile = options.saveDir + "/" + sample + "_" + options.output
    if options.pkl:
        outFile += ".pkl"
        logging.info("Saving outputs to " + outFile)
        with open(outFile, "wb") as f:
            pickle.dump({"metadata": metadata, "hists": output}, f)
    else:
        outFile += ".root"
        logging.info("Saving outputs to " + outFile)
        with uproot.recreate(outFile) as froot:
            # write out metadata
            for k, m in metadata.items():
                froot[f"metadata/{k}"] = str(m)

            # write out histograms
            for h, hist in output.items():
                froot[h] = hist


if __name__ == "__main__":
    main()
