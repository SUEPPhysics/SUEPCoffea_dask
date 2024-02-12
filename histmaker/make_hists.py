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

import plotting.plot_utils


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
        default="sample",
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
        help="xrootd redirector",
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
        "--verbose",
        type=int,
        default=0,
        help="verbosity level",
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
    if options.isMC == 1:

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
        )

        # if there are no events left after selections, no need to fill histograms
        if df_plot is None:
            continue

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
        config = {
            # keys of config must include "HighestPT", but otherwise can be named to convenience,
            # NOTE: it functions as a label, so useful to name according to selections that the key points to
            # input method should always be HighestPT
            "HighestPT": {
                "input_method": "HighestPT",
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

    ### Plotting loop ################################################################################################

    logging.info("Setup ready, filling histograms now.")

    sample = None
    for ifile in tqdm(files):
        # get the file
        df, metadata = fill_utils.open_ntuple(
            ifile, redirector=options.redirector, xrootd=options.xrootd
        )
        logging.debug(f"Opened file {ifile}")

        # check if file is corrupted
        if type(df) == int:
            nfailed += 1
            logging.debug(f"File {ifile} is corrupted, skipping.")
            continue

        # check sample consistency
        if "sample" in metadata.keys():
            if sample is None:
                sample = metadata["sample"]
            else:
                assert (
                    sample == metadata["sample"]
                ), "This script should only run on one sample at a time."

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

    # apply xsec and gensumweight (but no xsec to SUEP signal samples)
    if options.isMC:
        logging.info("Applying normalization.")
        xsection = 1
        if "SUEP" not in sample:
            xsection = fill_utils.getXSection(sample, options.era)
            logging.debug(f"Applying cross section {xsection}.")
        logging.debug(f"Applying total_gensumweight {total_gensumweight}.")
        output = fill_utils.apply_normalization(output, xsection / total_gensumweight)
        cutflow = fill_utils.apply_normalization(cutflow, xsection / total_gensumweight)

    # Make ABCD expected histogram for signal region
    if options.doABCD and options.predictSR:
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

            # Only calculate predicted for 9 region ABCD for now, should be made flexible based on the number of regions defined in the config
            if (len(xregions) != 4) or (len(yregions) != 4):
                logging.warning(
                    f"Can only calculate SR for 9 region ABCD, skipping {label_out}"
                )
                continue

            try:
                # Calculate expected SR from ABCD method
                # sum_var = 'x' corresponds to scaling F histogram
                _, SR_exp = plot_utils.ABCD_9regions_errorProp(
                    output[hist_name], xregions, yregions, sum_var="x"
                )
                output[f"I_{yvar}_{label_out}_exp"] = SR_exp
            except ZeroDivisionError:
                logging.warning(f"ZeroDivisionError for {label_out}, skipping.")
                continue

    # form metadata
    metadata = {
        "ntuple_tag": options.tag,
        "analysis": options.channel,
        "scouting": options.scouting,
        "isMC": options.isMC,
        "era": options.era,
        "sample": sample,
        "xsec": xsection,
        "gensumweight": total_gensumweight,
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

    # write histograms and metadata to a root file
    outFile = options.saveDir + "/" + sample + "_" + options.output + ".root"
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
