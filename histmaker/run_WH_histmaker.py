"""
Script to run the SUEPDaskHistMaker for the WH analysis.
Author: Luca Lavezzo
Date: August 2024
"""

import argparse
import os
from types import SimpleNamespace
from copy import deepcopy

import fill_utils
import hist_defs
import var_defs
from SUEPDaskHistMaker import SUEPDaskHistMaker


def getOptions() -> dict:

    username = os.environ["USER"]

    parser = argparse.ArgumentParser(description="Run the SUEP histogram maker")

    # arguments for this script
    parser.add_argument(
        "--input", "-i", type=str, help="Text file of samples to process"
    )
    parser.add_argument(
        "--sample", type=str, help="Sample to process, alternative to --input"
    )
    parser.add_argument(
        "--client",
        type=str,
        default="local",
        choices=["local", "slurm"],
        help="Where to set up the client",
    )
    parser.add_argument(
        "--nworkers", "-n", type=int, default=20, help="Number of workers to use"
    )
    parser.add_argument(
        "--limits", action="store_true", help="Make hists for limits"
    )

    # required SUEPDaskHistMaker arguments
    parser.add_argument("--isMC", type=int, default=1, help="isMC")
    parser.add_argument("--channel", type=str, default="WH", help="Channel")
    parser.add_argument("--era", type=str, required=True, help="Era")
    parser.add_argument("--tag", type=str, required=True, help="Ntuple tag")
    parser.add_argument("--output", type=str, required=True, help="Output tag")

    # optional SUEPDaskHistMaker arguments
    parser.add_argument("--doSyst", type=int, default=0, help="Do systematics")
    parser.add_argument("--verbose", type=int, default=0, help="Verbose")
    parser.add_argument(
        "--xrootd", type=int, default=0, help="Use xrootd to read the ntuples"
    )
    parser.add_argument(
        "--saveDir",
        type=str,
        default="/ceph/submit/data/user/"
        + username[0]
        + "/"
        + username
        + "/SUEP/outputs/",
        help="Save directory",
    )
    parser.add_argument(
        "--logDir",
        type=str,
        default="/work/submit/" + username + "/SUEP/logs/",
        help="Log directory",
    )
    parser.add_argument(
        "--dataDirLocal",
        type=str,
        default="/ceph/submit/data/user/"
        + username[0]
        + "/"
        + username
        + "/SUEP/{}/{}/",
        help="Local ntuple directory",
    )
    parser.add_argument(
        "--dataDirXRootD",
        type=str,
        default="/cms/store/user/" + username + "/SUEP/{}/{}/",
        help="XRootD ntuple directory",
    )
    parser.add_argument("--merged", type=int, default=0, help="Use merged ntuples")
    parser.add_argument(
        "--maxFiles", type=int, default=-1, help="Maximum number of files to run on"
    )
    parser.add_argument("--pkl", type=int, default=1, help="Use pickle files")
    parser.add_argument("--printEvents", type=int, default=0, help="Print events")
    parser.add_argument(
        "--redirector", type=str, default="root://submit50.mit.edu/", help="Redirector"
    )
    parser.add_argument("--doABCD", type=int, default=0, help="Do ABCD")
    parser.add_argument(
        "--file",
        type=str,
        help="Ntuple single filename, in case you don't want to process a whole sample",
    )
    parser.add_argument("--blind", type=int, default=1, help="Blind")

    options = parser.parse_args()
    options = vars(options)

    # either input or sample must be provided
    if not options.get("input") and not options.get("sample"):
        raise ValueError("Either --input or --sample must be provided")
    if options.get("input"):
        with open(options["input"]) as f:
            options["samples"] = [line.split("/")[-1] for line in f.read().splitlines()]
    else:
        options["samples"] = [options["sample"]]

    return options


def main():

    # get the options
    options = getOptions()

    # set up your configuration
    if options['limits']:
        options['pkl'] = 0 # need .root files for combine
        options['doABCD'] = 1
        options['doSyst'] = 1
        if options['channel'] == "WH":
            config = {
                "SR": {
                    "input_method": "HighestPT",
                    "method_var": "SUEP_nconst_HighestPT",
                    "xvar": "SUEP_S1_HighestPT",
                    "xvar_regions": [0.3, 0.4, 0.5, 2.0],
                    "yvar": "SUEP_nconst_HighestPT",
                    "yvar_regions": [20, 30, 1000],
                    "SR": [
                        ["SUEP_S1_HighestPT", ">=", 0.5],
                        ["SUEP_nconst_HighestPT", ">=", 30]
                    ],
                    "selections": [
                        "PuppiMET_pt > 30",
                        "W_pt > 60",
                        "W_mt < 130",
                        "W_mt > 30",
                        "bjetSel == 1",
                        "deltaPhi_SUEP_W > 1.5",
                        "deltaPhi_SUEP_MET > 1.5",
                        "deltaPhi_lepton_SUEP > 1.5",
                        "ak4jets_inSUEPcluster_n_HighestPT >= 1",
                        "W_SUEP_BV < 2",
                        "deltaPhi_minDeltaPhiMETJet_MET > 1.5",
                        "SUEP_S1_HighestPT > 0.3",
                    ],
                    "syst":  [
                        "puweights_up",
                        "puweights_down",
                        "PSWeight_ISR_up",
                        "PSWeight_ISR_down",
                        "PSWeight_FSR_up",
                        "PSWeight_FSR_down",
                        "higgs_weights_up",
                        "higgs_weights_down",
                        "LepSFElUp",
                        "LepSFElDown",
                        "LepSFMuUp",
                        "LepSFMuDown",
                        "bTagWeight_HFcorrelated_Up",
                        "bTagWeight_HFcorrelated_Dn",
                        "bTagWeight_HFuncorrelated_Up",
                        "bTagWeight_HFuncorrelated_Dn",
                        "bTagWeight_LFcorrelated_Up",
                        "bTagWeight_LFcorrelated_Dn",
                        "bTagWeight_LFuncorrelated_Up",
                        "bTagWeight_LFuncorrelated_Dn",
                        "prefire_up",
                        "prefire_down",
                        # TODO: trigger scale factors, prefire
                    ],
                },
                # "CRWJ": {
                #     "input_method": "HighestPT",
                #     "method_var": "SUEP_nconst_HighestPT",
                #     "xvar": "SUEP_S1_HighestPT",
                #     "xvar_regions": [0.05, 0.15, 0.25, 0.3],
                #     "yvar": "SUEP_nconst_HighestPT",
                #     "yvar_regions": [20, 30, 1000],
                #     "selections": [
                #             "PuppiMET_pt > 30",
                #             "W_pt > 60",
                #             "W_mt < 130",
                #             "W_mt > 30",
                #             "bjetSel == 1",
                #             "deltaPhi_SUEP_W > 1.5",
                #             "deltaPhi_SUEP_MET > 1.5",
                #             "deltaPhi_lepton_SUEP > 1.5",
                #             "ak4jets_inSUEPcluster_n_HighestPT >= 1",
                #             "W_SUEP_BV < 2",
                #             "deltaPhi_minDeltaPhiMETJet_MET > 1.5",
                #             "SUEP_S1_HighestPT < 0.3",
                #     ],
                #     "syst":  [],
                # }
            }

            if options.get("isMC"):

                # add systematic variations, which are identical, except for the dataframe name
                variations = [
                    "MuScaleUp",
                    "MuScaleDown",
                    "track_down"
                ]
                var_config = {}
                for var in variations:
                    for tag, config_tag in config.items():
                        _var_config = deepcopy(config_tag)
                        _var_config["syst"] = []    # don't need to run SFs for systematic variations
                        _var_config["df_name"] = "vars_" + var # name of the dataframe in the hdf5 ntuple
                        var_config.update({tag + "_" + var: _var_config})

                # systematic variations that change the MET and W selections
                met_variations = [
                    "JER_up",
                    "JER_down",
                    "JES_up",
                    "JES_down",
                ]
                for var in met_variations:
                    for tag, config_tag in config.items():
                        _var_config = deepcopy(config_tag)
                        _var_config["syst"] = []
                        # add the MET variation to the selection name, variables are defined in var_defs
                        for iSel in range(len(_var_config["selections"])):
                            s = _var_config["selections"][iSel]
                            if "MET_" in s or "W_" in s:
                                s = s.split(" ")
                                s[0] = s[0] + "_" + var
                                s = " ".join(s)
                                _var_config["selections"][iSel] = s
                            var_config.update({tag + "_" + var: _var_config})
                        
                config.update(var_config)

        elif options['channel'] == "WH-VRGJ":
            config = {
                "VRGJlowS": {
                    "input_method": "HighestPT",
                    "method_var": "SUEP_nconst_HighestPT",
                    "xvar": "SUEP_S1_HighestPT",
                    "xvar_regions": [0.05, 0.15, 0.25, 0.3],
                    "yvar": "SUEP_nconst_HighestPT",
                    "yvar_regions": [20, 30, 1000],
                    "selections": [
                        "SUEP_nconst_HighestPT >= 10",
                        "bjetSel == 1",
                        "minDeltaPhiJetPhoton > 1.5",
                        "deltaPhi_SUEP_photon > 1.5",
                        "ak4jets_inSUEPcluster_n_HighestPT >= 1",
                        "photon_SUEP_BV < 2",
                        "SUEP_S1_HighestPT < 0.3",
                    ],
                    "syst":  [
                        "puweights_up",
                        "puweights_down",
                        "PSWeight_ISR_up",
                        "PSWeight_ISR_down",
                        "PSWeight_FSR_up",
                        "PSWeight_FSR_down",
                        "higgs_weights_up",
                        "higgs_weights_down",
                        "bTagWeight_HFcorrelated_Up",
                        "bTagWeight_HFcorrelated_Dn",
                        "bTagWeight_HFuncorrelated_Up",
                        "bTagWeight_HFuncorrelated_Dn",
                        "bTagWeight_LFcorrelated_Up",
                        "bTagWeight_LFcorrelated_Dn",
                        "bTagWeight_LFuncorrelated_Up",
                        "bTagWeight_LFuncorrelated_Dn",
                        "prefire_up",
                        "prefire_down",
                        "photon_SF_up",
                        "photon_SF_down",
                    ],
                },
                "VRGJhighS": {
                    "input_method": "HighestPT",
                    "method_var": "SUEP_nconst_HighestPT",
                    "xvar": "SUEP_S1_HighestPT",
                    "xvar_regions": [0.3, 0.4, 0.5, 2.0],
                    "yvar": "SUEP_nconst_HighestPT",
                    "yvar_regions": [20, 30, 1000],
                    "selections": [
                        "SUEP_nconst_HighestPT >= 10",
                        "bjetSel == 1",
                        "minDeltaPhiJetPhoton > 1.5",
                        "deltaPhi_SUEP_photon > 1.5",
                        "ak4jets_inSUEPcluster_n_HighestPT >= 1",
                        "photon_SUEP_BV < 2",
                        "SUEP_S1_HighestPT > 0.3",
                    ],
                    "syst":  [
                        "puweights_up",
                        "puweights_down",
                        "PSWeight_ISR_up",
                        "PSWeight_ISR_down",
                        "PSWeight_FSR_up",
                        "PSWeight_FSR_down",
                        "higgs_weights_up",
                        "higgs_weights_down",
                        "bTagWeight_HFcorrelated_Up",
                        "bTagWeight_HFcorrelated_Dn",
                        "bTagWeight_HFuncorrelated_Up",
                        "bTagWeight_HFuncorrelated_Dn",
                        "bTagWeight_LFcorrelated_Up",
                        "bTagWeight_LFcorrelated_Dn",
                        "bTagWeight_LFuncorrelated_Up",
                        "bTagWeight_LFuncorrelated_Dn",
                        "prefire_up",
                        "prefire_down",
                        "photon_SF_up",
                        "photon_SF_down",
                    ],
                },
            }

            if options.get("isMC"):

                # add systematic variations, which are identical, except for the dataframe name
                variations = [
                    "track_down"
                ]
                var_config = {}
                for var in variations:
                    for tag, config_tag in config.items():
                        _var_config = deepcopy(config_tag)
                        _var_config["syst"] = []    # don't need to run SFs for systematic variations
                        _var_config["df_name"] = "vars_" + var # name of the dataframe in the hdf5 ntuple
                        var_config.update({tag + "_" + var: _var_config})

    elif options["channel"] == "WH":
        config = {
            # "WJnoB": {
            #     "input_method": "HighestPT",
            #     "method_var": "SUEP_nconst_HighestPT",
            #     "SR": [
            #         ["SUEP_S1_HighestPT", ">=", 0.5],
            #         ["SUEP_nconst_HighestPT", ">=", 30]
            #     ],
            #     "selections": [
            #         "SUEP_nconst_HighestPT >= 10",
            #         "PuppiMET_pt > 30",
            #         "W_pt > 60",
            #         "W_mt < 130",
            #         "W_mt > 30",
            #         "deltaPhi_SUEP_W > 1.5",
            #         "deltaPhi_SUEP_MET > 1.5",
            #         "deltaPhi_lepton_SUEP > 1.5",
            #         "ak4jets_inSUEPcluster_n_HighestPT >= 1",
            #         "W_SUEP_BV < 2",
            #         "deltaPhi_minDeltaPhiMETJet_MET > 1.5",
            #     ],
            #     "syst":  [
            #     ],
            #  },
             "SR": {
                "input_method": "HighestPT",
                "method_var": "SUEP_nconst_HighestPT",
                "SR": [
                    ["SUEP_S1_HighestPT", ">=", 0.5],
                    ["SUEP_nconst_HighestPT", ">=", 30]
                ],
                "selections": [
                    "SUEP_nconst_HighestPT >= 10",
                    "PuppiMET_pt > 30",
                    "W_pt > 60",
                    "W_mt < 130",
                    "W_mt > 30",
                    "bjetSel == 1",
                    "deltaPhi_SUEP_W > 1.5",
                    "deltaPhi_SUEP_MET > 1.5",
                    "deltaPhi_lepton_SUEP > 1.5",
                    "ak4jets_inSUEPcluster_n_HighestPT >= 1",
                    "W_SUEP_BV < 2",
                    "deltaPhi_minDeltaPhiMETJet_MET > 1.5",
                    "SUEP_S1_HighestPT > 0.3",
                ],
                "syst":  [
                    "puweights_up",
                    "puweights_down",
                    "PSWeight_ISR_up",
                    "PSWeight_ISR_down",
                    "PSWeight_FSR_up",
                    "PSWeight_FSR_down",
                    "higgs_weights_up",
                    "higgs_weights_down",
                    "LepSFElUp",
                    "LepSFElDown",
                    "LepSFMuUp",
                    "LepSFMuDown",
                    "bTagWeight_HFcorrelated_Up",
                    "bTagWeight_HFcorrelated_Dn",
                    "bTagWeight_HFuncorrelated_Up",
                    "bTagWeight_HFuncorrelated_Dn",
                    "bTagWeight_LFcorrelated_Up",
                    "bTagWeight_LFcorrelated_Dn",
                    "bTagWeight_LFuncorrelated_Up",
                    "bTagWeight_LFuncorrelated_Dn",
                    "prefire_up",
                    "prefire_down",
                ],
             },
            "CRWJ": {
                "input_method": "HighestPT",
                "method_var": "SUEP_nconst_HighestPT",
                "SR": [
                    ["SUEP_S1_HighestPT", ">=", 2.0],
                    ["SUEP_nconst_HighestPT", ">=", 1000]
                ],
                "selections": [
                    "SUEP_nconst_HighestPT >= 10",
                    "PuppiMET_pt > 30",
                    "W_pt > 60",
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
                ],
                "syst": [
                    "puweights_up",
                    "puweights_down",
                    "PSWeight_ISR_up",
                    "PSWeight_ISR_down",
                    "PSWeight_FSR_up",
                    "PSWeight_FSR_down",
                    "higgs_weights_up",
                    "higgs_weights_down",
                    "LepSFElUp",
                    "LepSFElDown",
                    "LepSFMuUp",
                    "LepSFMuDown",
                    "bTagWeight_HFcorrelated_Up",
                    "bTagWeight_HFcorrelated_Dn",
                    "bTagWeight_HFuncorrelated_Up",
                    "bTagWeight_HFuncorrelated_Dn",
                    "bTagWeight_LFcorrelated_Up",
                    "bTagWeight_LFcorrelated_Dn",
                    "bTagWeight_LFuncorrelated_Up",
                    "bTagWeight_LFuncorrelated_Dn",
                    "prefire_up",
                    "prefire_down",
                ],
            }
        }

        if options.get("isMC") and options.get("doSyst"):

            # add systematic variations, which are identical, except for the dataframe name
            variations = [
                "MuScaleUp",
                "MuScaleDown",
                "track_down"
            ]
            var_config = {}
            for var in variations:
                for tag, config_tag in config.items():
                    _var_config = deepcopy(config_tag)
                    _var_config["syst"] = []    # don't need to run SFs for systematic variations
                    _var_config["df_name"] = "vars_" + var # name of the dataframe in the hdf5 ntuple
                    var_config.update({tag + "_" + var: _var_config})

            # systematic variations that change the MET and W selections
            met_variations = [
                "JER_up",
                "JER_down",
                "JES_up",
                "JES_down",
            ]
            for var in met_variations:
                for tag, config_tag in config.items():
                    _var_config = deepcopy(config_tag)
                    _var_config["syst"] = []
                    # add the MET variation to the selection name, variables are defined in var_defs
                    for iSel in range(len(_var_config["selections"])):
                        s = _var_config["selections"][iSel]
                        if "MET_" in s or "W_" in s:
                            s = s.split(" ")
                            s[0] = s[0] + "_" + var
                            s = " ".join(s)
                            _var_config["selections"][iSel] = s
                        var_config.update({tag + "_" + var: _var_config})
                    
            config.update(var_config)

    elif options['channel'] == 'WH-VRGJ':
        config = {
            # "VRGJnoB": {
            #    "input_method": "HighestPT",
            #    "method_var": "SUEP_nconst_HighestPT",
            #    "selections": [
            #        "SUEP_nconst_HighestPT >= 10",
            #        "minDeltaPhiJetPhoton > 1.5",
            #        "deltaPhi_SUEP_photon > 1.5",
            #        "ak4jets_inSUEPcluster_n_HighestPT >= 1",
            #        "photon_SUEP_BV < 2",
            #        "SUEP_S1_HighestPT < 0.3",
            #    ],
            # },
            "VRGJlowS": {
               "input_method": "HighestPT",
               "method_var": "SUEP_nconst_HighestPT",
               "selections": [
                   "SUEP_nconst_HighestPT >= 10",
                   "bjetSel == 1",
                   "minDeltaPhiJetPhoton > 1.5",
                   "deltaPhi_SUEP_photon > 1.5",
                   "ak4jets_inSUEPcluster_n_HighestPT >= 1",
                   "photon_SUEP_BV < 2",
                   "SUEP_S1_HighestPT < 0.3",
               ],
            },
            "VRGJhighS": {
               "input_method": "HighestPT",
               "method_var": "SUEP_nconst_HighestPT",
               "selections": [
                   "SUEP_nconst_HighestPT >= 10",
                   "bjetSel == 1",
                   "minDeltaPhiJetPhoton > 1.5",
                   "deltaPhi_SUEP_photon > 1.5",
                   "ak4jets_inSUEPcluster_n_HighestPT >= 1",
                   "photon_SUEP_BV < 2",
                   "SUEP_S1_HighestPT > 0.3",
               ],
            },
        }
    
    hists = {}
    for output_method in config.keys():
        var_defs.initialize_new_variables(
            output_method, SimpleNamespace(**options), config[output_method]
        )

    # run the SUEP histogram maker
    histmaker = SUEPDaskHistMaker(config=config, options=options, hists=hists)
    if options["client"] == "local":
        client = histmaker.setupLocalClient(options["nworkers"])
    else:
        client = histmaker.setupSlurmClient(
            n_workers=options["nworkers"], min_workers=2, max_workers=200
        )
    histmaker.run(client, options["samples"])
    client.close()


if __name__ == "__main__":
    main()
