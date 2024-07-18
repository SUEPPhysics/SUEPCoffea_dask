import fill_utils
import numpy as np

def initialize_new_variables(label: str, options, config: dict) -> list:
    """
    Form a list of new variables in form:
    [
        [name, function, [input_vars]],
        ...
    ]
    and adds it to the config dictionary, if not alreayd there
    """

    if 'new_variables' in config.keys():
        return

    new_vars = []

    if options.channel == 'WH':

        new_vars += [
            [
                "bjetSel",
                lambda x, y: ((x == 0) & (y < 2)),
                ["nBTight", "nBLoose"],
            ],
            [
                "W_SUEP_BV",
                fill_utils.balancing_var,
                ["W_pt_PuppiMET", "SUEP_pt_HighestPT"],
            ],
            [
                "W_jet1_BV",
                fill_utils.balancing_var,
                ["W_pt_PuppiMET", "jet1_pt"],
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
                    "W_phi_PuppiMET",
                    "SUEP_phi_HighestPT",
                    "W_pt_PuppiMET",
                    "SUEP_pt_HighestPT",
                ],
            ],
            [
                "W_SUEP_vBV2",
                fill_utils.vector_balancing_var2,
                [
                    "W_phi_PuppiMET",
                    "SUEP_phi_HighestPT",
                    "W_pt_PuppiMET",
                    "SUEP_pt_HighestPT",
                ],
            ],
            [
                "W_jet1_vBV",
                fill_utils.vector_balancing_var,
                ["W_phi_PuppiMET", "jet1_phi", "W_pt_PuppiMET", "jet1_pt"],
            ],
            [
                "deltaPhi_SUEP_W",
                fill_utils.deltaPhi_x_y,
                [
                    "SUEP_phi_HighestPT",
                    "W_phi_PuppiMET",
                ],
            ],
            [
                "deltaPhi_SUEP_MET",
                fill_utils.deltaPhi_x_y,
                [
                    "SUEP_phi_HighestPT",
                    "PuppiMET_phi",
                ],
            ],
            [
                "deltaPhi_SUEP_METUnc",
                fill_utils.deltaPhi_x_y,
                [
                    "SUEP_phi_HighestPT",
                    "MET_phi",
                ],
            ],
            [
                "deltaPhi_lepton_MET",
                fill_utils.deltaPhi_x_y,
                ["lepton_phi", "MET_JEC_phi"],
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
                    "MET_JEC_phi",
                ],
            ],
            [
                "deltaPhi_minDeltaPhiMETJet_METUnc",
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
            new_vars += [
                ["deltaPhi_W_genW", fill_utils.deltaPhi_x_y, ["genW_phi", "W_phi"]],
                ["deltaPt_W_genW", lambda x, y: x - y, ["genW_pt", "W_pt"]],
            ]

    config['new_variables'] = new_vars