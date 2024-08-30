import vector
import awkward as ak
import numpy as np

def initialize_new_variables(label: str, options, config: dict):
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
                balancing_var,
                ["W_pt_PuppiMET", "SUEP_pt_HighestPT"],
            ],
            [
                "W_jet1_BV",
                balancing_var,
                ["W_pt_PuppiMET", "jet1_pt"],
            ],
            [
                "ak4SUEP1_SUEP_BV",
                balancing_var,
                ["ak4jet1_inSUEPcluster_pt_HighestPT", "SUEP_pt_HighestPT"],
            ],
            [
                "W_SUEP_vBV",
                vector_balancing_var,
                [
                    "W_phi_PuppiMET",
                    "SUEP_phi_HighestPT",
                    "W_pt_PuppiMET",
                    "SUEP_pt_HighestPT",
                ],
            ],
            [
                "W_jet1_vBV",
                vector_balancing_var,
                ["W_phi_PuppiMET", "jet1_phi", "W_pt_PuppiMET", "jet1_pt"],
            ],
            [
                "deltaPhi_SUEP_W",
                deltaPhi_x_y,
                [
                    "SUEP_phi_HighestPT",
                    "W_phi_PuppiMET",
                ],
            ],
            [
                "deltaPhi_SUEP_MET",
                deltaPhi_x_y,
                [
                    "SUEP_phi_HighestPT",
                    "WH_MET_phi",
                ],
            ],
            [
                "deltaPhi_lepton_MET",
                deltaPhi_x_y,
                ["lepton_phi", "WH_MET_phi"],
            ],
            [
                "deltaPhi_lepton_SUEP",
                deltaPhi_x_y,
                [
                    "lepton_phi",
                    "SUEP_phi_HighestPT",
                ],
            ],
            [
                "deltaPhi_minDeltaPhiMETJet_SUEP",
                deltaPhi_x_y,
                [
                    "minDeltaPhiMETJet_phi",
                    "SUEP_phi_HighestPT",
                ],
            ],
            [
                "deltaPhi_minDeltaPhiMETJet_MET",
                deltaPhi_x_y,
                [
                    "minDeltaPhiMETJet_phi",
                    "WH_MET_phi",
                ],
            ],
            [
                "deltaPhi_minDeltaPhiMETJet_METUnc",
                deltaPhi_x_y,
                [
                    "minDeltaPhiMETJet_phi",
                    "MET_phi",
                ],
            ],
            [
                "deltaPhi_SUEP_jet1",
                deltaPhi_x_y,
                [
                    "SUEP_phi_HighestPT",
                    "jet1_phi",
                ],
            ],
            [
                "deltaPhi_SUEP_bjet",
                deltaPhi_x_y,
                [
                    "SUEP_phi_HighestPT",
                    "bjet_phi",
                ],
            ],
            [
                "deltaPhi_jet1_bjet",
                deltaPhi_x_y,
                ["jet1_phi", "bjet_phi"],
            ],
            [
                "deltaPhi_lepton_bjet",
                deltaPhi_x_y,
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
                balancing_var,
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
                deltaPhi_x_y,
                ["SUEP_genPhi", "SUEP_phi_HighestPT"],
            ],
            [
                "deltaR_genSUEP_SUEP",
                deltaR,
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
                deltaPhi_x_y,
                ["SUEP_genPhi", "MaxConstAK15_phi"],
            ],
            [
                "deltaR_SUEPgen_MaxConstAK15",
                deltaR,
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
            ["new_W_pt_PuppiMET", lambda lepton_phi, MET_phi, lepton_pt, MET_pt: calc_vector_sum_pt(lepton_phi, MET_phi, lepton_pt, MET_pt), ["lepton_phi", "PuppiMET_phi", "lepton_pt", "PuppiMET_pt"]],
            ["new_W_pt_PFMET", lambda lepton_phi, MET_phi, lepton_pt, MET_pt: calc_vector_sum_pt(lepton_phi, MET_phi, lepton_pt, MET_pt), ["lepton_phi", "MET_phi", "lepton_pt", "MET_pt"]],
            ["new_W_mt_PuppiMET", lambda lepton_phi, MET_phi, lepton_pt, MET_pt: calc_mt(lepton_phi, MET_phi, lepton_pt, MET_pt), ["lepton_phi", "PuppiMET_phi", "lepton_pt", "PuppiMET_pt"]],
            ["new_W_mt_PFMET", lambda lepton_phi, MET_phi, lepton_pt, MET_pt: calc_mt(lepton_phi, MET_phi, lepton_pt, MET_pt), ["lepton_phi", "MET_phi", "lepton_pt", "MET_pt"]],
            ["new_W_phi_PuppiMET", lambda lepton_phi, MET_phi, lepton_pt, MET_pt: calc_vector_sum_phi(lepton_phi, MET_phi, lepton_pt, MET_pt), ["lepton_phi", "PuppiMET_phi", "lepton_pt", "PuppiMET_pt"]],
            ["new_W_phi_PFMET", lambda lepton_phi, MET_phi, lepton_pt, MET_pt: calc_vector_sum_phi(lepton_phi, MET_phi, lepton_pt, MET_pt), ["lepton_phi", "MET_phi", "lepton_pt", "MET_pt"]],
            [
                "deltaPhi_SUEP_PuppiMET",
                deltaPhi_x_y,
                [
                    "SUEP_phi_HighestPT",
                    "PuppiMET_phi",
                ],
            ],
            [
                "deltaPhi_SUEP_PFMET",
                deltaPhi_x_y,
                [
                    "SUEP_phi_HighestPT",
                    "MET_phi",
                ],
            ],
            [
                "deltaPhi_SUEP_W_PuppiMET",
                deltaPhi_x_y,
                [
                    "SUEP_phi_HighestPT",
                    "new_W_phi_PuppiMET",
                ],
            ],
            [
                "deltaPhi_SUEP_W_PFMET",
                deltaPhi_x_y,
                [
                    "SUEP_phi_HighestPT",
                    "new_W_phi_PFMET",
                ],
            ],
            [
                "W_PFMET_SUEP_BV",
                balancing_var,
                ["new_W_pt_PFMET", "SUEP_pt_HighestPT"],
            ],
        ]
        if options.isMC:
            new_vars += [
                ["deltaPhi_W_genW", deltaPhi_x_y, ["genW_phi", "W_phi"]],
                ["deltaPt_W_genW", lambda x, y: x - y, ["genW_pt", "W_pt"]],
            ]

    config['new_variables'] = new_vars

def deltaPhi_x_y(xphi, yphi):

    # cast inputs to numpy arrays
    yphi = np.array(yphi)

    x_v = vector.arr({"pt": np.ones(len(xphi)), "phi": xphi})
    y_v = vector.arr({"pt": np.ones(len(yphi)), "phi": yphi})

    signed_dphi = x_v.deltaphi(y_v)
    abs_dphi = np.abs(signed_dphi.tolist())

    # deal with the cases where phi was initialized to a moot value like -999
    abs_dphi[xphi > 2 * np.pi] = -999
    abs_dphi[yphi > 2 * np.pi] = -999
    abs_dphi[xphi < -2 * np.pi] = -999
    abs_dphi[yphi < -2 * np.pi] = -999

    return abs_dphi


def deltaR(xEta, yEta, xPhi, yPhi):

    # cast inputs to numpy arrays
    xEta = np.array(xEta)
    yEta = np.array(yEta)
    xPhi = np.array(xPhi)
    yPhi = np.array(yPhi)

    x_v = vector.arr({"eta": xEta, "phi": xPhi, "pt": np.ones(len(xEta))})
    y_v = vector.arr({"eta": yEta, "phi": yPhi, "pt": np.ones(len(yEta))})

    dR = x_v.deltaR(y_v)

    if type(dR) is ak.highlevel.Array:
        dR = dR.to_numpy()

    # deal with the cases where eta and phi were initialized to a moot value like -999
    dR[xEta < -100] = -999
    dR[yEta < -100] = -999
    dR[xPhi < -100] = -999
    dR[yPhi < -100] = -999

    return dR


def balancing_var(xpt, ypt):

    # cast inputs to numpy arrays
    xpt = np.array(xpt)
    ypt = np.array(ypt)

    var = np.where(ypt > 0, (xpt - ypt) / ypt, np.ones(len(xpt)) * -999)

    # deal with the cases where pt was initialized to a moot value, and set it to a moot value of -999
    var[xpt < 0] = -999
    var[ypt < 0] = -999

    return var


def calc_vector_sum(xphi, yphi, xpt, ypt):

    # cast inputs to numpy arrays
    xpt = np.array(xpt)
    ypt = np.array(ypt)
    xphi = np.array(xphi)
    yphi = np.array(yphi)

    x_v = vector.arr({"pt": xpt, "phi": xphi})
    y_v = vector.arr({"pt": ypt, "phi": yphi})

    return x_v + y_v


def calc_vector_sum_pt(xphi, yphi, xpt, ypt):

    vector_sum = calc_vector_sum(xphi, yphi, xpt, ypt)
    vector_sum_pt = vector_sum.pt

    if type(vector_sum_pt) is ak.highlevel.Array:
        vector_sum_pt = vector_sum_pt.to_numpy()

    # deal with the cases where pt was initialized to a moot value, and set it to a moot value of -999
    vector_sum_pt[xpt < 0] = -999
    vector_sum_pt[ypt < 0] = -999

    return vector_sum_pt


def calc_vector_sum_phi(xphi, yphi, xpt, ypt):

    vector_sum = calc_vector_sum(xphi, yphi, xpt, ypt)
    vector_sum_phi = vector_sum.phi

    if type(vector_sum_phi) is ak.highlevel.Array:
        vector_sum_phi = vector_sum_phi.to_numpy()

    # deal with the cases where pt was initialized to a moot value, and set it to a moot value of -999
    vector_sum_phi[xpt < 0] = -999
    vector_sum_phi[ypt < 0] = -999

    return vector_sum_phi


def vector_balancing_var(xphi, yphi, xpt, ypt):

    vector_sum_pt = calc_vector_sum_pt(xphi, yphi, xpt, ypt)

    var = np.where(ypt > 0, vector_sum_pt / ypt, np.ones(len(xpt)) * -999)

    # deal with the cases where pt was initialized to a moot value, and set it to a moot value of -999
    var[xpt < 0] = -999
    var[ypt < 0] = -999

    return var

def calc_mt(xphi, yphi, xpt, ypt):

    # cast inputs to numpy arrays
    xpt = np.array(xpt)
    ypt = np.array(ypt)
    xphi = np.array(xphi)
    yphi = np.array(yphi)

    x_v = vector.arr({"pt": xpt, "phi": xphi})
    y_v = vector.arr({"pt": ypt, "phi": yphi})

    return np.sqrt(2 * xpt * ypt * (1 - np.cos(x_v.deltaphi(y_v))))