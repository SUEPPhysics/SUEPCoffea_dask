import awkward as ak
import numpy as np
import vector


def initialize_new_variables(label: str, options, config: dict):
    """
    Form a list of new variables in form:
    [
        [name, function, [input_vars]],
        ...
    ]
    and adds it to the config dictionary, if not already there
    """

    if "new_variables" in config.keys():
        return

    new_vars = []

    if options.channel in ["WH", "WH-VRGJ"]:
        new_vars += [
            [
                "bjetSel",
                lambda x, y: ((x == 0) & (y < 2)),
                ["nBTight", "nBLoose"],
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
                "nak4jets_outsideSUEP",
                lambda x, y: (x - y),
                ["ngood_ak4jets", "ak4jets_inSUEPcluster_n_HighestPT"],
            ],
            [
                "deltaPhi_ak4jet1_inSUEPcluster_SUEP",
                deltaPhi_x_y,
                [
                    "ak4jet1_inSUEPcluster_phi_HighestPT",
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
            # [
            #     "SUEPMostNumerous",
            #     lambda x, y: x > y,
            #     ["SUEP_nconst_HighestPT", "otherAK15_maxConst_nconst_HighestPT"],
            # ],
        ]

        if options.isMC:

            new_vars += [
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
                    'SUEP_genSUEP_BV',
                    balancing_var,
                    ['SUEP_pt_HighestPT', 'SUEP_genPt']
                ],
                [
                    "percent_darkphis_inTracker",
                    lambda x, y: x / y,
                    ["n_darkphis_inTracker", "n_darkphis"],
                ],
            ]

        if options.channel == "WH":
            
            new_vars += [
                [
                    "W_SUEP_BV",
                    balancing_var,
                    ["W_pt", "SUEP_pt_HighestPT"],
                ],
                [
                    "SUEP_W_BV",
                    balancing_var,
                    ["SUEP_pt_HighestPT", "W_pt"],
                ],
                [
                    "W_SUEP_vBV",
                    vector_balancing_var,
                    [
                        "W_phi",
                        "W_pt",
                        "SUEP_phi_HighestPT",
                        "SUEP_pt_HighestPT",
                    ],
                ],
                [
                    "ak4jet1_inSUEPcluster_W_BV",
                    balancing_var,
                    ["ak4jet1_inSUEPcluster_pt_HighestPT", "W_pt"],
                ],
                [
                    "W_ak4jet1_inSUEPcluster_vBV",
                    vector_balancing_var,
                    ["W_phi", "W_pt", "ak4jet1_inSUEPcluster_phi_HighestPT", "ak4jet1_inSUEPcluster_pt_HighestPT"],
                ],

                [
                    "sumAK4W_pt",
                    calc_vector_sum_pt,
                    [
                        "jet1_phi",
                        "jet1_pt",
                        "jet2_phi",
                        "jet2_pt",
                        "jet3_phi",
                        "jet3_pt",
                        "W_phi",
                        "W_pt",
                    ],
                ],
                [
                    "sumAK4W_W_BV",
                    vector_balancing_var,
                    [
                        "jet1_phi",
                        "jet1_pt",
                        "jet2_phi",
                        "jet2_pt",
                        "jet3_phi",
                        "jet3_pt",
                        "W_phi",
                        "W_pt",
                    ],
                ],
                [
                    "deltaPhi_SUEP_W",
                    deltaPhi_x_y,
                    [
                        "SUEP_phi_HighestPT",
                        "W_phi",
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
                # [
                #     'deltaPhi_ak4jet1_outsideSUEPcluster_SUEP',
                #     deltaPhi_x_y,
                #     [
                #         'ak4jet1_outsideSUEPcluster_phi_HighestPT',
                #         'SUEP_phi_HighestPT'
                #     ]
                # ],
                ["isMuon", lambda x: abs(x) == 13, ["lepton_flavor"]],
                ["isElectron", lambda x: abs(x) == 11, ["lepton_flavor"]],
                # ["new_W_pt_PuppiMET", lambda lepton_phi, MET_phi, lepton_pt, MET_pt: calc_vector_sum_pt(lepton_phi, MET_phi, lepton_pt, MET_pt), ["lepton_phi", "PuppiMET_phi", "lepton_pt", "PuppiMET_pt"]],
                # ["new_W_pt_PFMET", lambda lepton_phi, MET_phi, lepton_pt, MET_pt: calc_vector_sum_pt(lepton_phi, MET_phi, lepton_pt, MET_pt), ["lepton_phi", "MET_phi", "lepton_pt", "MET_pt"]],
                # ["new_W_mt_PuppiMET", lambda lepton_phi, MET_phi, lepton_pt, MET_pt: calc_mt(lepton_phi, MET_phi, lepton_pt, MET_pt), ["lepton_phi", "PuppiMET_phi", "lepton_pt", "PuppiMET_pt"]],
                # ["new_W_mt_PFMET", lambda lepton_phi, MET_phi, lepton_pt, MET_pt: calc_mt(lepton_phi, MET_phi, lepton_pt, MET_pt), ["lepton_phi", "MET_phi", "lepton_pt", "MET_pt"]],
                # ["new_W_phi_PuppiMET", lambda lepton_phi, MET_phi, lepton_pt, MET_pt: calc_vector_sum_phi(lepton_phi, MET_phi, lepton_pt, MET_pt), ["lepton_phi", "PuppiMET_phi", "lepton_pt", "PuppiMET_pt"]],
                # ["new_W_phi_PFMET", lambda lepton_phi, MET_phi, lepton_pt, MET_pt: calc_vector_sum_phi(lepton_phi, MET_phi, lepton_pt, MET_pt), ["lepton_phi", "MET_phi", "lepton_pt", "MET_pt"]],
                # [
                #     "deltaPhi_SUEP_PuppiMET",
                #     deltaPhi_x_y,
                #     [
                #         "SUEP_phi_HighestPT",
                #         "PuppiMET_phi",
                #     ],
                # ],
                # [
                #     "deltaPhi_SUEP_PFMET",
                #     deltaPhi_x_y,
                #     [
                #         "SUEP_phi_HighestPT",
                #         "MET_phi",
                #     ],
                # ],
                # [
                #     "deltaPhi_SUEP_W_PuppiMET",
                #     deltaPhi_x_y,
                #     [
                #         "SUEP_phi_HighestPT",
                #         "new_W_phi_PuppiMET",
                #     ],
                # ],
                # [
                #     "deltaPhi_SUEP_W_PFMET",
                #     deltaPhi_x_y,
                #     [
                #         "SUEP_phi_HighestPT",
                #         "new_W_phi_PFMET",
                #     ],
                # ],
                # [
                #     "W_PFMET_SUEP_BV",
                #     balancing_var,
                #     ["new_W_pt_PFMET", "SUEP_pt_HighestPT"],
                # ]
            ]
            if options.isMC:
                new_vars += [
                    #["deltaPhi_W_genW", deltaPhi_x_y, ["genW_phi", "W_phi"]],
                    #["deltaPt_W_genW", lambda x, y: x - y, ["genW_pt", "W_pt"]],
                    ["W_genW_BV", balancing_var, ["W_pt", "genW_pt"]],
                    #['LepSFMu', lambda x, y: x * y, ['LepSF', 'isMuon']],
                    #['LepSFEl', lambda x, y: x * y, ['LepSF', 'isElectron']],
                ]

        if options.channel == "WH-VRGJ":

            new_vars += [
                [
                    "photon_WP80",
                    lambda mvaID, isScEtaEB, isScEtaEE: ((mvaID > 0.42) & isScEtaEB) | ((mvaID > 0.14) & isScEtaEE),
                    ["photon_mvaID", "photon_isScEtaEB", "photon_isScEtaEE"],
                ],
                [
                    "gammaTriggerSel",
                    gammaTriggerSel,
                    ["photon_pt", "WH_gammaTriggerBits"],
                ],
                [
                    "photon_SUEP_BV",
                    balancing_var,
                    ["photon_pt", "SUEP_pt_HighestPT"],
                ],
                [
                    "ak4jet1_inSUEPcluster_photon_BV",
                    balancing_var,
                    ["ak4jet1_inSUEPcluster_pt_HighestPT", "photon_pt"],
                ],
                [
                    "photon_ak4jet1_inSUEPcluster_vBV",
                    vector_balancing_var,
                    [
                        "photon_phi",
                        "photon_pt",
                        "ak4jet1_inSUEPcluster_phi_HighestPT",
                        "ak4jet1_inSUEPcluster_pt_HighestPT",
                    ],
                ],
                [
                    "deltaPhi_SUEP_photon",
                    deltaPhi_x_y,
                    [
                        "SUEP_phi_HighestPT",
                        "photon_phi",
                    ],
                ],
                [
                    "SUEP_photon_BV",
                    balancing_var,
                    ["SUEP_pt_HighestPT", "photon_pt"],
                ],
                [
                    "deltaPhi_photon_looseNotTightLepton",
                    deltaPhi_x_y,
                    [
                        "photon_phi",
                        "looseNotTightLepton1_phi",
                    ],
                ],
                [
                    "deltaPhi_photon_looseNotTightHardLepton",
                    deltaPhi_x_y_pTReq(50),
                    [
                        "photon_phi",
                        "looseNotTightLepton1_phi",
                        "looseNotTightLepton1_pt",
                    ],
                ],
                [
                    "deltaPhi_photon_hardMET",
                    deltaPhi_x_y_pTReq(50),
                    [
                        "photon_phi",
                        "WH_MET_phi",
                        "WH_MET_pt",
                    ],
                ],
                [
                    "deltaPhi_photon_MET",
                    deltaPhi_x_y,
                    [
                        "photon_phi",
                        "WH_MET_phi",
                    ],
                ],
                [
                    "sumAK4PhotonMET_pt",
                    calc_vector_sum_pt,
                    [
                        "jet1_phi",
                        "jet1_pt",
                        "jet2_phi",
                        "jet2_pt",
                        "jet3_phi",
                        "jet3_pt",
                        "photon_phi",
                        "photon_pt",
                        "WH_MET_phi",
                        "WH_MET_pt",
                    ],
                ],
                [
                    "sumAK4PhotonMET_photon_BV",
                    vector_balancing_var,
                    [
                        "jet1_phi",
                        "jet1_pt",
                        "jet2_phi",
                        "jet2_pt",
                        "jet3_phi",
                        "jet3_pt",
                        "photon_phi",
                        "photon_pt",
                        "WH_MET_phi",
                        "WH_MET_pt",
                    ],
                ]
            ]

        # deal with MET variations
        _met_variations = ['JER_up', 'JER_down', 'JES_up', 'JES_down', 'Unclustered_up', 'Unclustered_down'] 
        for _var in _met_variations:
            if _var not in label: continue
            new_vars += [
                [
                    f"W_pt_{_var}",
                    calc_vector_sum_pt,
                    ["lepton_phi", "lepton_pt", f"PuppiMET_phi_{_var}", f"PuppiMET_pt_{_var}"],
                ],
                [
                    f"W_phi_{_var}",
                    calc_vector_sum_phi,
                    ["lepton_phi", "lepton_pt", f"PuppiMET_phi_{_var}", f"PuppiMET_pt_{_var}"],
                ],
                [
                    f"W_mt_{_var}",
                    calc_mt,
                    ["lepton_phi", "lepton_pt", f"PuppiMET_phi_{_var}", f"PuppiMET_pt_{_var}"],
                ],
                [
                    f"deltaPhi_SUEP_MET_{_var}",
                    deltaPhi_x_y,
                    [
                        "SUEP_phi_HighestPT",
                        f"PuppiMET_phi_{_var}",
                    ],
                ],
                [
                    f"deltaPhi_SUEP_W_MET_{_var}",
                    deltaPhi_x_y,
                    [
                        "SUEP_phi_HighestPT",
                        f"W_phi_{_var}",
                    ],
                ],
                [
                    f"W_SUEP_BV_{_var}",
                    balancing_var,
                    [f"W_pt_{_var}", "SUEP_pt_HighestPT"],
                ]
            ]

    config["new_variables"] = new_vars


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

def deltaPhi_x_y_pTReq(ypt_threshold):

    def deltaPhi_x_y_pTReq_inner(xphi, yphi, ypt):

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

        # set values to -999 if ypt < 50
        abs_dphi[ypt < ypt_threshold] = -999

        return abs_dphi
    
    return deltaPhi_x_y_pTReq_inner


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

    _eps = 1e-10
    var = np.where(ypt > 0, (xpt - ypt) / (ypt + _eps), np.ones(len(xpt)) * -999)

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


def calc_vector_sum_pt(*args):

    if len(args) % 2 != 0:
        raise ValueError("Arguments must be in pairs of (phi, pt)")

    vectors = []
    for i in range(0, len(args), 2):
        phi = np.array(args[i])
        pt = np.array(args[i + 1])
        vectors.append(vector.arr({"pt": pt, "phi": phi}))

    vector_sum = vectors[0]
    for vec in vectors[1:]:
        vector_sum = vector_sum + vec
    vector_sum_pt = vector_sum.pt

    if type(vector_sum_pt) is ak.highlevel.Array:
        vector_sum_pt = vector_sum_pt.to_numpy()

    # deal with the cases where pt was initialized to a moot value, and set it to a moot value of -999
    for i in range(0, len(args), 2):
        pt = np.array(args[i + 1])
        vector_sum_pt[pt < 0] = -999

    return vector_sum_pt


def calc_vector_sum_phi(*args):

    if len(args) % 2 != 0:
        raise ValueError("Arguments must be in pairs of (phi, pt)")

    vectors = []
    for i in range(0, len(args), 2):
        phi = np.array(args[i])
        pt = np.array(args[i + 1])
        vectors.append(vector.arr({"pt": pt, "phi": phi}))

    vector_sum = vectors[0]
    for vec in vectors[1:]:
        vector_sum = vector_sum + vec
    vector_sum_phi = vector_sum.phi

    if type(vector_sum_phi) is ak.highlevel.Array:
        vector_sum_phi = vector_sum_phi.to_numpy()

    # deal with the cases where pt was initialized to a moot value, and set it to a moot value of -999
    for i in range(0, len(args), 2):
        pt = np.array(args[i + 1])
        vector_sum_phi[pt < 0] = -999

    return vector_sum_phi


def vector_balancing_var(*args):

    if len(args) % 2 != 0:
        raise ValueError("Arguments must be in pairs of (phi, pt)")

    vector_sum_pt = calc_vector_sum_pt(*args)

    ypt = np.array(args[-1])  # Assuming ypt is the last pt argument

    _eps = 1e-10
    var = np.where(ypt > 0, vector_sum_pt / (ypt + _eps), np.ones(len(ypt)) * -999)

    # deal with the cases where pt was initialized to a moot value, and set it to a moot value of -999
    for i in range(1, len(args), 2):
        pt = np.array(args[i])
        var[pt < 0] = -999

    return var


def calc_mt(xphi, xpt, yphi, ypt):

    # cast inputs to numpy arrays
    xpt = np.array(xpt)
    ypt = np.array(ypt)
    xphi = np.array(xphi)
    yphi = np.array(yphi)

    x_v = vector.arr({"pt": xpt, "phi": xphi})
    y_v = vector.arr({"pt": ypt, "phi": yphi})

    return np.sqrt(2 * xpt * ypt * (1 - np.cos(x_v.deltaphi(y_v))))


def gammaTriggerSel(photon_pt, bits):

    photon200 = (bits % 2) == 1
    return (photon_pt > 235) & photon200
