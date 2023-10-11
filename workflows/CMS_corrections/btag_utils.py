import awkward as ak
import correctionlib
import numpy as np
import pickle5 as pickle


def doBTagWeights(events, jetsPre, era, wp="L", do_syst=False):
    jetsPre = jetsPre[jetsPre.pt >= 30]
    jets, njets = ak.flatten(jetsPre), np.array(ak.num(jetsPre))
    hadronFlavourLightAsB = np.array(
        np.where(jets.hadronFlavour == 0, 5, jets.hadronFlavour)
    )
    hadronFlavourBCAsLight = np.array(
        np.where(jets.hadronFlavour != 0, 0, jets.hadronFlavour)
    )
    if era == 2015:
        btagfile = "data/BTagUL16APV/btagging.json.gz"
        btagfileL = "data/BTagUL16/btagging.json.gz"
    if era == 2016:
        btagfile = "data/BTagUL16/btagging.json.gz"
        btagfileL = btagfile
    if era == 2017:
        btagfile = "data/BTagUL17/btagging.json.gz"
        btagfileL = btagfile
    if era == 2018:
        btagfile = "data/BTagUL18/btagging.json.gz"
        btagfileL = btagfile
    corrector = correctionlib.CorrectionSet.from_file(btagfile)
    correctorL = correctionlib.CorrectionSet.from_file(btagfileL)
    SF = {}
    flattened_pt = np.array(jets.pt)
    flattened_eta = np.array(np.abs(jets.eta))
    SF["central"] = corrector["deepJet_comb"].evaluate(
        "central", wp, hadronFlavourLightAsB, flattened_eta, flattened_pt
    )  # SF per jet, later argument is dummy as will be overwritten next
    SF["central"] = np.where(
        abs(jets.hadronFlavour) == 0,
        correctorL["deepJet_incl"].evaluate(
            "central", wp, hadronFlavourBCAsLight, flattened_eta, flattened_pt
        ),
        SF["central"],
    )
    if do_syst:
        SF["HFcorrelated_Up"] = np.where(
            (abs(jets.hadronFlavour) == 4) | (abs(jets.hadronFlavour) == 5),
            corrector["deepJet_comb"].evaluate(
                "up_correlated", wp, hadronFlavourLightAsB, flattened_eta, flattened_pt
            ),
            SF["central"],
        )
        SF["HFcorrelated_Dn"] = np.where(
            (abs(jets.hadronFlavour) == 4) | (abs(jets.hadronFlavour) == 5),
            corrector["deepJet_comb"].evaluate(
                "down_correlated",
                wp,
                hadronFlavourLightAsB,
                flattened_eta,
                flattened_pt,
            ),
            SF["central"],
        )
        SF["HFuncorrelated_Up"] = np.where(
            (abs(jets.hadronFlavour) == 4) | (abs(jets.hadronFlavour) == 5),
            corrector["deepJet_comb"].evaluate(
                "up_uncorrelated",
                wp,
                hadronFlavourLightAsB,
                flattened_eta,
                flattened_pt,
            ),
            SF["central"],
        )
        SF["HFuncorrelated_Dn"] = np.where(
            (abs(jets.hadronFlavour) == 4) | (abs(jets.hadronFlavour) == 5),
            corrector["deepJet_comb"].evaluate(
                "down_uncorrelated",
                wp,
                hadronFlavourLightAsB,
                flattened_eta,
                flattened_pt,
            ),
            SF["central"],
        )
        SF["LFcorrelated_Up"] = np.where(
            abs(jets.hadronFlavour) == 0,
            correctorL["deepJet_incl"].evaluate(
                "up_correlated", wp, hadronFlavourBCAsLight, flattened_eta, flattened_pt
            ),
            SF["central"],
        )
        SF["LFcorrelated_Dn"] = np.where(
            abs(jets.hadronFlavour) == 0,
            correctorL["deepJet_incl"].evaluate(
                "down_correlated",
                wp,
                hadronFlavourBCAsLight,
                flattened_eta,
                flattened_pt,
            ),
            SF["central"],
        )
        SF["LFuncorrelated_Up"] = np.where(
            abs(jets.hadronFlavour) == 0,
            correctorL["deepJet_incl"].evaluate(
                "up_uncorrelated",
                wp,
                hadronFlavourBCAsLight,
                flattened_eta,
                flattened_pt,
            ),
            SF["central"],
        )
        SF["LFuncorrelated_Dn"] = np.where(
            abs(jets.hadronFlavour) == 0,
            correctorL["deepJet_incl"].evaluate(
                "down_uncorrelated",
                wp,
                hadronFlavourBCAsLight,
                flattened_eta,
                flattened_pt,
            ),
            SF["central"],
        )
    effs = getBTagEffs(events, jets, era, wp)
    wps = {"L": "Loose", "M": "Medium", "T": "Tight"}  # For safe conversion
    weights = {}
    effs = ak.unflatten(effs, njets)
    for key in SF:
        SF[key] = ak.unflatten(SF[key], njets)
    for (
        syst_var
    ) in (
        SF.keys()
    ):  # Method (1.a) here: https://twiki.cern.ch/twiki/bin/view/CMS/BTagSFMethods
        mceff = ak.prod(
            np.where(jetsPre.btag >= btagcuts(wps[wp], era), effs, 1 - effs), axis=1
        )
        dataeff = ak.prod(
            np.where(
                jetsPre.btag >= btagcuts(wps[wp], era),
                SF[syst_var] * effs,
                1 - SF[syst_var] * effs,
            ),
            axis=1,
        )
        weights[syst_var] = dataeff / mceff

    return weights


def getBTagEffs(events, jets, era, wp="L"):
    if wp != "L":
        print("Warning, efficiencies are computed for the Loose WP only!")
    if era == 2015:
        btagfile = "data/BTagUL16APV/eff.pickle"
    if era == 2016:
        btagfile = "data/BTagUL16/eff.pickle"
    if era == 2017:
        btagfile = "data/BTagUL17/eff.pickle"
    if era == 2018:
        btagfile = "data/BTagUL18/eff.pickle"
    bfile = open(btagfile, "rb")
    effsLoad = pickle.load(bfile)
    effs = effsLoad["L"](jets.pt, np.abs(jets.eta))
    effs = np.where(
        abs(jets.hadronFlavour) == 4, effsLoad["C"](jets.pt, np.abs(jets.eta)), effs
    )
    effs = np.where(
        abs(jets.hadronFlavour) == 5, effsLoad["B"](jets.pt, np.abs(jets.eta)), effs
    )
    return effs


def btagcuts(WP, era):
    if era == 2015:  # 2016APV
        if WP == "Loose":
            return 0.0480
        if WP == "Medium":
            return 0.2489
        if WP == "Tight":
            return 0.6377
    if era == 2016:
        if WP == "Loose":
            return 0.0508
        if WP == "Medium":
            return 0.2598
        if WP == "Tight":
            return 0.6502
    if era == 2017:
        if WP == "Loose":
            return 0.0532
        if WP == "Medium":
            return 0.3040
        if WP == "Tight":
            return 0.7476
    if era == 2018:
        if WP == "Loose":
            return 0.0490
        if WP == "Medium":
            return 0.2783
        if WP == "Tight":
            return 0.7100
