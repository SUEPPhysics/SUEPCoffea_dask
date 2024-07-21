import pickle

import awkward as ak
import correctionlib
import numpy as np


def doBTagWeights(jetsPre, era: int, wps: str, channel: str = 'zh', do_syst: bool = False) -> dict:
    """
    Compute btagging weights for a given jet collection.
    We support working points (wps): 'L', 'M', 'T', 'TL', 'LT',
    where the last two are the same, and represent a combination of Tight and Loose WPs.
    """
    if wps not in ('T', 'L', 'M', 'TL', 'LT'):
        raise ValueError(f"Invalid WP: {wp}")

    # some jet pre selection
    jetsPre = jetsPre[jetsPre.pt >= 30]
    jets, njets = ak.flatten(jetsPre), np.array(ak.num(jetsPre))
    hadronFlavourLightAsB = np.array(
        np.where(jets.hadronFlavour == 0, 5, jets.hadronFlavour)
    )
    hadronFlavourBCAsLight = np.array(
        np.where(jets.hadronFlavour != 0, 0, jets.hadronFlavour)
    )
    flattened_pt = np.array(jets.pt)
    flattened_eta = np.array(np.abs(jets.eta))

    # grab the correct weights based on the era
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

    # calculate SFs for each WP, variation
    SF = {}
    for wp in wps:
        SF[wp] = {}
        SF[wp]["central"] = corrector["deepJet_comb"].evaluate(
            "central", wp, hadronFlavourLightAsB, flattened_eta, flattened_pt
        )  # SF per jet, later argument is dummy as will be overwritten next
        SF[wp]["central"] = np.where(
            abs(jets.hadronFlavour) == 0,
            correctorL["deepJet_incl"].evaluate(
                "central", wp, hadronFlavourBCAsLight, flattened_eta, flattened_pt
            ),
            SF[wp]["central"],
        )
        if do_syst:
            SF[wp]["HFcorrelated_Up"] = np.where(
                (abs(jets.hadronFlavour) == 4) | (abs(jets.hadronFlavour) == 5),
                corrector["deepJet_comb"].evaluate(
                    "up_correlated", wp, hadronFlavourLightAsB, flattened_eta, flattened_pt
                ),
                SF[wp]["central"],
            )
            SF[wp]["HFcorrelated_Dn"] = np.where(
                (abs(jets.hadronFlavour) == 4) | (abs(jets.hadronFlavour) == 5),
                corrector["deepJet_comb"].evaluate(
                    "down_correlated",
                    wp,
                    hadronFlavourLightAsB,
                    flattened_eta,
                    flattened_pt,
                ),
                SF[wp]["central"],
            )
            SF[wp]["HFuncorrelated_Up"] = np.where(
                (abs(jets.hadronFlavour) == 4) | (abs(jets.hadronFlavour) == 5),
                corrector["deepJet_comb"].evaluate(
                    "up_uncorrelated",
                    wp,
                    hadronFlavourLightAsB,
                    flattened_eta,
                    flattened_pt,
                ),
                SF[wp]["central"],
            )
            SF[wp]["HFuncorrelated_Dn"] = np.where(
                (abs(jets.hadronFlavour) == 4) | (abs(jets.hadronFlavour) == 5),
                corrector["deepJet_comb"].evaluate(
                    "down_uncorrelated",
                    wp,
                    hadronFlavourLightAsB,
                    flattened_eta,
                    flattened_pt,
                ),
                SF[wp]["central"],
            )
            SF[wp]["LFcorrelated_Up"] = np.where(
                abs(jets.hadronFlavour) == 0,
                correctorL["deepJet_incl"].evaluate(
                    "up_correlated", wp, hadronFlavourBCAsLight, flattened_eta, flattened_pt
                ),
                SF[wp]["central"],
            )
            SF[wp]["LFcorrelated_Dn"] = np.where(
                abs(jets.hadronFlavour) == 0,
                correctorL["deepJet_incl"].evaluate(
                    "down_correlated",
                    wp,
                    hadronFlavourBCAsLight,
                    flattened_eta,
                    flattened_pt,
                ),
                SF[wp]["central"],
            )
            SF[wp]["LFuncorrelated_Up"] = np.where(
                abs(jets.hadronFlavour) == 0,
                correctorL["deepJet_incl"].evaluate(
                    "up_uncorrelated",
                    wp,
                    hadronFlavourBCAsLight,
                    flattened_eta,
                    flattened_pt,
                ),
                SF[wp]["central"],
            )
            SF[wp]["LFuncorrelated_Dn"] = np.where(
                abs(jets.hadronFlavour) == 0,
                correctorL["deepJet_incl"].evaluate(
                    "down_uncorrelated",
                    wp,
                    hadronFlavourBCAsLight,
                    flattened_eta,
                    flattened_pt,
                ),
                SF[wp]["central"],
            )

    # these are the efficiencies computed for each analysis and WP
    effs = {wp: getBTagEffs(jets, era, wp, channel) for wp in wps}
    for wp in effs:
        effs[wp] = ak.unflatten(effs[wp], njets)

    wps_name = {"L": "Loose", "M": "Medium", "T": "Tight"}  # For safe conversion
    weights = {}
    for wp in SF.keys():
        for syst in SF[wp].keys():
            SF[wp][syst] = ak.unflatten(SF[wp][syst], njets)

    # single WP: tight, loose, or medium
    if wps in ('T', 'L', 'M'):
        for (
            syst_var
        ) in (
            SF[wps].keys()
        ):  # Method (1.a) here: https://twiki.cern.ch/twiki/bin/view/CMS/BTagSFMethods
            mceff = ak.prod(
                np.where(jetsPre.btag >= btagcuts(wps_name[wps], era), effs[wps], 1 - effs[wps]), axis=1
            )
            dataeff = ak.prod(
                np.where(
                    jetsPre.btag >= btagcuts(wps_name[wps], era),
                    SF[wps][syst_var] * effs[wps],
                    1 - SF[wps][syst_var] * effs[wps],
                ),
                axis=1,
            )
            weights[syst_var] = dataeff / mceff
    
    # combination of two WPs: TL or LT (same thing)
    elif wps in ('TL', 'LT'):
        for (
            syst_var
        ) in (
            SF['L'].keys()
        ): # https://twiki.cern.ch/twiki/bin/viewauth/CMS/BTagSFMethods#Extension_to_multiple_operating 
            term1 = np.where(jetsPre.btag >= btagcuts(wps_name['T'], era), effs['T'], 1)
            term2 = np.where((jetsPre.btag < btagcuts(wps_name['T'], era)) & (jetsPre.btag >= btagcuts(wps_name['L'], era)), effs['L'] - effs['T'], 1)
            term3 = np.where(jetsPre.btag < btagcuts(wps_name['L'], era), 1 - effs['L'], 1)
            mceff = ak.prod(term1 * term2 * term3, axis=1)

            term1 = np.where(jetsPre.btag >= btagcuts(wps_name['T'], era), SF['T'][syst_var] * effs['T'], 1)
            term2 = np.where((jetsPre.btag < btagcuts(wps_name['T'], era)) & (jetsPre.btag >= btagcuts(wps_name['L'], era)), SF['L'][syst_var] * effs['L'], 1)
            term3 = np.where(jetsPre.btag < btagcuts(wps_name['L'], era), 1 - SF['L'][syst_var] * effs['L'], 1)
            dataeff = ak.prod(term1 * term2 * term3, axis=1)

            weights[syst_var] = dataeff / mceff

    return weights


def getBTagEffs(jets, era: int, wp:str="L", channel:str="zh") -> dict:
    """
    Get the efficiencies of b-tagging for a given jet collection,
    binned in jet pt and abs(eta), for a given analysis (channel),
    and for a given WP.
    """
    if era == 2015:
        btagfile = f"data/BTagUL16APV/{channel}_eff.pickle"
    if era == 2016:
        btagfile = f"data/BTagUL16/{channel}_eff.pickle"
    if era == 2017:
        btagfile = f"data/BTagUL17/{channel}_eff.pickle"
    if era == 2018:
        btagfile = f"data/BTagUL18/{channel}_eff.pickle"

    bfile = open(btagfile, "rb")
    effsLoad = pickle.load(bfile)

    effs = effsLoad[wp]["L"](jets.pt, np.abs(jets.eta))
    effs = np.where(
        abs(jets.hadronFlavour) == 4, effsLoad[wp]["C"](jets.pt, np.abs(jets.eta)), effs
    )
    effs = np.where(
        abs(jets.hadronFlavour) == 5, effsLoad[wp]["B"](jets.pt, np.abs(jets.eta)), effs
    )

    return effs
    

def btagcuts(WP: str, era: int) -> float:
    if era == 2015:  # 2016APV
        if WP == "Loose":
            return 0.0480
        elif WP == "Medium":
            return 0.2489
        elif WP == "Tight":
            return 0.6377
        raise ValueError(f"Invalid WP: {WP}")
    elif era == 2016:
        if WP == "Loose":
            return 0.0508
        elif WP == "Medium":
            return 0.2598
        elif WP == "Tight":
            return 0.6502
        raise ValueError(f"Invalid WP: {WP}")
    elif era == 2017:
        if WP == "Loose":
            return 0.0532
        elif WP == "Medium":
            return 0.3040
        elif WP == "Tight":
            return 0.7476
        raise ValueError(f"Invalid WP: {WP}")
    elif era == 2018:
        if WP == "Loose":
            return 0.0490
        elif WP == "Medium":
            return 0.2783
        elif WP == "Tight":
            return 0.7100
        raise ValueError(f"Invalid WP: {WP}")
    else:
        raise ValueError(f"Invalid era: {era}")
