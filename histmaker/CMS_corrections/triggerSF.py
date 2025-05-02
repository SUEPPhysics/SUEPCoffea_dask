import json
import os

import numpy as np
import uproot


def triggerSF(era):
    if era == "2018":
        f_weight = uproot.open("../data/trigSF/trigSF_2018.root")
    elif era == "2017":
        f_weight = uproot.open("../data/trigSF/trigSF_2017.root")
    elif era == "2016" or era == "2016apv":
        f_weight = uproot.open("../data/trigSF/trigSF_2016.root")
    else:
        print("no TriggerSFs because no year was selected for function triggerSF")

    hist = f_weight["TriggerSF"].to_boost()
    bins = hist.axes[0].edges
    weights = hist.values()
    weights_up = hist.values() + hist.variances()
    weights_down = np.clip((hist.values() - hist.variances()), 0, 15)

    return bins, weights, weights_up, weights_down


def get_trigSF_weight(
    df, sys, trig_bins, trig_weights, trig_weights_up, trig_weights_down
):
    ht = np.array(df["ht"]).astype(int)
    ht_bin = np.digitize(ht, trig_bins) - 1  # digitize the values to bins
    ht_bin = np.clip(ht_bin, 0, 49)  # Set overl flow to last SF
    if "trigSF_up" in sys:
        trigSF = trig_weights_up[ht_bin]
    elif "trigSF_down" in sys:
        trigSF = trig_weights_down[ht_bin]
    else:
        trigSF = trig_weights[ht_bin]
    return trigSF


def get_scout_trigSF_weight(htarray, sys, era="2018"):
    if "16" in era:
        scaleFactor = 1
    else:
        bins, trigwgts, wgterr = np.loadtxt(f"../data/trigSF/scout_trigSF_{era}.txt")
        htbin = np.digitize(htarray, bins)
        trigwgts = np.insert(trigwgts, 0, 0)
        wgterr = np.insert(wgterr, 0, 0)
        scaleFactorNom = np.take(trigwgts, htbin)
        scaleFactorErr = np.take(wgterr, htbin)
        if "trigSF_up" in sys:
            scaleFactor = scaleFactorNom + scaleFactorErr
        elif "trigSF_down" in sys:
            scaleFactor = scaleFactorNom - scaleFactorErr
        else:
            scaleFactor = scaleFactorNom
    return scaleFactor


def WH(leptonpt, pdgids, sys, era):

    data_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

    mu_filename = f"{data_dir}/data/WH_triggerSFs/muon{era}sfs.json"
    ele_filename = f"{data_dir}/data/WH_triggerSFs/electron{era}sfs.json"

    with open(mu_filename) as f:
        muSF_dict = json.load(f)
    with open(ele_filename) as f:
        eleSF_dict = json.load(f)

    ele_bins = eleSF_dict["lep1pt"]
    mu_bins = muSF_dict["lep1pt"]

    ele_bin_edges = [entry["bin_range"][0] for entry in ele_bins]
    ele_bin_edges.append(ele_bins[-1]["bin_range"][1])
    ele_bin_edges = np.array(ele_bin_edges)

    mu_bin_edges = [entry["bin_range"][0] for entry in mu_bins]
    mu_bin_edges.append(mu_bins[-1]["bin_range"][1])
    mu_bin_edges = np.array(mu_bin_edges)

    ele_SFs = np.array(
        [
            entry["scale_factor"] if entry["scale_factor"] > 0.0 else 1.0
            for entry in ele_bins
        ]
    )
    mu_SFs = np.array(
        [
            entry["scale_factor"] if entry["scale_factor"] > 0.0 else 1.0
            for entry in mu_bins
        ]
    )

    ele_SFs_up = np.array([entry["total_uncertainty_up"] for entry in ele_bins])
    mu_SFs_up = np.array([entry["total_uncertainty_up"] for entry in mu_bins])
    ele_SFs_down = np.array([entry["total_uncertainty_low"] for entry in ele_bins])
    mu_SFs_down = np.array([entry["total_uncertainty_low"] for entry in mu_bins])

    SFs = np.ones_like(leptonpt, dtype=float)

    is_electron = np.isin(np.abs(pdgids), [11])
    is_muon = np.isin(np.abs(pdgids), [13])

    ele_indices = np.digitize(leptonpt[is_electron], ele_bin_edges) - 1
    ele_indices = np.clip(ele_indices, 0, len(ele_SFs) - 1)

    mu_indices = np.digitize(leptonpt[is_muon], mu_bin_edges) - 1
    mu_indices = np.clip(mu_indices, 0, len(mu_SFs) - 1)

    if "trigSF_up" in sys:
        SFs[is_electron] = ele_SFs[ele_indices] + ele_SFs_up[ele_indices]
        SFs[is_muon] = mu_SFs[mu_indices] + mu_SFs_up[mu_indices]
    elif "trigSF_down" in sys:
        SFs[is_electron] = ele_SFs[ele_indices] - ele_SFs_down[ele_indices]
        SFs[is_muon] = mu_SFs[mu_indices] - mu_SFs_down[mu_indices]
    else:
        SFs[is_electron] = ele_SFs[ele_indices]
        SFs[is_muon] = mu_SFs[mu_indices]

    return SFs
