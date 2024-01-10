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
