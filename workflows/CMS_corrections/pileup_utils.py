import numpy as np
import uproot


def pileup_weight(era, nTrueInt, sys=""):
    """
    Function to get the pileup weights for a given era and systematic variation
    The pileup weights are calculated as the ratio of the data distribution to the MC distribution
    The data distribution is normalized to 1

    Parameters:
    era: str
        The year of the data taking
    nTrueInt: array
        The number of true interactions
    sys: str
        The systematic variation to be applied to the pileup weights
        
    Returns:
    weights: array
        The pileup weights
    """
    if era not in ["2016", "2017", "2018"]:
        raise ValueError(
            "no pileup weights because no year was selected for function pileup_weight"
        )

    f_MC = uproot.open(f"data/pileup/mcPileupUL{era}.root")
    f_data = uproot.open(
        f"data/pileup/PileupHistogram-UL{era}-100bins_withVar.root"
    )

    variation = ""
    if "up" in sys:
        variation = "_plus"
    elif "down" in sys:
        variation = "_minus"

    hist_MC = f_MC["pu_mc"].to_numpy()
    hist_data = f_data["pileup" + variation].to_numpy()
    hist_data[0].sum()
    norm_data = hist_data[0] / hist_data[0].sum()
    weights = np.divide(
        norm_data, hist_MC[0], out=np.ones_like(norm_data), where=hist_MC[0] != 0
    )

    return weights[nTrueInt]
