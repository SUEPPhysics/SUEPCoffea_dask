import numpy as np
import uproot


def pileup_weight(era):
    if era == "2018":
        f_MC = uproot.open("../data/pileup/mcPileupUL2018.root")
        f_data = uproot.open(
            "../data/pileup/PileupHistogram-UL2018-100bins_withVar.root"
        )
    elif era == "2017":
        f_MC = uproot.open("../data/pileup/mcPileupUL2017.root")
        f_data = uproot.open(
            "../data/pileup/PileupHistogram-UL2017-100bins_withVar.root"
        )
    elif era == "2016" or era == "2016apv":
        f_MC = uproot.open("../data/pileup/mcPileupUL2016.root")
        f_data = uproot.open(
            "../data/pileup/PileupHistogram-UL2016-100bins_withVar.root"
        )
    else:
        print(
            "no pileup weights because no year was selected for function pileup_weight"
        )

    hist_MC = f_MC["pu_mc"].to_numpy()
    # Normalize the data distribution
    hist_data = f_data["pileup"].to_numpy()
    hist_data[0].sum()
    norm_data = hist_data[0] / hist_data[0].sum()
    weights = np.divide(
        norm_data, hist_MC[0], out=np.ones_like(norm_data), where=hist_MC[0] != 0
    )

    # The plus version of the pileup
    hist_data_plus = f_data["pileup_plus"].to_numpy()
    hist_data_plus[0].sum()
    norm_data_plus = hist_data_plus[0] / hist_data_plus[0].sum()
    weights_plus = np.divide(
        norm_data_plus,
        hist_MC[0],
        out=np.ones_like(norm_data_plus),
        where=hist_MC[0] != 0,
    )

    # The minus version of the pileup
    hist_data_minus = f_data["pileup_minus"].to_numpy()
    hist_data_minus[0].sum()
    norm_data_minus = hist_data_minus[0] / hist_data_minus[0].sum()
    weights_minus = np.divide(
        norm_data_minus,
        hist_MC[0],
        out=np.ones_like(norm_data_minus),
        where=hist_MC[0] != 0,
    )

    return weights, weights_plus, weights_minus


def get_pileup_weights(df, sys, puweights, puweights_up, puweights_down):
    Pileup_nTrueInt = np.array(df["Pileup_nTrueInt"]).astype(int)
    if "puweights_up" in sys:
        pu = puweights_up[Pileup_nTrueInt]
    elif "puweights_down" in sys:
        pu = puweights_down[Pileup_nTrueInt]
    else:
        pu = puweights[Pileup_nTrueInt]
    return pu
