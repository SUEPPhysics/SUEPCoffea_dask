import matplotlib.pyplot as plt
import numpy as np


def higgs_reweight(gen_pt):
    # numbers taken from table 1 here: https://cds.cern.ch/record/2669113/files/LHCHXSWG-2019-002.pdf
    bins = np.array(
        [
            0,
            400,
            450,
            500,
            550,
            600,
            650,
            700,
            750,
            800,
            850,
            900,
            950,
            1000,
            1050,
            1100,
            1150,
            1200,
            1250,
            15000,
        ]
    )
    Higgs_factor = np.array(
        [
            1.25,
            1.25,
            1.25,
            1.25,
            1.25,
            1.24,
            1.24,
            1.24,
            1.24,
            1.24,
            1.24,
            1.24,
            1.24,
            1.24,
            1.24,
            1.24,
            1.24,
            1.24,
            1.24,
        ]
    )
    up_factor = np.array(
        [
            1.092,
            1.092,
            1.089,
            1.088,
            1.088,
            1.087,
            1.087,
            1.087,
            1.087,
            1.087,
            1.085,
            1.086,
            1.086,
            1.086,
            1.087,
            1.087,
            1.087,
            1.086,
            1.086,
        ]
    )
    down_factor = np.array(
        [
            0.88,
            0.88,
            0.89,
            0.89,
            0.89,
            0.89,
            0.89,
            0.89,
            0.89,
            0.89,
            0.89,
            0.89,
            0.89,
            0.89,
            0.89,
            0.89,
            0.88,
            0.88,
            0.88,
        ]
    )

    vals = plt.hist(gen_pt, bins=bins)

    freqs = vals[0] * Higgs_factor
    ups = freqs * up_factor
    downs = freqs * down_factor

    start = vals[0].sum()
    end = freqs.sum()
    factor = start / end

    freqs = freqs * factor
    ups = ups * factor
    downs = downs * factor

    weights = np.divide(freqs, vals[0], out=np.ones_like(freqs), where=vals[0] != 0)
    weights_up = np.divide(ups, vals[0], out=np.ones_like(ups), where=vals[0] != 0)
    weights_down = np.divide(
        downs, vals[0], out=np.ones_like(downs), where=vals[0] != 0
    )

    return bins, weights, weights_up, weights_down


def get_higgs_weight(
    df, sys, higgs_bins, higgs_weights, higgs_weights_up, higgs_weights_down
):
    gen_pt = np.array(df["SUEP_genPt"]).astype(int)
    gen_bin = np.digitize(gen_pt, higgs_bins) - 1
    if "higgs_weights_up" in sys:
        higgs_weight = higgs_weights_up[gen_bin]
    elif "higgs_weights_down" in sys:
        higgs_weight = higgs_weights_down[gen_bin]
    else:
        higgs_weight = higgs_weights[gen_bin]
    return higgs_weight
