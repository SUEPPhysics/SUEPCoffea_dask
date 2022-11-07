import numpy as np
import matplotlib.pyplot as plt

def higgs_reweight(gen_pt):
    #numbers taken from table 1 here: https://cds.cern.ch/record/2669113/files/LHCHXSWG-2019-002.pdf
    bins = np.array([   0,  400,  450,  500,  550,  600,  650,  700,  750,  800,  850,
            900,  950, 1000, 1050, 1100, 1150, 1200, 1250, 1500])
    Higgs_factor = np.array([1.25, 1.25, 1.25, 1.25, 1.25, 1.24, 1.24, 1.24, 1.24, 1.24, 1.24,
           1.24, 1.24, 1.24, 1.24, 1.24, 1.24, 1.24, 1.24])
    up_factor   = np.array([1.092, 1.092, 1.089, 1.088, 1.088, 1.087, 1.087, 1.087, 1.087,
           1.087, 1.085, 1.086, 1.086, 1.086, 1.087, 1.087, 1.087, 1.086,
           1.086])
    down_factor = np.array([0.88, 0.88, 0.89, 0.89, 0.89, 0.89, 0.89, 0.89, 0.89, 0.89, 0.89,
           0.89, 0.89, 0.89, 0.89, 0.89, 0.88, 0.88, 0.88])
    
    vals = plt.hist(gen_pt, bins=bins)
    
    freqs = vals[0] * Higgs_factor
    ups   = freqs * up_factor
    downs = freqs * down_factor
    
    start = vals[0].sum()
    end = freqs.sum()
    factor = start/end
    
    freqs = freqs * factor
    ups = ups * factor
    downs = downs * factor
    
    weights = freqs / vals[0]
    weights_up = ups / vals[0]
    weights_down = downs / vals[0]
    
    return bins, weights, weights_up, weights_down
