import numpy as np
import uproot


def triggerSF(era):
    if era == 2018:
        f_weight = uproot.open("../data/trigSF/trigSF_2018.root")
    elif era == 2017:
        f_weight = uproot.open("../data/trigSF/trigSF_2017.root")
    elif era == 2016:
        f_weight = uproot.open("../data/trigSF/trigSF_2016.root")
    else:
        print("no TriggerSFs because no year was selected for function triggerSF")

    hist = f_weight["TriggerSF"].to_boost()
    bins = hist.axes[0].edges
    weights = hist.values()
    weights_up = hist.values() + hist.variances()
    weights_down = np.clip((hist.values() - hist.variances()), 0, 15)

    return bins, weights, weights_up, weights_down
