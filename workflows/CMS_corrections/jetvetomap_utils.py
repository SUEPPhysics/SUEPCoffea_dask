import numpy as np
import awkward as ak
import correctionlib

def JetVetoMap(jets, era: str, prefix: str = "./"):
    """
    Calculate the jet veto map to the jets. Events are selected only if they have no jets in the vetoed region.
    Following reccomendation from JERC: https://cms-jerc.web.cern.ch/Recommendations/#jet-energy-resolution
    """

    # these json files are from the JERC group, found in link above
    fname = prefix + "data/JetVetoMaps/jetvetomaps_" + era + ".json"
    hname = {
        "2016apv": "Summer19UL16_V1",
        "2016"   : "Summer19UL16_V1",
        "2017"   : "Summer19UL17_V1",
        "2018"   : "Summer19UL18_V1"
    }

    # load the correction set
    evaluator = correctionlib.CorrectionSet.from_file(fname)

    # correctionlib doesn't support awkward arrays, so we have to flatten them out
    etaflat, phiflat, counts = ak.flatten(jets.eta), ak.flatten(jets.phi), ak.num(jets.eta)

    # apply the correction and recreate the awkward array shape
    weight = evaluator[hname[era]].evaluate('jetvetomap', etaflat, phiflat)
    weight_ak = ak.unflatten(
        np.array(weight),
        counts=counts
    )

    # any non-zero weight means the jet is vetoed
    jetmask = (weight_ak == 0)

    # events are selected only if they have no jets in the vetoed region
    eventmask = ak.sum(weight_ak, axis=-1) == 0

    return jetmask, eventmask