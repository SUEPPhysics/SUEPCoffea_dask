import awkward as ak
import correctionlib
import numpy as np


def JetVetoMap(jets, era: str):
    """
    Calculate the jet veto map to the jets. Events are selected only if they have no jets in the vetoed region.
    Following reccomendation from JERC: https://cms-jerc.web.cern.ch/Recommendations/#jet-energy-resolution
    """

    # these json files are from the JERC group, found in link above
    era_tags = {
        "2016apv": "2016preVFP_UL",
        "2016": "2016postVFP_UL",
        "2017": "2017_UL",
        "2018": "2018_UL",
    }
    era_tag = era_tags[era]
    fname = f"/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/JME/{era_tag}/jetvetomaps.json.gz"

    # load the correction set
    evaluator = correctionlib.CorrectionSet.from_file(fname)

    # correctionlib doesn't support awkward arrays, so we have to flatten them out
    etaflat, phiflat, counts = (
        ak.flatten(jets.eta),
        ak.flatten(jets.phi),
        ak.num(jets.eta),
    )
    phiflat = np.clip(np.array(phiflat), -3.1415, 3.1415)
    etaflat = np.clip(np.array(etaflat), -4.7, 4.7)

    # apply the correction and recreate the awkward array shape
    hname = {
        "2016apv": "Summer19UL16_V1",
        "2016": "Summer19UL16_V1",
        "2017": "Summer19UL17_V1",
        "2018": "Summer19UL18_V1",
    }
    weight = evaluator[hname[era]].evaluate("jetvetomap", etaflat, phiflat)
    weight_ak = ak.unflatten(np.array(weight), counts=counts)

    # any non-zero weight means the jet is vetoed
    jetmask = weight_ak == 0

    # events are selected only if they have no jets in the vetoed region
    eventmask = ak.sum(weight_ak, axis=-1) == 0

    return jetmask, eventmask
