from typing import List

import awkward as ak
import correctionlib
import numpy as np
from correctionlib import _core


def getPhotonSFs(
    photons: ak.Array, era: str, wp: str
) -> List[float]:
    """
    Get the photon ID scale factors for an awkward array of photons.
    https://twiki.cern.ch/twiki/bin/viewauth/CMS/EgammaSFJSON
    """
    output = {}

    era_map = {
        "2016apv": "2016preVFP",
        "2016": "2016postVFP",
        "2017": "2017",
        "2018": "2018",
    }
    evaluator = correctionlib.CorrectionSet.from_file(
        "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/EGM/"
        + era_map[era]
        + "_UL/photon.json.gz"
    )

    # if the awkward array is a list of photons (1 dimensional), it's already flat
    # elif it's a list of list of photons (2 dimensional), we need to flatten it
    need_to_unflatten = False
    if str(photons.type).count("*") == 2:
        photons_flat = photons
    elif str(photons.type).count("*") == 3:
        photons_flat, photons_n = ak.flatten(photons), np.array(ak.num(photons))
        need_to_unflatten = True

    output["nominal"] = evaluator["UL-Photon-ID-SF"].evaluate(
        era_map[era], "sf", wp, photons_flat.eta, photons_flat.pt
    )
    if need_to_unflatten:
        output["nominal"] = ak.unflatten(output["nominal"], photons_n)
    output["up"] = evaluator["UL-Photon-ID-SF"].evaluate(
        era_map[era], "sfup", wp, photons_flat.eta, photons_flat.pt
    )
    if need_to_unflatten:
        output["up"] = ak.unflatten(output["up"], photons_n)
    output["down"] = evaluator["UL-Photon-ID-SF"].evaluate(
        era_map[era], "sfdown", wp, photons_flat.eta, photons_flat.pt
    )
    if need_to_unflatten:
        output["down"] = ak.unflatten(output["down"], photons_n)

    return output
