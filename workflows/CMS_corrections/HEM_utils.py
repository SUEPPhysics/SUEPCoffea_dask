import awkward as ak
import numpy as np


def jetHEMFilter(self, jets):
    """
    Due to the HEM issue in year 2018, we veto either the jets in the region -3<eta<-1.3 and -1.57<phi<-0.87, or the events with jets in this region to remove fake MET.
    """
    if self.era == "2018":
        jetHEMCut = (
            (jets.eta <= -3)
            | (jets.eta >= -1.3)
            | (jets.phi <= -1.57)
            | (jets.phi >= -0.87)
        )
    else:
        jetHEMCut = jets.pt > 0
    eventHEMCut = ak.sum(jetHEMCut == False, axis=1) == 0
    return jetHEMCut, eventHEMCut
