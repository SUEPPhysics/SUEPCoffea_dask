"""
SUEP_coffea.py
Coffea producer for SUEP analysis. Uses fastjet package to recluster large jets:
https://github.com/scikit-hep/fastjet
Chad Freer and Luca Lavezzo, 2021
"""

from typing import Optional

import awkward as ak
import numpy as np
import pandas as pd
import vector
from coffea import processor

vector.register_awkward()

# Importing SUEP specific functions
import workflows.SUEP_utils as SUEP_utils

# Importing CMS corrections
from workflows.CMS_corrections.golden_jsons_utils import applyGoldenJSON
from workflows.SUEP_coffea import SUEP_cluster


class cutflow_cluster(processor.ProcessorABC):
    def __init__(
        self,
        isMC: int,
        era: int,
        scouting: int,
        sample: str,
        do_syst: bool,
        syst_var: str,
        weight_syst: bool,
        flag: bool,
        do_inf: bool,
        output_location: Optional[str],
    ) -> None:
        self._flag = flag
        self.output_location = output_location
        self.do_syst = do_syst
        self.gensumweight = 1.0
        self.scouting = scouting
        self.era = era
        self.isMC = isMC
        self.sample = sample
        self.syst_var, self.syst_suffix = (
            (syst_var, f"_sys_{syst_var}") if do_syst and syst_var else ("", "")
        )
        self.weight_syst = weight_syst
        self.do_inf = do_inf
        self.prefixes = {"SUEP": "SUEP"}

        self.out_vars = pd.DataFrame()

        # Set up the image size and pixels
        self.eta_pix = 280
        self.phi_pix = 360
        self.eta_span = (-2.5, 2.5)
        self.phi_span = (-np.pi, np.pi)
        self.eta_scale = self.eta_pix / (self.eta_span[1] - self.eta_span[0])
        self.phi_scale = self.phi_pix / (self.phi_span[1] - self.phi_span[0])
        self.models = [
            "model125"
        ]  # Add to this list. There will be an output for each prediction in this list

        self._accumulator = processor.dict_accumulator(
            {
                "sumw": processor.defaultdict_accumulator(float),
                "total": processor.defaultdict_accumulator(float),
                "cut1": processor.defaultdict_accumulator(float),
                "cut2": processor.defaultdict_accumulator(float),
                "cut3": processor.defaultdict_accumulator(float),
                "cut4": processor.defaultdict_accumulator(float),
                "cut5": processor.defaultdict_accumulator(float),
            }
        )

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, events):
        output = self.accumulator.identity()
        dataset = events.metadata["dataset"]

        # gen weights
        if self.isMC and self.scouting == 1:
            self.gensumweight = ak.num(events.PFcand.pt, axis=0)
        elif self.isMC:
            self.gensumweight = ak.sum(events.genWeight)

        # golden jsons for offline data
        if not self.isMC and self.scouting != 1:
            events = applyGoldenJSON(events)

        output["sumw"][dataset] += ak.sum(events.genWeight)
        output["total"][dataset] += len(events)

        #####################################################################################
        # ---- Trigger event selection
        # Cut based on ak4 jets to replicate the trigger
        #####################################################################################

        Jets = ak.zip(
            {
                "pt": events.Jet.pt,
                "eta": events.Jet.eta,
                "phi": events.Jet.phi,
                "mass": events.Jet.mass,
                "jetId": events.Jet.jetId,
            }
        )
        jetCut = (Jets.pt > 30) & (abs(Jets.eta) < 2.4)
        ak4jets = Jets[jetCut]
        ht = ak.sum(ak4jets.pt, axis=-1)

        # apply trigger selection
        if self.scouting == 1:
            events = events[(ht > 600)]
            ak4jets = ak4jets[(ht > 600)]
        else:
            if self.era == 2016:
                trigger = events.HLT.PFHT900 == 1
            else:
                trigger = events.HLT.PFHT1050 == 1

            events = events[(trigger)]
            ak4jets = ak4jets[(trigger)]
            ht = ht[(trigger)]

            output["cut1"][dataset] += len(events)

            events = events[(ht > 1200)]
            ak4jets = ak4jets[(ht > 1200)]

            output["cut2"][dataset] += len(events)

        # output empty dataframe if no events pass trigger
        if len(events) == 0:
            print("No events passed trigger.")
            return output

        #####################################################################################
        # ---- Track selection
        # Prepare the clean PFCand matched to tracks collection
        #####################################################################################

        if self.scouting == 1:
            tracks, Cleaned_cands = SUEP_cluster.getScoutingTracks(self, events)
        else:
            tracks, Cleaned_cands = SUEP_cluster.getTracks(self, events)

        #####################################################################################
        # ---- FastJet reclustering
        # The jet clustering part
        #####################################################################################

        ak_inclusive_jets, ak_inclusive_cluster = SUEP_utils.FastJetReclustering(
            self, tracks, r=1.5, minPt=150
        )

        #####################################################################################
        # ---- Cut Based Analysis
        #####################################################################################

        # remove events with at least 2 clusters (i.e. need at least SUEP and ISR jets for IRM)
        clusterCut = ak.num(ak_inclusive_jets, axis=1) > 1
        ak_inclusive_cluster = ak_inclusive_cluster[clusterCut]
        ak_inclusive_jets = ak_inclusive_jets[clusterCut]
        tracks = tracks[clusterCut]
        output["cut3"][dataset] += len(tracks)

        # output file if no events pass selections, avoids errors later on
        if len(tracks) == 0:
            print("No events pass clusterCut.")
            return output

        tracks, indices, topTwoJets = SUEP_utils.getTopTwoJets(
            self,
            tracks,
            np.arange(0, len(tracks)),
            ak_inclusive_jets,
            ak_inclusive_cluster,
        )
        SUEP_cand, ISR_cand, SUEP_cluster_tracks, ISR_cluster_tracks = topTwoJets

        # boost into frame of SUEP
        boost_SUEP = ak.zip(
            {
                "px": SUEP_cand.px * -1,
                "py": SUEP_cand.py * -1,
                "pz": SUEP_cand.pz * -1,
                "mass": SUEP_cand.mass,
            },
            with_name="Momentum4D",
        )

        # SUEP tracks for this method are defined to be the ones from the cluster
        # that was picked to be the SUEP jet
        SUEP_tracks_b = SUEP_cluster_tracks.boost_p4(boost_SUEP)

        # SUEP jet variables
        eigs = sphericity(SUEP_tracks_b, 1.0)  # Set r=1.0 for IRC safe
        S1 = 1.5 * (eigs[:, 1] + eigs[:, 0])
        nconst = ak.num(SUEP_tracks_b)

        tracks = tracks[(nconst > 80)]
        S1 = S1[(nconst > 80)]
        output["cut4"][dataset] += len(tracks)

        tracks = tracks[(S1 > 0.5)]
        output["cut5"][dataset] += len(tracks)

        return output

    def postprocess(self, accumulator):
        return accumulator
