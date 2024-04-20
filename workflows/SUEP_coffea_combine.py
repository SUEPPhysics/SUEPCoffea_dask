"""
SUEP_coffea.py
Coffea producer for SUEP analysis. Uses fastjet package to recluster large jets:
https://github.com/scikit-hep/fastjet
Chad Freer and Luca Lavezzo, 2021
"""

from typing import Optional

import awkward as ak
import hist
import numpy as np
import pandas as pd
import vector
from coffea import processor

# Importing CMS corrections
from workflows.CMS_corrections.golden_jsons_utils import applyGoldenJSON
from workflows.pandas_accumulator import pandas_accumulator
import workflows.SUEP_utils as SUEP_utils

# Set vector behavior
vector.register_awkward()


class SUEP_cluster(processor.ProcessorABC):
    def __init__(
        self,
        isMC: int,
        era: int,
        sample: str,
        do_syst: bool,
        syst_var: str,
        weight_syst: bool,
        flag: bool,
        output_location: Optional[str],
        accum: Optional[bool] = None,
        trigger: Optional[str] = None,
        blind: Optional[bool] = False,
        region: Optional[str] = None,
        debug: Optional[bool] = None,
    ) -> None:
        self._flag = flag
        self.output_location = output_location
        self.do_syst = do_syst
        self.gensumweight = 1.0
        self.era = int(era)
        self.isMC = bool(isMC)
        self.sample = sample
        self.syst_var, self.syst_suffix = (
            (syst_var, f"_sys_{syst_var}") if do_syst and syst_var else ("", "")
        )
        self.weight_syst = weight_syst
        self.prefixes = {"SUEP": "SUEP"}
        self.accum = accum
        self.trigger = trigger
        self.blind = blind
        self.region = region
        self.debug = debug

    def eventSelection(self, events):
        """
        Applies trigger, returns events.
        Default is PFHT triggers. Can use selection variable for customization.
        """
        if self.trigger == "TripleMu":
            trigger = (events.HLT.TripleMu_5_3_3_Mass3p8_DZ == 1) | (events.HLT.TripleMu_10_5_5_DZ == 1)
            # trigger = events.HLT.TripleMu_5_3_3_Mass3p8_DZ == 1
            # if self.era == 2016:
            #     trigger = events.HLT.TripleMu_5_3_3 == 1
            # elif self.era == 2017:
            #     trigger = events.HLT.TripleMu_5_3_3_Mass3p8to60_DZ == 1
            # elif self.era == 2018:
            #     if "TripleMu_5_3_3_Mass3p8_DZ" in events.HLT.fields:
            #         trigger = events.HLT.TripleMu_5_3_3_Mass3p8_DZ == 1
            #     else:
            #         if self.isMC:
            #             raise ValueError(
            #                 "This 2018 file seems to have the 2017 trigger names"
            #             )
            #         trigger = ak.zeros_like(events.HLT.ZeroBias)
            # else:
            #     raise ValueError("Invalid era")
            events = events[trigger]
        else:
            raise ValueError("Invalid trigger path")
        return events

    def ht(self, events):
        jet_Cut = (events.Jet.pt > 20) & (abs(events.Jet.eta) < 2.4)
        jets = events.Jet[jet_Cut]
        return ak.sum(jets.pt, axis=-1)

    def muon_filter(
        self,
        events,
    ):
        """
        Filter events after the TripleMu trigger.
        Cleans muons and electrons.
        Requires at least nMuons with mediumId, pt, dxy, dz, and eta cuts.
        """
        muons = events.Muon
        # remove events with no muons
        events, muons = events[ak.num(muons) > 0], muons[ak.num(muons) > 0]
        clean_muons = (
            (events.Muon.mediumId)
            & (events.Muon.pt > 3)
            & (abs(events.Muon.eta) < 2.4)
            & (abs(events.Muon.dz) < 0.2)
        )
        if self.region == "CR_prompt" or self.region == "SR":
            clean_muons = clean_muons & (
                (abs(events.Muon.dxy) <= 0.02) & (abs(events.Muon.dz) <= 0.1)
            )

        muons = muons[clean_muons]
        select_by_muons_high = ak.num(muons, axis=-1) < 5
        if self.region == "SR":
            select_by_muons_high = ak.num(muons, axis=-1) > 4
        select_by_muons_low = ak.num(muons, axis=-1) > 2
        events = events[select_by_muons_high & select_by_muons_low]
        muons = muons[select_by_muons_high & select_by_muons_low]

        # Apply extra very tight cuts
        CR_requirement = ak.ones_like(muons.pt, dtype=bool)
        if self.region == "CR_prompt":
            CR_requirement = (
                (muons.pt > 25)
                & (muons.miniPFRelIso_all < 0.1)
                & (abs(muons.dxy) < 0.003)
                & (abs(muons.dz) < 0.01)
            )
        elif self.region == "CR_cb":
            CR_requirement = (abs(muons.dxy) >= 0.01) & (abs(muons.dxy) <= 0.2)

        muons = muons[CR_requirement]

        # Make sure there is at least one muon in the event after the cuts
        select_by_muons_final = ak.num(CR_requirement, axis=-1) > 0
        events = events[select_by_muons_final]
        muons = muons[select_by_muons_final]

        # Extra cuts for the SR
        # Require the leading muon to have inter-iso > 1.0
        if self.region == "SR":
            muonsCollection = ak.zip(
                {
                    "pt": events.Muon.pt,
                    "eta": events.Muon.eta,
                    "phi": events.Muon.phi,
                    "mass": events.Muon.mass,
                    "charge": events.Muon.pdgId / (-13),
                },
                with_name="Momentum4D",
            )
            SR_requirements = (
                SUEP_utils.inter_isolation(muonsCollection[:, 0], muonsCollection) > 1.0
            ) & (ak.mean(events.Muon.pt, axis=-1) < 10)
            events = events[SR_requirements]
            muons = muons[SR_requirements]

        return events, muons

    def fill_histograms(self, events, muons, output):
        dataset = events.metadata["dataset"]

        # These arrays need to be broadcasted to the per muon dims from per event dims
        nMuons = ak.flatten(ak.broadcast_arrays(ak.num(muons), muons.pt)[0])
        nMuons = ak.where(nMuons > 7, 7, nMuons)
        weights = np.ones(len(events))
        muons_genPartFlav = ak.flatten(ak.zeros_like(muons.pt, dtype=int))
        nMuons_ndf = np.zeros(len(events))
        if self.isMC:
            weights = events.genWeight
            muons_genPartFlav = ak.flatten(muons.genPartFlav).to_numpy().astype(int)
            nMuons_ndf = ak.sum(
                (muons.genPartFlav == 0) | (muons.genPartFlav == 3),
                axis=-1,
            )
            nMuons_ndf = ak.where(nMuons_ndf > 7, 7, nMuons_ndf)
        weights_per_muon = ak.flatten(ak.broadcast_arrays(weights, muons.pt)[0])

        if len(events) == 0 or ak.all(ak.num(muons) == 0):
            return

        # Fill the histograms
        HT = self.ht(events)
        output[dataset]["histograms"]["ht"].fill(
            ak.where(HT >= 2000, 1999, HT),
            nMuons_ndf,
            ak.where(ak.num(muons) > 7, 7, ak.num(muons)),
            weight=weights,
        )
        output[dataset]["histograms"]["av_muon_pt"].fill(
            ak.fill_none(
                ak.where(
                    ak.mean(muons.pt, axis=-1) >= 500, 499, ak.mean(muons.pt, axis=-1)
                ),
                0,
            ),
            nMuons_ndf,
            ak.where(ak.num(muons) > 7, 7, ak.num(muons)),
            weight=weights,
        )
        output[dataset]["histograms"]["muon_pt"].fill(
            ak.where(ak.flatten(muons.pt) >= 500, 499, ak.flatten(muons.pt)),
            muons_genPartFlav,
            ak.flatten(ak.broadcast_arrays(nMuons_ndf, muons.pt)[0]),
            nMuons,
            weight=weights_per_muon,
        )
        output[dataset]["histograms"]["muon_eta"].fill(
            ak.where(
                ak.flatten(muons.eta) >= 2.4,
                2.39,
                ak.where(ak.flatten(muons.eta) <= -2.4, -2.39, ak.flatten(muons.eta)),
            ),
            muons_genPartFlav,
            ak.flatten(ak.broadcast_arrays(nMuons_ndf, muons.pt)[0]),
            nMuons,
            weight=weights_per_muon,
        )
        return

    def analysis(self, events, output):
        #####################################################################################
        # ---- Trigger event selection
        # Cut based on ak4 jets to replicate the trigger
        #####################################################################################

        # get dataset name
        dataset = events.metadata["dataset"]

        # take care of weights
        weights = np.ones(len(events))
        if self.isMC:
            weights = events.genWeight

        # Fill the cutflow columns for all
        output[dataset]["cutflow"].fill(len(events) * ["all"], weight=weights)

        # golden jsons for offline data
        if not self.isMC:
            events = applyGoldenJSON(self, events)

        events = self.eventSelection(events)

        # Apply HT selection for WJets stiching
        if "WJetsToLNu_HT" in dataset:
            events = events[self.ht(events) >= 70]
        elif "WJetsToLNu_TuneCP5" in dataset:
            events = events[self.ht(events) < 70]

        weights = np.ones(len(events))
        if self.isMC:
            weights = events.genWeight

        # Fill the cutflow columns for trigger
        output[dataset]["cutflow"].fill(
            len(events) * ["trigger"],
            weight=weights,
        )

        # fill the histograms
        events, muons = self.muon_filter(events)
        self.fill_histograms(events, muons, output)

        return

    def process(self, events):
        dataset = events.metadata["dataset"]
        cutflow = hist.Hist.new.StrCategory(
            [
                "all",
                "trigger",
            ],
            name="cutflow",
            label="cutflow",
        ).Weight()
        histograms = {
            "ht": hist.Hist.new.Regular(
                30,
                20,
                2000,
                name="ht",
                label="ht",
                transform=hist.axis.transform.log,
            )
            .Regular(8, 0, 8, name="nMuon_ndf", label="nMuon_ndf")
            .Regular(8, 0, 8, name="nMuon", label="nMuon")
            .Weight(),
            "av_muon_pt": hist.Hist.new.Regular(
                30,
                3,
                500,
                name="av_Muon_pt",
                label="av_Muon_pt",
                transform=hist.axis.transform.log,
            )
            .Regular(8, 0, 8, name="nMuon_ndf", label="nMuon_ndf")
            .Regular(8, 0, 8, name="nMuon", label="nMuon")
            .Weight(),
            "muon_pt": hist.Hist.new.Regular(
                30,
                3,
                500,
                name="Muon_pt",
                label="Muon_pt",
                transform=hist.axis.transform.log,
            )
            .IntCategory(
                [0, 1, 3, 4, 5, 15], name="Muon_genPartFlav", label="Muon_genPartFlav"
            )
            .Regular(8, 0, 8, name="nMuon_ndf", label="nMuon_ndf")
            .Regular(8, 0, 8, name="nMuon", label="nMuon")
            .Weight(),
            "muon_eta": hist.Hist.new.Regular(
                20,
                -2.4,
                2.4,
                name="Muon_eta",
                label="Muon_eta",
            )
            .IntCategory(
                [0, 1, 3, 4, 5, 15], name="Muon_genPartFlav", label="Muon_genPartFlav"
            )
            .Regular(8, 0, 8, name="nMuon_ndf", label="nMuon_ndf")
            .Regular(8, 0, 8, name="nMuon", label="nMuon")
            .Weight(),
        }
        output = {
            dataset: {
                "cutflow": cutflow,
                "gensumweight": processor.value_accumulator(float, 0),
                "vars": pandas_accumulator(pd.DataFrame()),
                "histograms": histograms,
            },
        }

        # gen weights
        if self.isMC:
            self.gensumweight = ak.sum(events.genWeight)
            output[dataset]["gensumweight"].add(self.gensumweight)

        # run the analysis
        self.analysis(events, output)

        return output

    def postprocess(self, accumulator):
        pass
