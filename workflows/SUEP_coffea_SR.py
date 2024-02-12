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

# Importing SUEP specific functions
import workflows.SUEP_utils as SUEP_utils

# Importing CMS corrections
from workflows.CMS_corrections.golden_jsons_utils import applyGoldenJSON
from workflows.pandas_accumulator import pandas_accumulator

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
        self.debug = debug

    def eventSelection(self, events):
        """
        Applies trigger, returns events.
        Default is PFHT triggers. Can use selection variable for customization.
        """
        if self.trigger == "TripleMu":
            if self.era == 2016:
                trigger = events.HLT.TripleMu_5_3_3 == 1
            elif self.era == 2017:
                trigger = events.HLT.TripleMu_5_3_3_Mass3p8to60_DZ == 1
            elif self.era == 2018:
                if "TripleMu_5_3_3_Mass3p8_DZ" in events.HLT.fields:
                    trigger = events.HLT.TripleMu_5_3_3_Mass3p8_DZ == 1
                else:
                    if self.isMC:
                        raise ValueError(
                            "This 2018 file seems to have the 2017 trigger names"
                        )
                    trigger = ak.zeros_like(events.HLT.ZeroBias)
            else:
                raise ValueError("Invalid era")
            events = events[trigger]
        else:
            raise ValueError("Invalid trigger path")
        return events

    def ht(self, events):
        jet_Cut = (events.Jet.pt > 30) & (abs(events.Jet.eta) < 2.4)
        jets = events.Jet[jet_Cut]
        return ak.sum(jets.pt, axis=-1)

    def clean_jets(self, jets):
        jet_Cut = (jets.pt > 30) & (abs(jets.eta) < 2.4)
        return jets[jet_Cut]

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
        clean_muons = (
            (events.Muon.mediumId)
            & (events.Muon.pt > 3)
            & (abs(events.Muon.eta) < 2.4)
            & (abs(events.Muon.dz) < 0.2)
        )
        prompt_muons = (abs(events.Muon.dxy) <= 0.02) & (abs(events.Muon.dz) <= 0.1)
        muons = muons[clean_muons & prompt_muons]
        select_by_muons_high = ak.num(muons, axis=-1) > 4
        events = events[select_by_muons_high]
        muons = muons[select_by_muons_high]

        return events, muons

    def fill_cutflow(self, events, muons, output):
        dataset = events.metadata["dataset"]
        weights = np.ones(len(events))
        if self.isMC:
            weights = events.genWeight

        output[dataset]["cutflow"].fill(
            len(events) * ["nMuon>4"],
            weight=weights,
        )

        mask = ak.num(muons) >= 6
        events = events[mask]
        muons = muons[mask]
        weights = weights[mask]
        output[dataset]["cutflow"].fill(
            len(events) * ["nMuon>=6"],
            weight=weights,
        )

        muonsCollection = ak.zip(
            {
                "pt": muons.pt,
                "eta": muons.eta,
                "phi": muons.phi,
                "mass": muons.mass,
                "charge": muons.pdgId / (-13),
            },
            with_name="Momentum4D",
        )
        mask = SUEP_utils.inter_isolation(muonsCollection[:, 0], muonsCollection) > 1.0
        events = events[mask]
        muons = muons[mask]
        weights = weights[mask]
        output[dataset]["cutflow"].fill(
            len(events) * ["lmu_inter_iso_16>1"],
            weight=weights,
        )

        mask = ak.mean(muons.pt, axis=1) < 10
        events = events[mask]
        muons = muons[mask]
        weights = weights[mask]
        output[dataset]["cutflow"].fill(
            len(events) * ["muon_pt_mean<10"],
            weight=weights,
        )


    def fill_histograms(self, events, muons, output):
        dataset = events.metadata["dataset"]

        # Apply cuts - comment out for now to check if this is the issue...
        # cut = (muons.pt > 30) & (muons.miniPFRelIso_all < 0.1)
        # muons = muons[cut]

        # These arrays need to be broadcasted to the per muon dims from per event dims
        nMuons = ak.flatten(ak.broadcast_arrays(ak.num(muons), muons.pt)[0])
        weights = np.ones(len(events))
        muons_genPartFlav = ak.flatten(ak.zeros_like(muons.pt, dtype=int))
        if self.isMC:
            weights = events.genWeight
            muons_genPartFlav = ak.flatten(muons.genPartFlav).to_numpy().astype(int)
        weights_per_muon = ak.flatten(ak.broadcast_arrays(weights, muons.pt)[0])

        if len(events) == 0 or ak.all(ak.num(muons) == 0):
            return

        # Fill the histograms
        # Per event histograms
        output[dataset]["histograms"]["ht"].fill(
            self.ht(events),
            weight=weights,
        )
        nGenJets = np.zeros(len(events))
        if self.isMC:
            nGenJets = ak.num(events.GenJet)
        output[dataset]["histograms"]["nJet_vs_nGenJet"].fill(
            ak.num(self.clean_jets(events.Jet)),
            nGenJets,
            weight=weights,
        )
        # Per muon histograms
        output[dataset]["histograms"]["muon_pt_vs_genPartFlav_vs_nMuon"].fill(
            ak.flatten(muons.pt),
            muons_genPartFlav,
            nMuons,
            weight=weights_per_muon,
        )
        output[dataset]["histograms"]["muon_eta_vs_genPartFlav_vs_nMuon"].fill(
            ak.flatten(muons.eta),
            muons_genPartFlav,
            nMuons,
            weight=weights_per_muon,
        )
        output[dataset]["histograms"]["muon_phi_vs_genPartFlav_vs_nMuon"].fill(
            ak.flatten(muons.phi),
            muons_genPartFlav,
            nMuons,
            weight=weights_per_muon,
        )
        output[dataset]["histograms"]["muon_dxy_vs_genPartFlav_vs_nMuon"].fill(
            ak.flatten(abs(muons.dxy)),
            muons_genPartFlav,
            nMuons,
            weight=weights_per_muon,
        )
        output[dataset]["histograms"]["muon_dz_vs_genPartFlav_vs_nMuon"].fill(
            ak.flatten(abs(muons.dz)),
            muons_genPartFlav,
            nMuons,
            weight=weights_per_muon,
        )
        output[dataset]["histograms"]["muon_miniPFRelIso_vs_genPartFlav_vs_nMuon"].fill(
            ak.flatten(muons.miniPFRelIso_all),
            muons_genPartFlav,
            nMuons,
            weight=weights_per_muon,
        )
        output[dataset]["histograms"][
            "muon_btagDeepFlavB_vs_genPartFlav_vs_nMuon"
        ].fill(
            ak.flatten(ak.fill_none(muons.matched_jet.btagDeepFlavB, 0)),
            muons_genPartFlav,
            nMuons,
            weight=weights_per_muon,
        )
        # Try to include underflow values in the first bin
        output[dataset]["histograms"][
            "muon_matched_jetPtRelv2_vs_genPartFlav_vs_nMuon"
        ].fill(
            ak.flatten(ak.where(muons.jetPtRelv2 > 1e-3, muons.jetPtRelv2, 1e-3)),
            muons_genPartFlav,
            nMuons,
            weight=weights_per_muon,
        )
        output[dataset]["histograms"][
            "muon_matched_jetRelIso_vs_genPartFlav_vs_nMuon"
        ].fill(
            ak.flatten(ak.where(muons.jetRelIso > 1e-4, muons.jetRelIso, 1e-4)),
            muons_genPartFlav,
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

        # Fill the cutflow columns for nMuon>4, nMuon>=6, nMuon>=6 && inter_iso > 1.0
        self.fill_cutflow(events, muons, output)

        # Fill the histograms
        self.fill_histograms(events, muons, output)

        return

    def process(self, events):
        dataset = events.metadata["dataset"]
        cutflow = hist.Hist.new.StrCategory(
            [
                "all",
                "trigger",
                "nMuon>4",
                "nMuon>=6",
                "lmu_inter_iso_16>1",
                "muon_pt_mean<10",
            ],
            name="cutflow",
            label="cutflow",
        ).Weight()
        histograms = {
            # Per event histograms
            "ht": hist.Hist.new.Reg(
                100,
                1,
                1e4,
                name="ht",
                label="ht",
                transform=hist.axis.transform.log,
            ).Weight(),
            "nJet_vs_nGenJet": hist.Hist.new.Reg(
                20,
                0,
                20,
                name="nJet",
                label="nJet",
            )
            .Reg(
                20,
                0,
                20,
                name="nGenJet",
                label="nGenJet",
            )
            .Weight(),
            # Muon histograms - per muon entry
            # Add an axis for nMuon
            # Make sure we have the following variables:
            #  - pt
            #  - dxy
            #  - miniPFRelIso
            #  - btagDeepFlavB
            #  - matched_jetPtRelv2
            #  - matched_jetRelIso
            "muon_pt_vs_genPartFlav_vs_nMuon": hist.Hist.new.Regular(
                50,
                1,
                1e3,
                name="Muon_pt",
                label="Muon_pt",
                transform=hist.axis.transform.log,
            )
            .IntCategory(
                [0, 1, 3, 4, 5, 15], name="Muon_genPartFlav", label="Muon_genPartFlav"
            )
            .Reg(8, 0, 8, name="nMuon", label="nMuon")
            .Weight(),
            "muon_eta_vs_genPartFlav_vs_nMuon": hist.Hist.new.Regular(
                50,
                -4,
                4,
                name="Muon_eta",
                label="Muon_eta",
            )
            .IntCategory(
                [0, 1, 3, 4, 5, 15], name="Muon_genPartFlav", label="Muon_genPartFlav"
            )
            .Reg(8, 0, 8, name="nMuon", label="nMuon")
            .Weight(),
            "muon_phi_vs_genPartFlav_vs_nMuon": hist.Hist.new.Regular(
                50,
                -4,
                4,
                name="Muon_phi",
                label="Muon_phi",
            )
            .IntCategory(
                [0, 1, 3, 4, 5, 15], name="Muon_genPartFlav", label="Muon_genPartFlav"
            )
            .Reg(8, 0, 8, name="nMuon", label="nMuon")
            .Weight(),
            "muon_dxy_vs_genPartFlav_vs_nMuon": hist.Hist.new.Reg(
                100,
                1e-3,
                1,
                name="Muon_dxy",
                label="Muon_dxy",
                transform=hist.axis.transform.log,
            )
            .IntCategory(
                [0, 1, 3, 4, 5, 15], name="Muon_genPartFlav", label="Muon_genPartFlav"
            )
            .Reg(8, 0, 8, name="nMuon", label="nMuon")
            .Weight(),
            "muon_dz_vs_genPartFlav_vs_nMuon": hist.Hist.new.Reg(
                100,
                1e-5,
                1,
                name="Muon_dz",
                label="Muon_dz",
                transform=hist.axis.transform.log,
            )
            .IntCategory(
                [0, 1, 3, 4, 5, 15], name="Muon_genPartFlav", label="Muon_genPartFlav"
            )
            .Reg(8, 0, 8, name="nMuon", label="nMuon")
            .Weight(),
            "muon_miniPFRelIso_vs_genPartFlav_vs_nMuon": hist.Hist.new.Regular(
                100,
                1e-4,
                1e3,
                name="Muon_miniPFRelIso_all",
                label="Muon_miniPFRelIso_all",
                transform=hist.axis.transform.log,
            )
            .IntCategory(
                [0, 1, 3, 4, 5, 15], name="Muon_genPartFlav", label="Muon_genPartFlav"
            )
            .Reg(8, 0, 8, name="nMuon", label="nMuon")
            .Weight(),
            "muon_btagDeepFlavB_vs_genPartFlav_vs_nMuon": hist.Hist.new.Reg(
                40,
                0,
                1,
                name="matched_jet_btagDeepFlavB",
                label="matched_jet_btagDeepFlavB",
            )
            .IntCategory(
                [0, 1, 3, 4, 5, 15], name="Muon_genPartFlav", label="Muon_genPartFlav"
            )
            .Reg(8, 0, 8, name="nMuon", label="nMuon")
            .Weight(),
            "muon_matched_jetPtRelv2_vs_genPartFlav_vs_nMuon": hist.Hist.new.Reg(
                100,
                1e-3,
                1e2,
                name="matched_jetPtRelv2",
                label="matched_jetPtRelv2",
                transform=hist.axis.transform.log,
            )
            .IntCategory(
                [0, 1, 3, 4, 5, 15], name="Muon_genPartFlav", label="Muon_genPartFlav"
            )
            .Reg(8, 0, 8, name="nMuon", label="nMuon")
            .Weight(),
            "muon_matched_jetRelIso_vs_genPartFlav_vs_nMuon": hist.Hist.new.Reg(
                100,
                1e-4,
                1e3,
                name="matched_jetRelIso",
                label="matched_jetRelIso",
                transform=hist.axis.transform.log,
            )
            .IntCategory(
                [0, 1, 3, 4, 5, 15], name="Muon_genPartFlav", label="Muon_genPartFlav"
            )
            .Reg(8, 0, 8, name="nMuon", label="nMuon")
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
