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
from workflows.CMS_corrections.pileup_utils import pileup_weight
from workflows.CMS_corrections.Prefire_utils import GetPrefireWeights
from workflows.pandas_accumulator import pandas_accumulator
import workflows.SUEP_utils as SUEP_utils

# Set vector behavior
vector.register_awkward()

muon_pt_distribution = np.array(
    [
        0.01444606, 0.01788267, 0.01863509, 0.01871521, 0.018271,
        0.01862738, 0.02023799, 0.02037756, 0.02292804, 0.02308553,
        0.02105714, 0.03559662, 0.04153534, 0.03952179, 0.03809815,
        0.03558699, 0.0383288 , 0.03530175, 0.03413093, 0.03184524,
        0.02880311, 0.02495798, 0.0267085 , 0.02652325, 0.02433806,
        0.02026294, 0.02393517, 0.02388314, 0.02415238, 0.02158113,
        0.01992112, 0.01734985, 0.01589269, 0.01681835, 0.01399663,
        0.01296229, 0.01209786, 0.01085313, 0.00997262, 0.00895907,
        0.0083739 , 0.00688307, 0.00622945, 0.0060278 , 0.0052202 ,
        0.00493383, 0.0043168 , 0.00398108, 0.00303587, 0.00304323,
        0.00272003, 0.0021949 , 0.00198677, 0.00172853, 0.00148774,
        0.00160906, 0.00108399, 0.00098058, 0.00079216, 0.00083225,
        0.0006787 , 0.00054658, 0.0004822 , 0.00046058, 0.00036664,
        0.00028516, 0.00025096, 0.00022413, 0.00021782, 0.0001452 ,
        0.0001075 , 0.00010252, 0.00008649, 0.00007262, 0.00005316,
        0.00005349, 0.00004146, 0.00003346, 0.00002879, 0.0000242 ,
        0.00001898, 0.00001544, 0.00001344, 0.00001056, 0.00000816,
        0.00000598, 0.00000601, 0.000004  , 0.00000379, 0.00000275,
        0.0000019 , 0.00000157, 0.00000162, 0.00000134, 0.00000085,
        0.00000074, 0.00000055, 0.00000029, 0.00000041, 0.00000021
    ]
)
hist_muon_pt_distribution = hist.Hist.new.Reg(
    100, 3, 300, name="muon_pt_distr", label="muon_pt_distr", transform=hist.axis.transform.log
).Double()
for i in range(100):
    hist_muon_pt_distribution[i] = muon_pt_distribution[i]

class SUEP_cluster(processor.ProcessorABC):
    def __init__(
        self,
        isMC: int,
        era: str,
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
        self.era = era
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
        """
        trigger1 = np.ones(len(events), dtype=bool)
        trigger2 = np.ones(len(events), dtype=bool)
        trigger3 = np.ones(len(events), dtype=bool)
        if self.era in ["2016", "2016APV"]:
            if "TripleMu_5_3_3" in events.HLT.fields:
                trigger1 = events.HLT.TripleMu_5_3_3 == 1
            if "TripleMu_5_3_3_DZ_Mass3p8" in events.HLT.fields:
                trigger2 = events.HLT.TripleMu_5_3_3_DZ_Mass3p8 == 1
        elif self.era == "2017":
            if "TripleMu_5_3_3_Mass3p8to60_DZ" in events.HLT.fields:
                trigger1 = events.HLT.TripleMu_5_3_3_Mass3p8to60_DZ == 1
            if "TripleMu_10_5_5_DZ" in events.HLT.fields:
                trigger2 = events.HLT.TripleMu_10_5_5_DZ == 1
        elif self.era in ["2018"]:
            if "TripleMu_5_3_3_Mass3p8to60_DZ" in events.HLT.fields:
                trigger1 = events.HLT.TripleMu_5_3_3_Mass3p8to60_DZ == 1
            if "TripleMu_5_3_3_Mass3p8_DZ" in events.HLT.fields:
                trigger2 = events.HLT.TripleMu_5_3_3_Mass3p8_DZ == 1
            if "TripleMu_10_5_5_DZ" in events.HLT.fields:
                trigger3 = events.HLT.TripleMu_10_5_5_DZ == 1
        elif self.era in ["2022", "2023"]:
            if "TripleMu_5_3_3_Mass3p8_DZ" in events.HLT.fields:
                trigger1 = events.HLT.TripleMu_5_3_3_Mass3p8_DZ == 1
            if "TripleMu_10_5_5_DZ" in events.HLT.fields:
                trigger2 = events.HLT.TripleMu_10_5_5_DZ == 1
        else:
            raise ValueError("Invalid era")
        trigger = np.any(np.array([trigger1, trigger2, trigger3]).T, axis=-1)
        events = events[trigger]
        return events

    def get_weights(self, events):
        if not self.isMC:
            return np.ones(len(events))
        # Pileup weights (need to be fed with integers)
        pu_weights = pileup_weight(self.era, ak.values_astype(events.Pileup.nTrueInt, np.int32))
        # L1 prefire weights
        prefire_weights = GetPrefireWeights(events)
        # Trigger scale factors
        # To be implemented
        return events.genWeight * pu_weights * prefire_weights


    def ht(self, events):
        jet_Cut = (events.Jet.pt > 20) & (abs(events.Jet.eta) < 2.4)
        jets = events.Jet[jet_Cut]
        return ak.sum(jets.pt, axis=-1)


    def get_tracks(self, events):
        Cands = ak.zip(
            {
                "pt": events.PFCands.trkPt,
                "eta": events.PFCands.trkEta,
                "phi": events.PFCands.trkPhi,
                "mass": events.PFCands.mass,
            },
            with_name="Momentum4D",
        )
        cut = (
            (events.PFCands.fromPV > 1)
            & (events.PFCands.trkPt >= 3)
            & (abs(events.PFCands.trkEta) <= 2.5)
            & (abs(events.PFCands.dz) < 10)
            & (events.PFCands.dzErr < 0.05)
        )
        Cleaned_cands = Cands[cut]

        # Prepare the Lost Track collection
        LostTracks = ak.zip(
            {
                "pt": events.lostTracks.pt,
                "eta": events.lostTracks.eta,
                "phi": events.lostTracks.phi,
                "mass": 0.0,
            },
            with_name="Momentum4D",
        )
        cut = (
            (events.lostTracks.fromPV > 1)
            & (events.lostTracks.pt >= 3)
            & (abs(events.lostTracks.eta) <= 1.0)
            & (abs(events.lostTracks.dz) < 10)
            & (events.lostTracks.dzErr < 0.05)
        )
        Lost_Tracks_cands = LostTracks[cut]

        tracks = ak.concatenate([Cleaned_cands, Lost_Tracks_cands], axis=1)
        return tracks

    def muon_filter(self, events, muon_iso_cut=0):
        """
        Filter events after the TripleMu trigger.
        Cleans muons and electrons.
        Requires at least nMuons with mediumId, pt, dxy, dz, and eta cuts.
        """
        muons = events.Muon
        events, muons = events[ak.num(muons) > 0], muons[ak.num(muons) > 0]
        electrons = events.Electron
        clean_muons = (
            (events.Muon.mediumId)
            & (events.Muon.pt > 3)
            & (abs(events.Muon.eta) < 2.4)
            & (abs(events.Muon.dxy) <= 0.02) 
            & (abs(events.Muon.dz) <= 0.1)
        )
        if muon_iso_cut < 87:
            clean_muons = clean_muons & (events.Muon.miniPFRelIso_all < muon_iso_cut)

        clean_electrons = (
            (events.Electron.mvaFall17V2noIso_WPL)
            & (events.Electron.pt > 3)
            & (
                abs(events.Electron.dxy)
                < 0.05 + 0.05 * (abs(events.Electron.eta) > 1.479)
            )
            & (
                abs(events.Electron.dz)
                < 0.10 + 0.10 * (abs(events.Electron.eta) > 1.479)
            )
            & ((abs(events.Electron.eta) < 1.444) | (abs(events.Electron.eta) > 1.566))
            & (abs(events.Electron.eta) < 2.5)
        )
        muons = muons[clean_muons]
        electrons = electrons[clean_electrons]
        select_by_muons_high = ak.num(muons, axis=-1) >= 3
        events = events[select_by_muons_high]
        muons = muons[select_by_muons_high]
        electrons = electrons[select_by_muons_high]
        return events, electrons, muons

    def fill_histograms(self, events, electrons, muons, output, muon_iso_cut=0):
        dataset = events.metadata["dataset"]

        # These arrays need to be broadcasted to the per muon dims from per event dims
        nMuons = ak.flatten(ak.broadcast_arrays(ak.num(muons), muons.pt)[0])
        nMuons = ak.where(nMuons > 9, 9, nMuons)
        weights = self.get_weights(events)
        weights_per_muon = ak.flatten(ak.broadcast_arrays(weights, muons.pt)[0])

        if len(events) == 0 or ak.all(ak.num(muons) == 0):
            return

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
        electronsCollection = ak.zip(
            {
                "pt": electrons.pt,
                "eta": electrons.eta,
                "phi": electrons.phi,
                "mass": electrons.mass,
                "charge": electrons.pdgId / (-13),
            },
            with_name="Momentum4D",
        )
        leptonsCollection = ak.concatenate([muonsCollection, electronsCollection], axis=-1)

        # Fill the histograms
        interiso_we = SUEP_utils.inter_isolation(muonsCollection[:, 0], leptonsCollection, dR=6.4)
        interiso = SUEP_utils.inter_isolation(muonsCollection[:, 0], muonsCollection, dR=6.4)
        output[dataset]["histograms"]["lead_interiso_norm_vs_muon_iso_cut_vs_nMuon"].fill(
            interiso / (ak.num(muons) - 1),
            muon_iso_cut * np.ones_like(interiso),
            ak.where(ak.num(muons) > 9, 9, ak.where(ak.num(muons) < 3, 3, ak.num(muons))),
            weight=weights,
        )

        if muon_iso_cut > 87:
            output[dataset]["histograms"]["muon_pt"].fill(
                ak.flatten(muons.pt),
                weight=weights_per_muon,
            )

            for i, nMuon_i in enumerate(ak.num(muons.pt)):
                likelihood = 1
                likelihood_normalized = 1
                for muon_pt_j in muons[i].pt:
                    if muon_pt_j >= 300:
                        likelihood *= hist_muon_pt_distribution[299 * 1j]
                        likelihood_normalized *= hist_muon_pt_distribution[299 * 1j] * nMuon_i
                        continue
                    likelihood *= hist_muon_pt_distribution[muon_pt_j * 1j]
                    likelihood_normalized *= hist_muon_pt_distribution[muon_pt_j * 1j] * nMuon_i
                if likelihood >= 1:
                    likelihood = 0.9999999999
                elif likelihood <= 1e-15:
                    likelihood = 1.01e-15
                nMuon_fill = nMuon_i
                if nMuon_fill >= 8:
                    nMuon_fill = 7
                elif nMuon_fill <= 2:
                    nMuon_fill = 3
                output[dataset]["histograms"]["likelihood"].fill(
                    likelihood,
                    nMuon_fill,
                    weight=weights[i],
                )
                output[dataset]["histograms"]["likelihood_normalized"].fill(
                    likelihood_normalized,
                    nMuon_fill,
                    weight=weights[i],
                )

            for muon_i in range(1, 8):
                at_least_i_muons = ak.num(muons) >= muon_i
                output[dataset]["histograms"][f"mu_{muon_i}_pt"].fill(
                    muons[at_least_i_muons][:, muon_i-1].pt,
                    ak.num(muons[at_least_i_muons]),
                    weight=weights[at_least_i_muons],
                )
            
            tracks = self.get_tracks(events)
            nTrack = ak.num(tracks)
            lead_track_pt = ak.max(tracks.pt, axis=-1)
            track_interiso = ((ak.sum(tracks.pt, axis=-1) - lead_track_pt) / lead_track_pt)
            output[dataset]["histograms"]["track_interiso_norm_vs_nTrack_vs_nMuon"].fill(
                track_interiso / (nTrack - 1),
                ak.where(nTrack >= 250, 249, nTrack),
                ak.where(ak.num(muons) > 9, 9, ak.where(ak.num(muons) < 3, 3, ak.num(muons))),
                weight=weights,
            )
            output[dataset]["histograms"]["track_interiso_norm_vs_lead_interiso_norm_vs_nMuon"].fill(
                track_interiso / (nTrack - 1),
                interiso / (ak.num(muons) - 1),
                ak.where(ak.num(muons) > 9, 9, ak.where(ak.num(muons) < 3, 3, ak.num(muons))),
                weight=weights,
            )
            output[dataset]["histograms"]["track_interiso_vs_lead_interiso_vs_nMuon"].fill(
                ak.where(track_interiso >= 20, 19.9, track_interiso),
                ak.where(interiso >= 10, 9.9, interiso),
                ak.where(ak.num(muons) > 9, 9, ak.where(ak.num(muons) < 3, 3, ak.num(muons))),
                weight=weights,
            )

        # Muon_miniPFRelIso_all = ak.flatten(muons.miniPFRelIso_all)
        # Muon_jetPtRelv2 = ak.flatten(muons.jetPtRelv2)
        # Muon_jetRelIso = ak.flatten(muons.jetRelIso)
        # jet_idx = muons.jetIdx
        # jet_idx = jet_idx[jet_idx > -1]
        # Muon_Jet_btagDeepFlavB = ak.flatten(events.Jet.btagDeepFlavB[jet_idx])
        # output[dataset]["histograms"]["Muon_miniPFRelIso_all"].fill(
        #     ak.where(
        #         Muon_miniPFRelIso_all <= 0.01, 
        #         0.01, 
        #         ak.where(Muon_miniPFRelIso_all >= 10, 9.9, Muon_miniPFRelIso_all)
        #     ),
        #     weight=weights_per_muon,
        # )
        # max_muon_iso = ak.max(muons.miniPFRelIso_all, axis=-1)
        # output[dataset]["histograms"]["Muon_miniPFRelIso_all_max"].fill(
        #     ak.where(max_muon_iso >= 10, 9.9, max_muon_iso),
        #     weight=weights,
        # )
        # output[dataset]["histograms"]["Muon_jetPtRelv2"].fill(
        #     ak.where(
        #         Muon_jetPtRelv2 <= 0.01, 
        #         0.01, 
        #         ak.where(Muon_jetPtRelv2 >= 10, 9.9, Muon_jetPtRelv2)
        #     ),
        #     weight=weights_per_muon,
        # )
        # output[dataset]["histograms"]["Muon_jetRelIso"].fill(
        #     ak.where(
        #         Muon_jetRelIso <= 0.01, 
        #         0.01, 
        #         ak.where(Muon_jetRelIso >= 10, 9.9, Muon_jetRelIso)
        #     ),
        #     weight=weights_per_muon,
        # )
        # output[dataset]["histograms"]["Muon_Jet_btagDeepFlavB"].fill(
        #     Muon_Jet_btagDeepFlavB,
        #     weight=weights_per_muon[ak.flatten(jet_idx > -1)],
        # )
        # jet_Cut = (events.Jet.pt > 20) & (abs(events.Jet.eta) < 2.4)
        # jets = events.Jet[jet_Cut]
        # btag_loose = jets.btagDeepFlavB > 0.0490
        # btag_medium = jets.btagDeepFlavB > 0.2783
        # btag_tight = jets.btagDeepFlavB > 0.7100
        # output[dataset]["histograms"]["nBtagJets_loose"].fill(
        #     ak.where(ak.sum(btag_loose, axis=-1) > 9, 9, ak.sum(btag_loose, axis=-1)),
        #     ak.where(ak.num(muons) > 9, 9, ak.num(muons)),
        #     weight=weights,
        # )
        # output[dataset]["histograms"]["nBtagJets_medium"].fill(
        #     ak.where(ak.sum(btag_medium, axis=-1) > 9, 9, ak.sum(btag_medium, axis=-1)),
        #     ak.where(ak.num(muons) > 9, 9, ak.num(muons)),
        #     weight=weights,
        # )
        # output[dataset]["histograms"]["nBtagJets_tight"].fill(
        #     ak.where(ak.sum(btag_tight, axis=-1) > 9, 9, ak.sum(btag_tight, axis=-1)),
        #     ak.where(ak.num(muons) > 9, 9, ak.num(muons)),
        #     weight=weights,
        # )
        return

    def analysis(self, events, output):
        #####################################################################################
        # ---- Trigger event selection
        # Cut based on ak4 jets to replicate the trigger
        #####################################################################################

        # get dataset name
        dataset = events.metadata["dataset"]

        # take care of weights
        weights = self.get_weights(events)

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

        weights = self.get_weights(events)

        # Fill the cutflow columns for trigger
        output[dataset]["cutflow"].fill(
            len(events) * ["trigger"],
            weight=weights,
        )

        # fill the histograms
        # for the iso cut, make sure values are within the bins
        iso_values = np.logspace(-1, 2, 51)[:-1] * 1.01
        for muon_iso_cut in iso_values:
            events_, electrons_, muons_ = self.muon_filter(events, muon_iso_cut=muon_iso_cut)
            self.fill_histograms(events_, electrons_, muons_, output, muon_iso_cut=muon_iso_cut)

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
            "muon_pt": hist.Hist.new.Regular(
                100, 3, 300, name="muon_pt", label="muon_pt", transform=hist.axis.transform.log
            ).Weight(),
            "likelihood": hist.Hist.new.Regular(
                150, 1e-15, 1e0, name="likelihood", label="likelihood", transform=hist.axis.transform.log
            )
            .Regular(5, 3, 8, name="nMuon", label="nMuon").Weight(),
            "likelihood_normalized": hist.Hist.new.Regular(
                150, 1e-15, 1e0, name="likelihood", label="likelihood", transform=hist.axis.transform.log
            )
            .Regular(5, 3, 8, name="nMuon", label="nMuon").Weight(),
            "lead_interiso_norm_vs_muon_iso_cut_vs_nMuon": hist.Hist.new.Regular(
                50,
                0,
                1,
                name="mu_interiso_norm",
                label="mu_interiso_norm",
            )
            .Regular(
                50,
                0.1,
                100,
                name="muon_iso cut",
                label="muon_iso cut",
                transform=hist.axis.transform.log,
            )
            .Regular(5, 3, 8, name="nMuon", label="nMuon")
            .Weight(),

            "mu_1_pt": hist.Hist.new.Regular(
                100, 3, 300, name="mu_1_pt", label="mu_1_pt", transform=hist.axis.transform.log
            )
            .Regular(5, 3, 8, name="nMuon", label="nMuon")
            .Weight(),
            "mu_2_pt": hist.Hist.new.Regular(
                50, 3, 200, name="mu_2_pt", label="mu_2_pt", transform=hist.axis.transform.log
            )
            .Regular(5, 3, 8, name="nMuon", label="nMuon")
            .Weight(),
            "mu_3_pt": hist.Hist.new.Regular(
                50, 3, 100, name="mu_3_pt", label="mu_3_pt"
            )
            .Regular(5, 3, 8, name="nMuon", label="nMuon")
            .Weight(),
            "mu_4_pt": hist.Hist.new.Regular(
                50, 3, 100, name="mu_4_pt", label="mu_4_pt"
            )
            .Regular(5, 3, 8, name="nMuon", label="nMuon")
            .Weight(),
            "mu_5_pt": hist.Hist.new.Regular(
                50, 3, 40, name="mu_5_pt", label="mu_5_pt"
            )
            .Regular(5, 3, 8, name="nMuon", label="nMuon")
            .Weight(),
            "mu_6_pt": hist.Hist.new.Regular(
                40, 3, 20, name="mu_6_pt", label="mu_6_pt"
            )
            .Regular(5, 3, 8, name="nMuon", label="nMuon")
            .Weight(),
            "mu_7_pt": hist.Hist.new.Regular(
                40, 3, 8, name="mu_7_pt", label="mu_7_pt"
            )
            .Regular(5, 3, 8, name="nMuon", label="nMuon")
            .Weight(),

            "track_interiso_norm_vs_nTrack_vs_nMuon": hist.Hist.new.Regular(
                50,
                0,
                1,
                name="track_interiso",
                label="track_interiso",
            )
            .Regular(50, 0, 250, name="nTrack", label="nTrack")
            .Regular(5, 3, 8, name="nMuon", label="nMuon")
            .Weight(),
            "track_interiso_norm_vs_lead_interiso_norm_vs_nMuon": hist.Hist.new.Regular(
                50,
                0,
                1,
                name="track_interiso_norm",
                label="track_interiso_norm",
            )
            .Regular(
                50,
                0,
                1,
                name="mu_interiso_norm",
                label="mu_interiso_norm",
            )
            .Regular(5, 3, 8, name="nMuon", label="nMuon")
            .Weight(),
            "track_interiso_vs_lead_interiso_vs_nMuon": hist.Hist.new.Regular(
                50,
                0,
                20,
                name="track_interiso",
                label="track_interiso",
            )
            .Regular(
                50,
                0,
                10,
                name="mu_interiso",
                label="mu_interiso",
            )
            .Regular(5, 3, 8, name="nMuon", label="nMuon")
            .Weight(),

            # "Muon_miniPFRelIso_all_max": hist.Hist.new.Regular(
            #     50,
            #     0,
            #     10,
            #     name="Muon_miniPFRelIso_all_max",
            #     label="Muon_miniPFRelIso_all_max",
            # )
            # .Weight(),
            # "Muon_miniPFRelIso_all": hist.Hist.new.Regular(
            #     50,
            #     0.01,
            #     10,
            #     name="Muon_miniPFRelIso_all",
            #     label="Muon_miniPFRelIso_all",
            #     transform=hist.axis.transform.log,
            # )
            # .Weight(),
            # "Muon_jetPtRelv2": hist.Hist.new.Regular(
            #     50,
            #     0.01,
            #     10,
            #     name="Muon_jetPtRelv2",
            #     label="Muon_jetPtRelv2",
            #     transform=hist.axis.transform.log,
            # )
            # .Weight(),
            # "Muon_jetRelIso": hist.Hist.new.Regular(
            #     50,
            #     0.01,
            #     10,
            #     name="Muon_jetRelIso",
            #     label="Muon_jetRelIso",
            #     transform=hist.axis.transform.log,
            # )
            # .Weight(),
            # "Muon_Jet_btagDeepFlavB": hist.Hist.new.Regular(
            #     50,
            #     0,
            #     1,
            #     name="Muon_Jet_btagDeepFlavB",
            #     label="Muon_Jet_btagDeepFlavB",
            # )
            # .Weight(),
            # "nBtagJets_loose": hist.Hist.new.Regular(
            #     10,
            #     0,
            #     10,
            #     name="nBtagJets_loose",
            #     label="nBtagJets_loose",
            # )
            # .Regular(4, 6, 10, name="nMuon", label="nMuon")
            # .Weight(),
            # "nBtagJets_medium": hist.Hist.new.Regular(
            #     10,
            #     0,
            #     10,
            #     name="nBtagJets_medium",
            #     label="nBtagJets_medium",
            # )
            # .Regular(4, 6, 10, name="nMuon", label="nMuon")
            # .Weight(),
            # "nBtagJets_tight": hist.Hist.new.Regular(
            #     10,
            #     0,
            #     10,
            #     name="nBtagJets_tight",
            #     label="nBtagJets_tight",
            # )
            # .Regular(4, 6, 10, name="nMuon", label="nMuon")
            # .Weight(),
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
