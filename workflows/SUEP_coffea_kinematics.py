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


    def getTracks(self, events):
        Cands = ak.zip(
            {
                "pt": events.PFCands.trkPt,
                "eta": events.PFCands.trkEta,
                "phi": events.PFCands.trkPhi,
                "mass": events.PFCands.mass,
                "charge": events.PFCands.charge,
            },
            with_name="Momentum4D",
        )
        cut = (
            (events.PFCands.fromPV > 1)
            & (events.PFCands.trkPt >= 0.75)
            & (abs(events.PFCands.trkEta) <= 2.5)
            & (abs(events.PFCands.dz) < 10)
            & (events.PFCands.dzErr < 0.05)
        )
        Cleaned_cands = Cands[cut]
        Cleaned_cands = ak.packed(Cleaned_cands)

        # Prepare the Lost Track collection
        LostTracks = ak.zip(
            {
                "pt": events.lostTracks.pt,
                "eta": events.lostTracks.eta,
                "phi": events.lostTracks.phi,
                "mass": 0.0,
                "charge": events.lostTracks.charge,
            },
            with_name="Momentum4D",
        )
        cut = (
            (events.lostTracks.fromPV > 1)
            & (events.lostTracks.pt >= 0.75)
            & (abs(events.lostTracks.eta) <= 1.0)
            & (abs(events.lostTracks.dz) < 10)
            & (events.lostTracks.dzErr < 0.05)
        )
        Lost_Tracks_cands = LostTracks[cut]
        Lost_Tracks_cands = ak.packed(Lost_Tracks_cands)

        # select which tracks to use in the script
        # dimensions of tracks = events x tracks in event x 4 momenta
        tracks = ak.concatenate([Cleaned_cands, Lost_Tracks_cands], axis=1)

        return tracks, Cleaned_cands

    def muon_filter(self, events, iso_cut=None):
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
        if (iso_cut is not None) and (iso_cut < 99):
            clean_muons = clean_muons & (events.Muon.miniPFRelIso_all < iso_cut)

        muons = muons[clean_muons]
        select_by_muons_high = ak.num(muons, axis=-1) >= 3
        events = events[select_by_muons_high]
        muons = muons[select_by_muons_high]
        return events, muons

    
    def get_dark_photons(self, muons):
        muon1, muon2 = ak.unzip(ak.combinations(muons, 2))
        os_mask = muon1.charge != muon2.charge
        muon1, muon2 = muon1[os_mask], muon2[os_mask]
        muon1_collection = ak.zip(
            {
                "px": muon1.px,
                "py": muon1.py,
                "pz": muon1.pz,
                "mass": muon1.mass,
            },
            with_name="Momentum4D",
        )
        muon2_collection = ak.zip(
            {
                "px": muon2.px,
                "py": muon2.py,
                "pz": muon2.pz,
                "mass": muon2.mass,
            },
            with_name="Momentum4D",
        )
        dimuon_collection = muon1_collection + muon2_collection
        dimuon_mass_cut = (dimuon_collection.mass > 0.2) & (dimuon_collection.mass < 1)
        dark_photons = dimuon_collection[dimuon_mass_cut]
        return dark_photons


    def get_dark_mesons(self, dark_photons):
        dark_photon_1, dark_photon_2 = ak.unzip(ak.combinations(dark_photons, 2))
        dark_meson_cands = dark_photon_1 + dark_photon_2
        dark_mesons = dark_meson_cands[(dark_meson_cands.mass > 1) & (dark_meson_cands.mass < 10)]
        return dark_mesons


    def fill_preclustering_histograms(self, events, output):
        # Reconctruct the dark photons and dark mesons
        dataset = events.metadata["dataset"]

        # Muon iso cuts
        for muon_iso_cut in np.logspace(-2, 1, 12)[:-1]:
            events_, muons = self.muon_filter(events, iso_cut=None)
            events_inv = events_
            muons_inv = muons[(muons.miniPFRelIso_all > 1) & ((muons.miniPFRelIso_all - muons.miniPFRelIso_chg) > muon_iso_cut)]
            muons = muons[muons.miniPFRelIso_all < 1]
            if muon_iso_cut < 9:
                muons = muons[(muons.miniPFRelIso_all - muons.miniPFRelIso_chg) < muon_iso_cut]
            select_by_muons_high = ak.num(muons) >= 3
            events_ = events_[select_by_muons_high]
            muons = muons[select_by_muons_high]
            weights = self.get_weights(events_)

            dark_photons = self.get_dark_photons(muons)
            dark_mesons = self.get_dark_mesons(dark_photons)
            output[dataset]["histograms"]["b_vetoed_vs_miniPFRelIso_cut_vs_nMuon"].fill(
                False,
                muon_iso_cut * 1.01,
                ak.num(muons),
                weight=weights,
            )
            output[dataset]["histograms"]["b_vetoed_vs_miniPFRelIso_invcut_vs_nMuon"].fill(
                False,
                muon_iso_cut * 1.01,
                ak.num(muons_inv),
                weight=self.get_weights(events_inv),
            )

            if muon_iso_cut == 15:
                output[dataset]["histograms"]["nMuon_muon_miniPFRelIso_all_cut_15"].fill(
                    ak.num(muons),
                    weight=weights,
                )
                muons_miniPFRelIso_neutral = muons.miniPFRelIso_all - muons.miniPFRelIso_chg
                muons_1 = muons[muons_miniPFRelIso_neutral < 1]
                muons_0p1 = muons[muons_miniPFRelIso_neutral < 0.1]
                output[dataset]["histograms"]["nMuon_muon_miniPFRelIso_neutral_cut_1"].fill(
                    ak.num(muons_1),
                    weight=weights,
                )
                output[dataset]["histograms"]["nMuon_muon_miniPFRelIso_neutral_cut_0p1"].fill(
                    ak.num(muons_0p1),
                    weight=weights,
                )
                output[dataset]["histograms"]["muon_miniPFRelIso_neutral_vs_nMuon"].fill(
                    ak.flatten(ak.where(muons.miniPFRelIso_all - muons.miniPFRelIso_chg > 1e-4, muons.miniPFRelIso_all - muons.miniPFRelIso_chg, 1e-4)),
                    ak.flatten(ak.broadcast_arrays(ak.num(muons), muons.pt)[0]),
                    weight=ak.flatten(ak.broadcast_arrays(weights, muons.pt)[0]),
                )
                output[dataset]["histograms"]["muon_pfRelIso03_neutral_vs_nMuon"].fill(
                    ak.flatten(ak.where(muons.pfRelIso03_all - muons.pfRelIso03_chg > 1e-4, muons.pfRelIso03_all - muons.pfRelIso03_chg, 1e-4)),
                    ak.flatten(ak.broadcast_arrays(ak.num(muons), muons.pt)[0]),
                    weight=ak.flatten(ak.broadcast_arrays(weights, muons.pt)[0]),
                )

                output[dataset]["histograms"]["iso_ratio_vs_nMuon"].fill(
                    ak.flatten(ak.where(muons.miniPFRelIso_all > 0, muons.miniPFRelIso_chg / muons.miniPFRelIso_all, 1)),
                    ak.flatten(ak.broadcast_arrays(ak.num(muons), muons.pt)[0]),
                    weight=ak.flatten(ak.broadcast_arrays(weights, muons.pt)[0]),
                )
                output[dataset]["histograms"]["Muon_pfRelIso03_all_vs_Muon_pfRelIso03_chg_vs_nMuon"].fill(
                    ak.flatten(muons.pfRelIso03_all),
                    ak.flatten(muons.pfRelIso03_chg),
                    ak.flatten(ak.broadcast_arrays(ak.num(muons), muons.pt)[0]),
                    weight=ak.flatten(ak.broadcast_arrays(weights, muons.pt)[0]),
                )
                output[dataset]["histograms"]["Muon_pfRelIso03_all_vs_Muon_pfRelIso04_all_vs_nMuon"].fill(
                    ak.flatten(muons.pfRelIso03_all),
                    ak.flatten(muons.pfRelIso04_all),
                    ak.flatten(ak.broadcast_arrays(ak.num(muons), muons.pt)[0]),
                    weight=ak.flatten(ak.broadcast_arrays(weights, muons.pt)[0]),
                )
                output[dataset]["histograms"]["Muon_miniPFRelIso_all_vs_Muon_miniPFRelIso_chg_vs_nMuon"].fill(
                    ak.flatten(muons.miniPFRelIso_all),
                    ak.flatten(muons.miniPFRelIso_chg),
                    ak.flatten(ak.broadcast_arrays(ak.num(muons), muons.pt)[0]),
                    weight=ak.flatten(ak.broadcast_arrays(weights, muons.pt)[0]),
                )
                output[dataset]["histograms"]["Muon_miniPFRelIso_all_vs_Muon_pfRelIso03_all_vs_nMuon"].fill(
                    ak.flatten(muons.miniPFRelIso_all),
                    ak.flatten(muons.pfRelIso03_all),
                    ak.flatten(ak.broadcast_arrays(ak.num(muons), muons.pt)[0]),
                    weight=ak.flatten(ak.broadcast_arrays(weights, muons.pt)[0]),
                )

                output[dataset]["histograms"]["Jet_chHEF_vs_nMuon"].fill(
                    ak.flatten(events_.Jet.chHEF),
                    ak.flatten(ak.broadcast_arrays(ak.num(muons), events_.Jet.chHEF)[0]),
                    weight=ak.flatten(ak.broadcast_arrays(weights, events_.Jet.chHEF)[0]),
                )
                output[dataset]["histograms"]["Jet_muEF_vs_nMuon"].fill(
                    ak.flatten(events_.Jet.muEF),
                    ak.flatten(ak.broadcast_arrays(ak.num(muons), events_.Jet.muEF)[0]),
                    weight=ak.flatten(ak.broadcast_arrays(weights, events_.Jet.muEF)[0]),
                )
                output[dataset]["histograms"]["Jet_muonSubtrFactor_vs_nMuon"].fill(
                    ak.flatten(events_.Jet.muonSubtrFactor),
                    ak.flatten(ak.broadcast_arrays(ak.num(muons), events_.Jet.muonSubtrFactor)[0]),
                    weight=ak.flatten(ak.broadcast_arrays(weights, events_.Jet.muonSubtrFactor)[0]),
                )
                output[dataset]["histograms"]["Jet_nMuons_vs_nMuon"].fill(
                    ak.flatten(events_.Jet.nMuons),
                    ak.flatten(ak.broadcast_arrays(ak.num(muons), events_.Jet.nMuons)[0]),
                    weight=ak.flatten(ak.broadcast_arrays(weights, events_.Jet.nMuons)[0]),
                )
                output[dataset]["histograms"]["Jet_neEmEF_vs_nMuon"].fill(
                    ak.flatten(events_.Jet.neEmEF),
                    ak.flatten(ak.broadcast_arrays(ak.num(muons), events_.Jet.neEmEF)[0]),
                    weight=ak.flatten(ak.broadcast_arrays(weights, events_.Jet.neEmEF)[0]),
                )
                output[dataset]["histograms"]["Jet_neHEF_vs_nMuon"].fill(
                    ak.flatten(events_.Jet.neHEF),
                    ak.flatten(ak.broadcast_arrays(ak.num(muons), events_.Jet.neHEF)[0]),
                    weight=ak.flatten(ak.broadcast_arrays(weights, events_.Jet.neHEF)[0]),
                )
                events.Jet.chHEF

                tracks, _ = self.getTracks(events_)
                output[dataset]["histograms"]["HT_neutral_vs_HT_charged_vs_nMuon"].fill(
                    ak.sum(tracks.pt[tracks.charge == 0], axis=-1),
                    ak.sum(tracks.pt[tracks.charge != 0], axis=-1),
                    ak.num(muons),
                )
                output[dataset]["histograms"]["nPFCands_neutral_vs_nPFCands_charged_vs_nMuon"].fill(
                    ak.num(tracks[tracks.charge == 0]),
                    ak.num(tracks[tracks.charge != 0]),
                    ak.num(muons),
                )
                output[dataset]["histograms"]["iso_ratio_vs_nMuon_iso_cut"].fill(
                    ak.flatten(ak.where(muons.miniPFRelIso_all > 0, muons.miniPFRelIso_chg / muons.miniPFRelIso_all, 1)),
                    ak.flatten(ak.broadcast_arrays(ak.num(muons), muons.pt)[0]),
                    weight=ak.flatten(ak.broadcast_arrays(weights, muons.pt)[0]),
                )
                output[dataset]["histograms"]["muon_pt"].fill(
                    ak.flatten(muons.pt),
                    ak.flatten(ak.broadcast_arrays(ak.num(muons), muons.pt)[0]),
                    weight=ak.flatten(ak.broadcast_arrays(weights, muons.pt)[0]),
                )
                output[dataset]["histograms"]["muon_pt_max"].fill(
                    ak.max(muons.pt, axis=-1),
                    ak.num(muons),
                    weight=weights,
                )

            if muon_iso_cut == 99:
                output[dataset]["histograms"]["muon_miniPFRelIso_neutral_vs_nMuon"].fill(
                    ak.flatten(ak.where(muons.miniPFRelIso_all - muons.miniPFRelIso_chg > 1e-4, muons.miniPFRelIso_all - muons.miniPFRelIso_chg, 1e-4)),
                    ak.flatten(ak.broadcast_arrays(ak.num(muons), muons.pt)[0]),
                    weight=ak.flatten(ak.broadcast_arrays(weights, muons.pt)[0]),
                )
                output[dataset]["histograms"]["muon_pfRelIso03_neutral_vs_nMuon"].fill(
                    ak.flatten(ak.where(muons.pfRelIso03_all - muons.pfRelIso03_chg > 1e-4, muons.pfRelIso03_all - muons.pfRelIso03_chg, 1e-4)),
                    ak.flatten(ak.broadcast_arrays(ak.num(muons), muons.pt)[0]),
                    weight=ak.flatten(ak.broadcast_arrays(weights, muons.pt)[0]),
                )

                output[dataset]["histograms"]["iso_ratio_vs_nMuon"].fill(
                    ak.flatten(ak.where(muons.miniPFRelIso_all > 0, muons.miniPFRelIso_chg / muons.miniPFRelIso_all, 1)),
                    ak.flatten(ak.broadcast_arrays(ak.num(muons), muons.pt)[0]),
                    weight=ak.flatten(ak.broadcast_arrays(weights, muons.pt)[0]),
                )
                output[dataset]["histograms"]["Muon_pfRelIso03_all_vs_Muon_pfRelIso03_chg_vs_nMuon"].fill(
                    ak.flatten(muons.pfRelIso03_all),
                    ak.flatten(muons.pfRelIso03_chg),
                    ak.flatten(ak.broadcast_arrays(ak.num(muons), muons.pt)[0]),
                    weight=ak.flatten(ak.broadcast_arrays(weights, muons.pt)[0]),
                )
                output[dataset]["histograms"]["Muon_pfRelIso03_all_vs_Muon_pfRelIso04_all_vs_nMuon"].fill(
                    ak.flatten(muons.pfRelIso03_all),
                    ak.flatten(muons.pfRelIso04_all),
                    ak.flatten(ak.broadcast_arrays(ak.num(muons), muons.pt)[0]),
                    weight=ak.flatten(ak.broadcast_arrays(weights, muons.pt)[0]),
                )
                output[dataset]["histograms"]["Muon_miniPFRelIso_all_vs_Muon_miniPFRelIso_chg_vs_nMuon"].fill(
                    ak.flatten(muons.miniPFRelIso_all),
                    ak.flatten(muons.miniPFRelIso_chg),
                    ak.flatten(ak.broadcast_arrays(ak.num(muons), muons.pt)[0]),
                    weight=ak.flatten(ak.broadcast_arrays(weights, muons.pt)[0]),
                )
                output[dataset]["histograms"]["Muon_miniPFRelIso_all_vs_Muon_pfRelIso03_all_vs_nMuon"].fill(
                    ak.flatten(muons.miniPFRelIso_all),
                    ak.flatten(muons.pfRelIso03_all),
                    ak.flatten(ak.broadcast_arrays(ak.num(muons), muons.pt)[0]),
                    weight=ak.flatten(ak.broadcast_arrays(weights, muons.pt)[0]),
                )

                output[dataset]["histograms"]["nJet_vs_nMuon"].fill(
                    ak.num(abs(events_.Jet.eta) < 2.4),
                    ak.num(muons),
                    weight=weights,
                )
                output[dataset]["histograms"]["nFatJet_vs_nMuon"].fill(
                    ak.num(abs(events_.FatJet.eta) < 2.4),
                    ak.num(muons),
                    weight=weights,
                )
                output[dataset]["histograms"]["ht_vs_nMuon"].fill(
                    ak.sum(events_.Jet.pt[abs(events_.Jet.eta) < 2.4], axis=-1),
                    ak.num(muons),
                    weight=weights,
                )
                output[dataset]["histograms"]["tot_muon_charge_vs_nMuon"].fill(
                    ak.sum(muons.charge, axis=-1),
                    ak.num(muons),
                    weight=weights,
                )

            # B veto
            muons_jetIdx_sanitized = ak.where(muons.jetIdx >= 0, muons.jetIdx, 0)
            muon_is_not_from_b = ak.where(muons.jetIdx >= 0, events_.Jet[muons_jetIdx_sanitized].btagDeepFlavB < 0.05, True)
            muons_b_veto = muons[muon_is_not_from_b]
            at_least_three_muons = ak.num(muons_b_veto) >= 3
            events_b_veto = events_[at_least_three_muons]
            muons_b_veto = muons_b_veto[at_least_three_muons]
            weights_b_veto = weights[at_least_three_muons]

            muons_inv_jetIdx_sanitized = ak.where(muons_inv.jetIdx >= 0, muons_inv.jetIdx, 0)
            muon_inv_is_not_from_b = ak.where(muons_inv.jetIdx >= 0, events_inv.Jet[muons_inv_jetIdx_sanitized].btagDeepFlavB < 0.05, True)
            muons_inv_b_veto = muons_inv[muon_inv_is_not_from_b]

            dark_photons = self.get_dark_photons(muons_b_veto)
            dark_mesons = self.get_dark_mesons(dark_photons)
            output[dataset]["histograms"]["b_vetoed_vs_miniPFRelIso_cut_vs_nMuon"].fill(
                True,
                muon_iso_cut * 1.01,
                ak.num(muons_b_veto),
                weight=weights_b_veto,
            )
            output[dataset]["histograms"]["b_vetoed_vs_miniPFRelIso_invcut_vs_nMuon"].fill(
                True,
                muon_iso_cut * 1.01,
                ak.num(muons_inv_b_veto),
                weight=self.get_weights(events_inv),
            )
        
        # pfRelIso03_all
        for muon_iso_cut in np.logspace(-2, 1, 12)[:-1]:
            events_, muons = self.muon_filter(events, iso_cut=None)
            events_inv = events_
            muons_inv = muons[(muons.pfRelIso03_all > 1) & ((muons.pfRelIso03_all - muons.pfRelIso03_chg) > muon_iso_cut)]
            muons = muons[muons.pfRelIso03_all < 1]
            if muon_iso_cut < 9:
                muons = muons[(muons.pfRelIso03_all - muons.pfRelIso03_chg) < muon_iso_cut]
            select_by_muons_high = ak.num(muons) >= 3
            events_ = events_[select_by_muons_high]
            muons = muons[select_by_muons_high]
            weights = self.get_weights(events_)

            output[dataset]["histograms"]["b_vetoed_vs_pfRelIso03_cut_vs_nMuon"].fill(
                False,
                muon_iso_cut * 1.01,
                ak.num(muons),
                weight=weights,
            )
            output[dataset]["histograms"]["b_vetoed_vs_pfRelIso03_invcut_vs_nMuon"].fill(
                False,
                muon_iso_cut * 1.01,
                ak.num(muons_inv),
                weight=self.get_weights(events_inv),
            )

            # B veto
            muons_jetIdx_sanitized = ak.where(muons.jetIdx >= 0, muons.jetIdx, 0)
            muon_is_not_from_b = ak.where(muons.jetIdx >= 0, events_.Jet[muons_jetIdx_sanitized].btagDeepFlavB < 0.05, True)
            muons_b_veto = muons[muon_is_not_from_b]
            at_least_three_muons = ak.num(muons_b_veto) >= 3
            events_b_veto = events_[at_least_three_muons]
            muons_b_veto = muons_b_veto[at_least_three_muons]
            weights_b_veto = weights[at_least_three_muons]

            muons_inv_jetIdx_sanitized = ak.where(muons_inv.jetIdx >= 0, muons_inv.jetIdx, 0)
            muon_inv_is_not_from_b = ak.where(muons_inv.jetIdx >= 0, events_inv.Jet[muons_inv_jetIdx_sanitized].btagDeepFlavB < 0.05, True)
            muons_inv_b_veto = muons_inv[muon_inv_is_not_from_b]

            output[dataset]["histograms"]["b_vetoed_vs_pfRelIso03_cut_vs_nMuon"].fill(
                True,
                muon_iso_cut * 1.01,
                ak.num(muons_b_veto),
                weight=weights_b_veto,
            )
            output[dataset]["histograms"]["b_vetoed_vs_pfRelIso03_invcut_vs_nMuon"].fill(
                True,
                muon_iso_cut * 1.01,
                ak.num(muons_inv_b_veto),
                weight=self.get_weights(events_inv),
            )

        return


    def fill_histograms(self, events, muons, tracks, SUEP_cand, SUEP_cluster_tracks, output):
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

        # Fill the histograms
        interiso = SUEP_utils.inter_isolation(muonsCollection[:, 0], muonsCollection, dR=6.4)

        boost_SUEP = ak.zip(
            {
                "px": SUEP_cand.px * -1,
                "py": SUEP_cand.py * -1,
                "pz": SUEP_cand.pz * -1,
                "mass": SUEP_cand.mass,
            },
            with_name="Momentum4D",
        ) 

        # Sphericity for muons
        muons_b = muonsCollection.boost_p4(
            boost_SUEP
        )
        eigs = SUEP_utils.sphericity(muons_b, 1.0)
        output[dataset]["histograms"]["sph_muon_boost_vs_nMuon"].fill(
            1.5 * (eigs[:, 1] + eigs[:, 0]),
            ak.where(ak.num(muons) > 9, 9, ak.num(muons)),
            weight=weights,
        )

        # Sphericity for tracks
        tracks_b = tracks.boost_p4(
            boost_SUEP
        )
        eigs = SUEP_utils.sphericity(tracks_b, 1.0)
        output[dataset]["histograms"]["sph_tracks_boost_vs_nMuon"].fill(
            1.5 * (eigs[:, 1] + eigs[:, 0]),
            ak.where(ak.num(muons) > 9, 9, ak.num(muons)),
            weight=weights,
        )

        # Sphericity for SUEP candidate tracks
        SUEP_cluster_tracks_b = SUEP_cluster_tracks.boost_p4(
            boost_SUEP
        )
        eigs = SUEP_utils.sphericity(SUEP_cluster_tracks_b, 1.0)
        output[dataset]["histograms"]["sph_SUEPcand_tracks_boost_vs_SUEP_jet_pt_vs_nMuon"].fill(
            1.5 * (eigs[:, 1] + eigs[:, 0]),
            SUEP_cand.pt,
            ak.where(ak.num(muons) > 9, 9, ak.num(muons)),
            weight=weights,
        )

        # Get dark mesons and scalar
        scalar = ak.flatten(events.GenPart[(events.GenPart.pdgId == 25) & (events.GenPart.status == 62)])
        if len(scalar) > 0:
            scalar_collection = ak.zip(
                {
                    "px": scalar.px,
                    "py": scalar.py,
                    "pz": scalar.pz,
                    "mass": scalar.mass,
                },
                with_name="Momentum4D",
            ) 
            output[dataset]["histograms"]["suep_jet_pt_vs_beta"].fill(
                SUEP_cand.pt,
                scalar_collection.beta,
                weight=weights,
            )

            true_boost = ak.zip(
                {
                    "px": scalar.px * -1,
                    "py": scalar.py * -1,
                    "pz": scalar.pz * -1,
                    "mass": scalar.mass,
                },
                with_name="Momentum4D",
            ) 
            SUEP_cluster_tracks_tb = SUEP_cluster_tracks.boost_p4(
                true_boost
            )
            eigs = SUEP_utils.sphericity(SUEP_cluster_tracks_tb, 1.0)
            output[dataset]["histograms"]["sph_SUEPcand_tracks_true_boost_vs_nMuon"].fill(
                1.5 * (eigs[:, 1] + eigs[:, 0]),
                ak.where(ak.num(muons) > 9, 9, ak.num(muons)),
                weight=weights,
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

        # events_, muons_ = self.muon_filter(events, iso_cut=muon_iso_cut)
        self.fill_preclustering_histograms(events, output)

        # # Clustering method for sphericity calculations
        # tracks, Cleaned_cands = self.getTracks(events_)
        # ak_inclusive_jets, ak_inclusive_cluster = SUEP_utils.FastJetReclustering(
        #     tracks, r=1.5, min_pt=0
        # )

        # # Make sure there are at least 2 jets
        # at_least_two_jets = ak.num(ak_inclusive_jets) >= 2
        # events_ = events_[at_least_two_jets]
        # muons_ = muons_[at_least_two_jets]
        # tracks = tracks[at_least_two_jets]
        # ak_inclusive_cluster = ak_inclusive_cluster[at_least_two_jets]
        # ak_inclusive_jets = ak_inclusive_jets[at_least_two_jets]

        # events_, tracks, muons_, topTwoJets = SUEP_utils.getTopTwoJets(
        #     self, events_, tracks, muons_, ak_inclusive_jets, ak_inclusive_cluster
        # )
        # SUEP_cand, ISR_cand, SUEP_cluster_tracks, ISR_cluster_tracks = topTwoJets

        # # fill the histograms
        # self.fill_histograms(events_, muons_, tracks, SUEP_cand, SUEP_cluster_tracks, output)

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
            "nMuon_muon_miniPFRelIso_all_cut_15": hist.Hist.new.Reg(
                7, 3, 10, name="nMuon", label="nMuon"
            ).Weight(),
            "nMuon_muon_miniPFRelIso_neutral_cut_0p1": hist.Hist.new.Reg(
                7, 3, 10, name="nMuon", label="nMuon"
            ).Weight(),
            "nMuon_muon_miniPFRelIso_neutral_cut_1": hist.Hist.new.Reg(
                7, 3, 10, name="nMuon", label="nMuon"
            ).Weight(),
            "muon_miniPFRelIso_neutral_vs_nMuon": hist.Hist.new.Reg(
                100, 1e-4, 100, name="muon_miniPFRelIso_neutral", label="muon_miniPFRelIso_neutral", transform=hist.axis.transform.log
            ).Reg(
                7, 3, 10, name="nMuon", label="nMuon"
            ).Weight(),
            "muon_pfRelIso03_neutral_vs_nMuon": hist.Hist.new.Reg(
                100, 1e-4, 100, name="muon_pfRelIso03_neutral", label="muon_pfRelIso03_neutral", transform=hist.axis.transform.log
            ).Reg(
                7, 3, 10, name="nMuon", label="nMuon"
            ).Weight(),
            "Jet_chHEF_vs_nMuon": hist.Hist.new.Reg(
                44, 0, 1.1, name="Jet_chHEF", label="Jet_chHEF"
            ).Reg(
                7, 3, 10, name="nMuon", label="nMuon"
            ).Weight(),
            "Jet_muEF_vs_nMuon": hist.Hist.new.Reg(
                44, 0, 1.1, name="Jet_muEF", label="Jet_muEF"
            ).Reg(
                7, 3, 10, name="nMuon", label="nMuon"
            ).Weight(),
            "Jet_muonSubtrFactor_vs_nMuon": hist.Hist.new.Reg(
                44, 0, 1.1, name="Jet_muonSubtrFactor", label="Jet_muonSubtrFactor"
            ).Reg(
                7, 3, 10, name="nMuon", label="nMuon"
            ).Weight(),
            "Jet_nMuons_vs_nMuon": hist.Hist.new.Reg(
                10, 0, 10, name="Jet_nMuons", label="Jet_nMuons"
            ).Reg(
                7, 3, 10, name="nMuon", label="nMuon"
            ).Weight(),
            "Jet_neEmEF_vs_nMuon": hist.Hist.new.Reg(
                44, 0, 1.1, name="Jet_neEmEF", label="Jet_neEmEF"
            ).Reg(
                7, 3, 10, name="nMuon", label="nMuon"
            ).Weight(),
            "Jet_neHEF_vs_nMuon": hist.Hist.new.Reg(
                44, 0, 1.1, name="Jet_neHEF", label="Jet_neHEF"
            ).Reg(
                7, 3, 10, name="nMuon", label="nMuon"
            ).Weight(),

            "HT_neutral_vs_HT_charged_vs_nMuon": hist.Hist.new.Reg(
                100, 1, 1000, name="HT_neutral", label="HT_neutral", transform=hist.axis.transform.log
            ).Reg(
                100, 1, 1000, name="HT_charged", label="HT_charged", transform=hist.axis.transform.log
            ).Reg(
                7, 3, 10, name="nMuon", label="nMuon"
            ).Weight(),
            "nPFCands_neutral_vs_nPFCands_charged_vs_nMuon": hist.Hist.new.Reg(
                100, 0, 300, name="nPFCands_neutral", label="nPFCands_neutral"
            ).Reg(
                100, 0, 300, name="nPFCands_charged", label="nPFCands_charged"
            ).Reg(
                7, 3, 10, name="nMuon", label="nMuon"
            ).Weight(),
            "iso_ratio_vs_nMuon": hist.Hist.new.Reg(
                44, 0, 1.1, name="iso_ratio", label="iso_ratio",
            ).Reg(
                7, 3, 10, name="nMuon", label="nMuon"
            ).Weight(),
            "iso_ratio_vs_nMuon_iso_cut": hist.Hist.new.Reg(
                44, 0, 1.1, name="iso_ratio", label="iso_ratio",
            ).Reg(
                7, 3, 10, name="nMuon", label="nMuon"
            ).Weight(),
            "Muon_pfRelIso03_all_vs_Muon_pfRelIso03_chg_vs_nMuon": hist.Hist.new.Reg(
                100, 0.1, 100, name="Muon_pfRelIso03_all", label="Muon_pfRelIso03_all", transform=hist.axis.transform.log
            ).Reg(
                100, 0.1, 100, name="Muon_pfRelIso03_chg", label="Muon_pfRelIso03_chg", transform=hist.axis.transform.log
            ).Reg(
                7, 3, 10, name="nMuon", label="nMuon"
            ).Weight(),
            "Muon_pfRelIso03_all_vs_Muon_pfRelIso04_all_vs_nMuon": hist.Hist.new.Reg(
                100, 0.1, 100, name="Muon_pfRelIso03_all", label="Muon_pfRelIso03_all", transform=hist.axis.transform.log
            ).Reg(
                100, 0.1, 100, name="Muon_pfRelIso04_all", label="Muon_pfRelIso04_all", transform=hist.axis.transform.log
            ).Reg(
                7, 3, 10, name="nMuon", label="nMuon"
            ).Weight(),
            "Muon_miniPFRelIso_all_vs_Muon_miniPFRelIso_chg_vs_nMuon": hist.Hist.new.Reg(
                100, 0.1, 100, name="Muon_miniPFRelIso_all", label="Muon_miniPFRelIso_all", transform=hist.axis.transform.log
            ).Reg(
                100, 0.1, 100, name="Muon_miniPFRelIso_chg", label="Muon_miniPFRelIso_chg", transform=hist.axis.transform.log
            ).Reg(
                7, 3, 10, name="nMuon", label="nMuon"
            ).Weight(),
            "Muon_miniPFRelIso_all_vs_Muon_pfRelIso03_all_vs_nMuon": hist.Hist.new.Reg(
                100, 0.1, 100, name="Muon_miniPFRelIso_all", label="Muon_miniPFRelIso_all", transform=hist.axis.transform.log
            ).Reg(
                100, 0.1, 100, name="Muon_pfRelIso03_all", label="Muon_pfRelIso03_all", transform=hist.axis.transform.log
            ).Reg(
                7, 3, 10, name="nMuon", label="nMuon"
            ).Weight(),


            ############################################################
            "b_vetoed_vs_miniPFRelIso_cut_vs_nMuon": hist.Hist.new.Bool(
                name="b_vetoed", label="b_vetoed"
            ).Regular(
                11, 0.01, 10, name="muon_miniPFRelIso_all cut", 
                label="muon_miniPFRelIso_all cut",
                transform=hist.axis.transform.log,
            ).Regular(
                9, 3, 12, name="nMuon", label="nMuon"
            ).Weight(),
            "b_vetoed_vs_pfRelIso03_cut_vs_nMuon": hist.Hist.new.Bool(
                name="b_vetoed", label="b_vetoed"
            ).Regular(
                11, 0.01, 10, name="muon_pfRelIso03_all cut", 
                label="muon_pfRelIso03_all cut",
                transform=hist.axis.transform.log,
            ).Regular(
                9, 3, 12, name="nMuon", label="nMuon"
            ).Weight(),
            "b_vetoed_vs_miniPFRelIso_invcut_vs_nMuon": hist.Hist.new.Bool(
                name="b_vetoed", label="b_vetoed"
            ).Regular(
                11, 0.01, 10, name="muon_miniPFRelIso_all cut", 
                label="muon_miniPFRelIso_all cut",
                transform=hist.axis.transform.log,
            ).Regular(
                9, 3, 12, name="nMuon", label="nMuon"
            ).Weight(),
            "b_vetoed_vs_pfRelIso03_invcut_vs_nMuon": hist.Hist.new.Bool(
                name="b_vetoed", label="b_vetoed"
            ).Regular(
                11, 0.01, 10, name="muon_pfRelIso03_all cut", 
                label="muon_pfRelIso03_all cut",
                transform=hist.axis.transform.log,
            ).Regular(
                9, 3, 12, name="nMuon", label="nMuon"
            ).Weight(),
            ############################################################

            "muon_pt": hist.Hist.new.Reg(
                100, 1, 1000, name="muon_pt", label="muon_pt", transform=hist.axis.transform.log
            ).Reg(
                9, 3, 12, name="nMuon", label="nMuon"
            ).Weight(),
            "muon_pt_max": hist.Hist.new.Reg(
                100, 1, 1000, name="muon_pt_max", label="muon_pt_max", transform=hist.axis.transform.log
            ).Reg(
                9, 3, 12, name="nMuon", label="nMuon"
            ).Weight(),
            "nJet_vs_nMuon": hist.Hist.new.Reg(
                20, 0, 20, name="nJet", label="nJet"
            ).Regular(
                9, 3, 12, name="nMuon", label="nMuon"
            ).Weight(),
            "nFatJet_vs_nMuon": hist.Hist.new.Reg(
                10, 0, 10, name="nFatJet", label="nFatJet"
            ).Regular(
                9, 3, 12, name="nMuon", label="nMuon"
            ).Weight(),
            "ht_vs_nMuon": hist.Hist.new.Reg(
                50, 10, 1e4, name="ht", label="ht", transform=hist.axis.transform.log
            ).Regular(
                9, 3, 12, name="nMuon", label="nMuon"
            ).Weight(),
            "tot_muon_charge_vs_nMuon": hist.Hist.new.Reg(
                10, -5, 5, name="tot_muon_charge", label="tot_muon_charge"
            ).Regular(
                9, 3, 12, name="nMuon", label="nMuon"
            ).Weight(),
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
