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
        if iso_cut is not None:
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

        events_no_cut, muons_no_cut = self.muon_filter(events)
        weights_no_cut = self.get_weights(events_no_cut)
        dark_photons = self.get_dark_photons(muons_no_cut)
        dark_mesons = self.get_dark_mesons(dark_photons)
        output[dataset]["histograms"]["nMuon_vs_nDarkPho_vs_nDarkMeson"].fill(
            ak.num(muons_no_cut),
            ak.num(dark_photons),
            ak.num(dark_mesons),
            weight=weights_no_cut,
        )

        # Muon iso cuts
        for muon_iso_cut in [0.1, 0.5, 1, 5, 15]:
            events_iso_cut, muons_iso_cut = self.muon_filter(events, iso_cut=muon_iso_cut)
            weights_iso_cut = self.get_weights(events_iso_cut)
            dark_photons = self.get_dark_photons(muons_iso_cut)
            dark_mesons = self.get_dark_mesons(dark_photons)
            output[dataset]["histograms"]["iso_cut_vs_nMuon_vs_nDarkPho_vs_nDarkMeson_iso_cut"].fill(
                muon_iso_cut * 1.01,
                ak.num(muons_iso_cut),
                ak.num(dark_photons),
                ak.num(dark_mesons),
                weight=weights_iso_cut,
            )

        # B veto
        events_b_veto, muons_b_veto = self.muon_filter(events)
        muons_jetIdx_sanitized = ak.where(muons_b_veto.jetIdx >= 0, muons_b_veto.jetIdx, 0)
        muon_is_not_from_b = ak.where(muons_b_veto.jetIdx >= 0, events_b_veto.Jet[muons_jetIdx_sanitized].btagDeepFlavB < 0.05, True)
        muons_b_veto = muons_b_veto[muon_is_not_from_b]
        at_least_three_muons = ak.num(muons_b_veto) >= 3
        events_b_veto = events_b_veto[at_least_three_muons]
        muons_b_veto = muons_b_veto[at_least_three_muons]
        weights_b_veto = self.get_weights(events_b_veto)
        dark_photons = self.get_dark_photons(muons_b_veto)
        dark_mesons = self.get_dark_mesons(dark_photons)
        output[dataset]["histograms"]["nMuon_vs_nDarkPho_vs_nDarkMeson_b_veto"].fill(
            ak.num(muons_b_veto),
            ak.num(dark_photons),
            ak.num(dark_mesons),
            weight=weights_b_veto,
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
            "b_vetoed_vs_iso_cut_vs_nMuon_vs_nDarkPho_vs_nDarkMeson": hist.Hist.new.Bool(
                name="b_vetoed", label="b_vetoed"
            ).Variable(
                [0.1, 0.5, 1, 5, 15, 99, 100], 
                name="muon_miniPFRelIso_all cut", 
                label="muon_miniPFRelIso_all cut"
            ).Reg(
                9, 3, 12, name="nMuon", label="nMuon"
            ).Regular(
                10, 0, 10, name="nDarkPho", label="nDarkPho"
            ).Regular(
                10, 0, 10, name="nDarkMeson", label="nDarkMeson"
            ).Weight(),

            # "muon_jet_btagDeepFlavB": hist.Hist.new.Reg(
            #     100, 0, 1, name="muon_jet_btagDeepFlavB", label="muon_jet_btagDeepFlavB"
            # ).Weight(),
            # "os_dimuon_mass_vs_deltaR": hist.Hist.new.Reg(
            #     100, 0.01, 100, name="os_dimuon_mass", label="os_dimuon_mass", transform=hist.axis.transform.log
            # ).Reg(
            #     100, 0, 5, name="os_dimuon_deltaR", label="os_dimuon_deltaR"
            # ).Weight(),
            # "diDarkPho_mass_vs_deltaR": hist.Hist.new.Reg(
            #     100, 0.01, 100, name="diDarkPho_mass", label="diDarkPho_mass", transform=hist.axis.transform.log
            # ).Reg(
            #     100, 0, 5, name="diDarkPho_deltaR", label="diDarkPho_deltaR"
            # ).Weight(),
            # "nDarkPho_vs_nDarkMeson_vs_nMuon": hist.Hist.new.Regular(
            #     10, 0, 10, name="nDarkPho", label="nDarkPho"
            # ).Regular(
            #     10, 0, 10, name="nDarkMeson", label="nDarkMeson"
            # ).Regular(
            #     10, 0, 10, name="nMuon", label="nMuon"
            # ).Weight(),
            # "sph_muon_boost_vs_nMuon": hist.Hist.new.Regular(
            #     100, 0, 1, name="sph_muon_boost", label="sph_muon_boost"
            # ).Regular(7, 3, 10, name="nMuon", label="nMuon").Weight(),
            # "sph_tracks_boost_vs_nMuon": hist.Hist.new.Regular(
            #     100, 0, 1, name="sph_tracks_boost", label="sph_tracks_boost"
            # ).Regular(7, 3, 10, name="nMuon", label="nMuon").Weight(),
            # "sph_SUEPcand_tracks_boost_vs_SUEP_jet_pt_vs_nMuon": hist.Hist.new.Regular(
            #     100, 0, 1, name="sph_SUEPcand_tracks_boost", label="sph_SUEPcand_tracks_boost"
            # ).Regular(
            #     100, 0, 1000, name="SUEP_jet_pt", label="SUEP_jet_pt"
            # ).Regular(
            #     7, 3, 10, name="nMuon", label="nMuon"
            # ).Weight(),
            # "suep_jet_pt_vs_beta": hist.Hist.new.Regular(
            #     100, 0, 1000, name="suep_jet_pt", label="suep_jet_pt"
            # ).Regular(100, 0, 1, name="beta", label="beta").Weight(),
            # "sph_SUEPcand_tracks_true_boost_vs_nMuon": hist.Hist.new.Regular(
            #     100, 
            #     0, 
            #     1, 
            #     name="sph_SUEPcand_tracks_true_boost", 
            #     label="sph_SUEPcand_tracks_true_boost"
            # ).Regular(
            #     7, 3, 10, name="nMuon", label="nMuon"
            # ).Weight(),
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
