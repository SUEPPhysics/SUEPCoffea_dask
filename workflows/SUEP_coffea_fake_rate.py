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
from rich.pretty import pprint

import numba as nb

# Importing CMS corrections
from workflows.CMS_corrections.golden_jsons_utils import applyGoldenJSON
from workflows.CMS_corrections.pileup_utils import pileup_weight
from workflows.CMS_corrections.Prefire_utils import GetPrefireWeights
from workflows.pandas_accumulator import pandas_accumulator
import workflows.SUEP_utils as SUEP_utils

# Set vector behavior
vector.register_awkward()

@nb.njit
def numba_n_unique(flat_array, starts, stops, placeholder):
    result = np.empty(len(starts), dtype=np.int64)
    
    # Loop over each sublist
    for i in range(len(starts)):
        seen = set()  # Set to track unique elements
        for j in range(starts[i], stops[i]):  # Loop over elements of the sublist
            elem = flat_array[j]
            if elem != placeholder:  # Skip placeholder values (e.g., -1)
                seen.add(elem)
        result[i] = len(seen)  # Store the count of unique elements
    
    return result

def n_unique(ak_array):
    # Flatten the awkward array
    # Use a placeholder (e.g., -1) for None values
    flat_array = ak.fill_none(ak.flatten(ak_array), -1)
    flat_array = np.array(flat_array)  # Convert to numpy array
    
    # Get the start and stop positions for each sublist and convert them to numpy arrays
    layout = ak_array.layout
    starts = np.array(layout.starts)
    stops = np.array(layout.stops)
    
    # Call numba function to count unique elements
    unique_counts = numba_n_unique(flat_array, starts, stops, -1)
    
    # Return result as an awkward array
    return ak.Array(unique_counts)

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

        clean_muons = (
            (events.Muon.mediumId)
            & (events.Muon.pt > 3)
            & (abs(events.Muon.eta) < 2.4)
            & (abs(events.Muon.dxy) <= 0.02) 
            & (abs(events.Muon.dz) <= 0.1)
            & ((muons.miniPFRelIso_all) < 1) 
            & ((muons.miniPFRelIso_all - muons.miniPFRelIso_chg) < 0.1) 

        )
        if (iso_cut is not None) and (iso_cut < 99):
            clean_muons = clean_muons & (events.Muon.miniPFRelIso_all < iso_cut)

        muons = muons[clean_muons]
        select_by_muons_high = ak.num(muons, axis=-1) >= 3
        events = events[select_by_muons_high]
        muons = muons[select_by_muons_high]
        return events, muons


    def match_muons_to_jets(self, jets, muons, delta_r=0.4):
        """
        Will match muons to each jet after checking the delta_r between them.
        """
        dr_jets_to_muons = jets.metric_table(muons)
        muons_per_jet = ak.unzip(ak.cartesian([jets, muons], nested=True))[1]
        muons_per_jet = muons_per_jet[dr_jets_to_muons < delta_r]
        return muons_per_jet


    def mini_iso_cone_size(self, muons):
        """
        Calculate the miniPFRelIso_all cone size for each muon.
        This is 0.2 for muons with pt <= 50, 0.05 for muons with pt >= 200, and 10/pt for muons with 50 < pt < 200.
        """
        return ak.where(
            muons.pt <= 50, 
            0.2, 
            ak.where(muons.pt >= 200, 0.05, 10 / muons.pt)
        )


    def muon_iso_sub_muons(self, muons_for_iso, muons_all):
        """
        Subtract the isolation of other muons from the isolation of the muon.
        """
        dR_muons = ak.fill_none(muons_for_iso.metric_table(muons_all), [], axis=0)
        muons_for_iso_broadcasted, muons_in_cone = ak.unzip(ak.cartesian([muons_for_iso, muons_all], nested=True))
        cone_size = self.mini_iso_cone_size(muons_for_iso_broadcasted)
        muons_in_cone = muons_in_cone[dR_muons < cone_size]
        muon_contributions = (
            ak.sum(muons_in_cone.pt, axis=-1) - muons_for_iso.pt
        ) / muons_for_iso.pt
        return muons_for_iso.miniPFRelIso_all - muon_contributions


    def b_veto(self, events, muons, threshold):
        muons_jetIdx_sanitized = ak.where(muons.jetIdx >= 0, muons.jetIdx, 0)
        muon_is_not_from_b = ak.where(muons.jetIdx >= 0, events.Jet[muons_jetIdx_sanitized].btagDeepFlavB < threshold, True)
        tight_cuts = (
            ((muons.miniPFRelIso_all) < 1) 
            & ((muons.miniPFRelIso_all - muons.miniPFRelIso_chg) < 0.1) 
            & muon_is_not_from_b
        )
        return ak.sum(tight_cuts, axis=-1)


    def ip3d_cut(self, muons, threshold):
        tight_cuts = (
            ((muons.miniPFRelIso_all) < 1) 
            & ((muons.miniPFRelIso_all - muons.miniPFRelIso_chg) < 0.1) 
            & (muons.ip3d < threshold)
        )
        return ak.sum(tight_cuts, axis=-1)


    def get_num_muons_in_associated_jet(self, muons):
        # Get sorted jet idxs & their run lengths
        sorted_jetIdx = ak.sort(muons.jetIdx)
        run_lengths = ak.run_lengths(ak.materialized(sorted_jetIdx))

        # Group the idxs that have the same value
        grouped_idxs = ak.unflatten(
            ak.unflatten(ak.flatten(sorted_jetIdx), ak.flatten(run_lengths)),
            ak.num(run_lengths)
        )

        # Get the length of each idx group and broadcast it so that each  
        # muon has a number that corresponds to the length of its group
        num_muons_in_associated_jet = ak.flatten(
            ak.broadcast_arrays(ak.num(grouped_idxs, axis=-1), grouped_idxs)[0], 
            axis=-1
        )

        # Get the sorted local idxs
        sorted_local_idx = ak.local_index(muons.jetIdx)[ak.argsort(muons.jetIdx)]

        # Unsort the numbers and replace the number for -1 idxs with 0
        num_muons_in_associated_jet = num_muons_in_associated_jet[sorted_local_idx]
        num_muons_in_associated_jet = ak.where(muons.jetIdx == -1, 0, num_muons_in_associated_jet)
        
        return num_muons_in_associated_jet


    def fill_preclustering_histograms(self, events, output):
        # Reconctruct the dark photons and dark mesons
        dataset = events.metadata["dataset"]

        events_, muons = self.muon_filter(events, iso_cut=None)
        
        # Get the muons in each jet
        jets = events_.Jet[(events_.Jet.pt > 15) & (abs(events_.Jet.eta) < 2.4)]
        muons_per_jet = self.match_muons_to_jets(jets, muons, delta_r=0.4)

        # Select events with at most 2 muons per jet
        up_to_two_muons_per_jet = ~ak.any(ak.num(muons_per_jet, axis=-1) > 2, axis=-1)
        events_ = events_[up_to_two_muons_per_jet]
        muons = muons[up_to_two_muons_per_jet]
        jets = jets[up_to_two_muons_per_jet]
        muons_per_jet = muons_per_jet[up_to_two_muons_per_jet]

        # # # Select jets with exactly N=0 muons and drop events without any jets left
        # # muons_per_jet = muons_per_jet[ak.num(muons_per_jet, axis=-1) == 3]
        # # events_ = events_[ak.num(muons_per_jet, axis=1) > 0]
        # # muons = muons[ak.num(muons_per_jet, axis=1) > 0]
        # # jets = jets[ak.num(muons_per_jet, axis=1) > 0]
        # # muons_per_jet = muons_per_jet[ak.num(muons_per_jet, axis=1) > 0]

        # # Instead, limit the number of muons inside jets
        # limit_jets_with_muons = ak.all(ak.num(muons_per_jet, axis=-1) <= 1, axis=-1)
        # # # Make sure all muons are inside jets
        # # limit_jets_with_muons = limit_jets_with_muons & ak.all(muons.jetIdx >= 0, axis=-1)
        # events_ = events_[limit_jets_with_muons]
        # muons = muons[limit_jets_with_muons]
        # jets = jets[limit_jets_with_muons]
        # muons_per_jet = muons_per_jet[limit_jets_with_muons]

        # # Make selection based on exact number of muons in the jets
        # limit_jets_with_muons = ak.all(ak.num(muons_per_jet, axis=-1) <= 1, axis=-1)
        # # # Make sure all muons are inside jets
        # # limit_jets_with_muons = limit_jets_with_muons & ak.all(muons.jetIdx >= 0, axis=-1)
        # events_ = events_[limit_jets_with_muons]
        # muons = muons[limit_jets_with_muons]
        # jets = jets[limit_jets_with_muons]
        # muons_per_jet = muons_per_jet[limit_jets_with_muons]

        # Get the number of muons in each jet associated with each muon
        num_muons_in_associated_jet = self.get_num_muons_in_associated_jet(muons)

        if len(events_) == 0:
            return

        weights = self.get_weights(events_)


        keep_jets_with_2_muons = ak.num(muons_per_jet, axis=-1) == 2
        muons_per_2muon_jet = muons_per_jet[keep_jets_with_2_muons]

        tight_muons_per_2muon_jet = muons_per_2muon_jet[muons_per_2muon_jet.ip3d < 0.01]
        inverse_muons_per_2muon_jet = muons_per_2muon_jet[muons_per_2muon_jet.ip3d >= 0.01]

        output[dataset]["histograms"]["nMuon_tight_per_jet_vs_nMuon_inverse_per_jet"].fill(
            ak.flatten(ak.num(tight_muons_per_2muon_jet, axis=-1)),
            ak.flatten(ak.num(inverse_muons_per_2muon_jet, axis=-1)),
            weight=ak.flatten(ak.broadcast_arrays(weights, ak.num(muons_per_2muon_jet, axis=-1))[0]),
        )


        output[dataset]["histograms"]["num_muons_in_associated_jet_vs_nMuon"].fill(
            ak.flatten(num_muons_in_associated_jet),
            ak.flatten(ak.broadcast_arrays(ak.num(muons), num_muons_in_associated_jet)[0]),
            weight=ak.flatten(ak.broadcast_arrays(weights, num_muons_in_associated_jet)[0]),
        )
        output[dataset]["histograms"]["nMuons_per_jet_vs_nMuon"].fill(
            ak.flatten(ak.num(muons_per_jet, axis=-1)),
            ak.flatten(ak.broadcast_arrays(ak.num(muons), ak.num(muons_per_jet, axis=-1))[0]),
            weight=ak.flatten(ak.broadcast_arrays(weights, ak.num(muons_per_jet, axis=-1))[0]),
        )

        ########################################################################
        # Section 1: fill nImT histograms for the events
        ########################################################################

        # Get isolation with respect to other muons
        # dr_table = muons.metric_table(muons)
        # counts = ak.num(muons.pt)
        # index = ak.local_index(muons.pt)
        # repeated_index = ak.unzip(ak.cartesian([index, index], axis=1))[1]
        # counts_repeated = ak.flatten(ak.broadcast_arrays(counts, muons.pt)[0])
        # pt_table = ak.unflatten(muons.pt[repeated_index], counts_repeated, axis=1)
        # muon_pfRelIso03_mu = (ak.sum(pt_table[dr_table < 0.2], axis=-1) - muons.pt) / muons.pt

        # muons = muons[ak.sum(dr_table < 0.8, axis=-1) == 1]

        # # Unpack muons 1st, 2nd, 3rd, etc, and subtract the isolation with respect to the other muons from the total isolation
        # unpacked_muons_iso_sub_muons = []
        # for iMuon in range(ak.max(ak.num(muons))):
        #     muon_i = muons[:, iMuon:(iMuon+1)]
        #     muon_i_iso_sub_muons = self.muon_iso_sub_muons(muon_i, muons)
        #     unpacked_muons_iso_sub_muons.append(muon_i_iso_sub_muons)
        # muons_iso_sub_muons = ak.concatenate(unpacked_muons_iso_sub_muons, axis=-1)

        # # Cut on the muon subtracted isolation
        # tight_cuts = muons_iso_sub_muons < 0.3
        # inverted_cuts = muons_iso_sub_muons >= 0.3

        # Full cuts
        # # B veto
        # muons_jetIdx_sanitized = ak.where(muons.jetIdx >= 0, muons.jetIdx, 0)
        # muon_is_not_from_b = ak.where(muons.jetIdx >= 0, events_.Jet[muons_jetIdx_sanitized].btagDeepFlavB < 0.05, True)

        tight_cuts = (
            (muons.ip3d < 0.01)
        )
        inverted_cuts = (
            (muons.ip3d >= 0.01)
        )

        output[dataset]["histograms"]["nMuon_inv_in_vs_nMuon_tight_in_vs_nMuon_inv_out_vs_nMuon_tight_out_vs_nMuon"].fill(
            ak.num(muons[inverted_cuts & (muons.jetIdx >= 0)]),
            ak.num(muons[tight_cuts & (muons.jetIdx >= 0)]),
            ak.num(muons[inverted_cuts & (muons.jetIdx == -1)]),
            ak.num(muons[tight_cuts & (muons.jetIdx == -1)]),
            ak.num(muons),
            weight=weights,
        )

        output[dataset]["histograms"]["nMuon_inv_vs_nMuon_tight_vs_nMuon"].fill(
            ak.num(muons[inverted_cuts]),
            ak.num(muons[tight_cuts]),
            ak.num(muons),
            weight=weights,
        )

        output[dataset]["histograms"]["nMuon_inv_0_nMuon_tight_0_through_4_vs_nMuon"].fill(
            ak.num(muons[inverted_cuts & (num_muons_in_associated_jet == 0)]),
            ak.num(muons[tight_cuts & (num_muons_in_associated_jet == 0)]),
            ak.num(muons[inverted_cuts & (num_muons_in_associated_jet == 1)]),
            ak.num(muons[tight_cuts & (num_muons_in_associated_jet == 1)]),
            ak.num(muons[inverted_cuts & (num_muons_in_associated_jet == 2)]),
            ak.num(muons[tight_cuts & (num_muons_in_associated_jet == 2)]),
            # ak.num(muons[inverted_cuts & (num_muons_in_associated_jet == 3)]),
            # ak.num(muons[tight_cuts & (num_muons_in_associated_jet == 3)]),
            # ak.num(muons[inverted_cuts & (num_muons_in_associated_jet == 4)]),
            # ak.num(muons[tight_cuts & (num_muons_in_associated_jet == 4)]),
            ak.num(muons),
            weight=weights,
        )

        muons_per_jet_with_2_muons = muons_per_jet[ak.num(muons_per_jet, axis=-1) == 2]
        nJets_with_2_tights = ak.sum(ak.sum(muons_per_jet_with_2_muons.ip3d < 0.01, axis=-1) == 2, axis=-1)
        nJets_with_2_inverse = ak.sum(ak.sum(muons_per_jet_with_2_muons.ip3d < 0.01, axis=-1) == 0, axis=-1)
        nJets_with_1_tight_1_inverse = ak.sum(ak.sum(muons_per_jet_with_2_muons.ip3d < 0.01, axis=-1) == 1, axis=-1)

        output[dataset]["histograms"]["full_method_hist"].fill(
            ak.num(muons[tight_cuts & (num_muons_in_associated_jet < 2)]),
            ak.num(muons[inverted_cuts & (num_muons_in_associated_jet < 2)]),
            nJets_with_2_tights,
            nJets_with_2_inverse,
            nJets_with_1_tight_1_inverse,
            ak.num(muons),
            weight=weights,
        )

        for pt in np.logspace(np.log10(3), np.log10(300), 10)[:-1]:
            muons_ = muons[(num_muons_in_associated_jet < 2) & (muons.pt >= pt) & (muons.pt < (pt + 2))]
            output[dataset]["histograms"]["check_pt_dependency"].fill(
                pt * 1.01,
                ak.num(muons_[muons_.ip3d < 0.01]),
                ak.num(muons_[muons_.ip3d >= 0.01]),
                ak.num(muons_),
                weight=weights,
            )

        ########################################################################
        # ROC curves for btag vs muon ip3d
        ########################################################################
        # btag_thresholds = np.linspace(0, 1, 51)[:-1]
        # ip3d_thresholds = np.logspace(-4, -1, 51)[:-1]

        # for btag_threshold, ip3d_threshold in zip(btag_thresholds, ip3d_thresholds):
        #     nTight_bveto = self.b_veto(events_, muons, btag_threshold)
        #     nTight_ip3d = self.ip3d_cut(muons, ip3d_threshold)

        #     output[dataset]["histograms"]["btag_score_cut_vs_nTights_vs_nMuon"].fill(
        #         btag_threshold*1.01,
        #         nTight_bveto,
        #         ak.num(muons),
        #         weight=weights,
        #     )

        #     output[dataset]["histograms"]["ip3d_cut_vs_nTights_vs_nMuon"].fill(
        #         ip3d_threshold*1.01,
        #         nTight_ip3d,
        #         ak.num(muons),
        #         weight=weights,
        #     )


        ########################################################################
        # Section 2: fill histogram for nMuon vs nJet with muon
        ########################################################################

        # Description: 
        #   1. For each muon calculate dR with all jets in the event
        #   2. Reduce innermost axis by selecting closest jet to each muon (minimum dR)
        #   3. Get the index of that closest jet
        #   4. only keep jets that have dR < 0.4 from the muon
        # dr_table = muons.metric_table(jets)
        # dr_closest_jet = ak.min(dr_table, axis=-1)
        # closest_jet_idx = ak.argmin(dr_table, axis=-1)
        # closest_jet_idx = closest_jet_idx[dr_closest_jet < 0.4]

        # output[dataset]["histograms"]["nMuon_vs_nJet_with_muon"].fill(
        #     ak.num(muons),
        #     n_unique(closest_jet_idx),
        #     weight=weights,
        # )

        ########################################################################
        # Section 3: fill nImT histograms for the jets
        ########################################################################

        # # Get muons in the jets
        # muon_0 = muons_per_jet[:, :, 0]
        # muon_1 = muons_per_jet[:, :, 1]
        # muon_2 = muons_per_jet[:, :, 2]

        # # Plain iso cut
        # nTight = (
        #     ak.values_astype(muon_0.miniPFRelIso_all < 1, np.int32)
        #     + ak.values_astype(muon_1.miniPFRelIso_all < 1, np.int32)
        #     + ak.values_astype(muon_2.miniPFRelIso_all < 1, np.int32)
        # )
        # nInverse = (
        #     ak.values_astype(muon_0.miniPFRelIso_all >= 1, np.int32) 
        #     + ak.values_astype(muon_1.miniPFRelIso_all >= 1, np.int32)
        #     + ak.values_astype(muon_2.miniPFRelIso_all >= 1, np.int32)
        # )

        # # Subtract other muons from iso
        # muon_0_iso_sub_muons = self.muon_iso_sub_muons(muon_0, muons)
        # muon_1_iso_sub_muons = self.muon_iso_sub_muons(muon_1, muons)
        # muon_2_iso_sub_muons = self.muon_iso_sub_muons(muon_2, muons)
        
        # nTight = (
        #     ak.values_astype(muon_0_iso_sub_muons < 0.3, np.int32)
        #     + ak.values_astype(muon_1_iso_sub_muons < 0.3, np.int32)
        #     + ak.values_astype(muon_2_iso_sub_muons < 0.3, np.int32)
        # )
        # nInverse = (
        #     ak.values_astype(muon_0_iso_sub_muons >= 0.3, np.int32)
        #     + ak.values_astype(muon_1_iso_sub_muons >= 0.3, np.int32)
        #     + ak.values_astype(muon_2_iso_sub_muons >= 0.3, np.int32)
        # )

        # # Neutral iso cut
        # muon_0_jetIdx_sanitized = ak.where(muon_0.jetIdx >= 0, muon_0.jetIdx, 0)
        # muon_0_is_not_from_b = ak.where(muon_0.jetIdx >= 0, events_.Jet[muon_0_jetIdx_sanitized].btagDeepFlavB < 0.05, True)
        # muon_1_jetIdx_sanitized = ak.where(muon_1.jetIdx >= 0, muon_1.jetIdx, 0)
        # muon_1_is_not_from_b = ak.where(muon_1.jetIdx >= 0, events_.Jet[muon_1_jetIdx_sanitized].btagDeepFlavB < 0.05, True)
        # muon_2_jetIdx_sanitized = ak.where(muon_2.jetIdx >= 0, muon_2.jetIdx, 0)
        # muon_2_is_not_from_b = ak.where(muon_2.jetIdx >= 0, events_.Jet[muon_2_jetIdx_sanitized].btagDeepFlavB < 0.05, True)
        # tight_cut_0 = (
        #     ((muon_0.miniPFRelIso_all - muon_0.miniPFRelIso_chg) < 0.1) & 
        #     (muon_0.miniPFRelIso_all < 1) &
        #     muon_0_is_not_from_b
        # )
        # tight_cut_1 = (
        #     ((muon_1.miniPFRelIso_all - muon_1.miniPFRelIso_chg) < 0.1) &
        #     (muon_1.miniPFRelIso_all < 1) &
        #     muon_1_is_not_from_b
        # )
        # tight_cut_2 = (
        #     ((muon_2.miniPFRelIso_all - muon_2.miniPFRelIso_chg) < 0.1) &
        #     (muon_2.miniPFRelIso_all < 1) &
        #     muon_2_is_not_from_b
        # )
        # inverse_cut_0 = (
        #     ((muon_0.miniPFRelIso_all - muon_0.miniPFRelIso_chg) >= 0.1) |
        #     (muon_0.miniPFRelIso_all >= 1) |
        #     ~muon_0_is_not_from_b
        # )
        # inverse_cut_1 = (
        #     ((muon_1.miniPFRelIso_all - muon_1.miniPFRelIso_chg) >= 0.1) |
        #     (muon_1.miniPFRelIso_all >= 1) |
        #     ~muon_1_is_not_from_b
        # )
        # inverse_cut_2 = (
        #     ((muon_2.miniPFRelIso_all - muon_2.miniPFRelIso_chg) >= 0.1) |
        #     (muon_2.miniPFRelIso_all >= 1) |
        #     ~muon_2_is_not_from_b
        # )
        # nTight = (
        #     ak.values_astype(tight_cut_0, np.int32)
        #     + ak.values_astype(tight_cut_1, np.int32)
        #     + ak.values_astype(tight_cut_2, np.int32)
        # )
        # nInverse = (
        #     ak.values_astype(inverse_cut_0, np.int32)
        #     + ak.values_astype(inverse_cut_1, np.int32)
        #     + ak.values_astype(inverse_cut_2, np.int32)
        # )

        # output[dataset]["histograms"]["nTight_in_jet_vs_nInverse_in_jet_vs_nMuon"].fill(
        #     ak.flatten(nTight),
        #     ak.flatten(nInverse),
        #     ak.flatten(ak.broadcast_arrays(ak.num(muons), nTight)[0]),
        #     weight=ak.flatten(ak.broadcast_arrays(weights, nTight)[0]),
        # )

        ########################################################################
        # Section 4: printout for event displays
        ########################################################################

        # # Printout for event displays
        # local_index = ak.local_index(events_, axis=0)[ak.any((nTight == 3), axis=-1) & (ak.num(muons) == 5)]
        # events_for_printout = events_[ak.any((nTight == 3), axis=-1) & (ak.num(muons) == 5)]
        # if len(events_for_printout) > 0:
        #     for iEvt, old_index in zip(range(len(events_for_printout)), local_index):
        #         print(
        #             "Event 0T3I --> ",
        #             "dataset:", dataset[:15],
        #             "run:", events_for_printout.run[iEvt],
        #             "lumi:", events_for_printout.luminosityBlock[iEvt],
        #             "event:", events_for_printout.event[iEvt],
        #             f"\n  muon_0: {muon_0[old_index].miniPFRelIso_all[0]:.2f} -> {muon_0_iso_sub_muons[old_index][0]:.2f}",
        #             f", muon_1: {muon_1[old_index].miniPFRelIso_all[0]:.2f} -> {muon_1_iso_sub_muons[old_index][0]:.2f}",
        #             f", muon_2: {muon_2[old_index].miniPFRelIso_all[0]:.2f} -> {muon_2_iso_sub_muons[old_index][0]:.2f}",
        #             f"\n  {muons_per_jet[:, 0].metric_table(muons_per_jet[:, 0])[old_index].to_numpy()}",
        #             f"\n  {muons_iso_sub_muons[old_index]}",
        #             f"\n  {muons[old_index].miniPFRelIso_all}",
        #             flush=True
        #         )
        #         pprint(muons_per_jet[:, 0].metric_table(muons_per_jet[:, 0])[old_index].to_numpy())

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

        self.fill_preclustering_histograms(events, output)

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
            # "muon_pt_tight": hist.Hist.new.Regular(
            #     100, 0, 100, name="muon_pt_tight", label="muon_pt_tight"
            # ).Weight(),
            # "muon_pt_inv": hist.Hist.new.Regular(
            #     100, 0, 100, name="muon_pt_inv", label="muon_pt_inv"
            # ).Weight(),
            # "jet_pt_3I0T": hist.Hist.new.Regular(
            #     100, 0, 100, name="jet_pt_3I0T", label="jet_pt_3I0T"
            # ).Weight(),
            # "jet_pt_0I3T": hist.Hist.new.Regular(
            #     100, 0, 100, name="jet_pt_0I3T", label="jet_pt_0I3T"
            # ).Weight(),
            # "muon_iso_sub_muons_vs_nMuon": hist.Hist.new.Regular(
            #     100, 1e-2, 100, 
            #     name="muon_iso_sub_muons", 
            #     label="muon_iso_sub_muons", 
            #     transform=hist.axis.transform.log
            # ).Regular(
            #     100, 1e-2, 100, 
            #     name="muon_iso", 
            #     label="muon_iso", 
            #     transform=hist.axis.transform.log
            # ).Regular(
            #     100, 1e-2, 100, 
            #     name="muon_iso_chg", 
            #     label="muon_iso_chg", 
            #     transform=hist.axis.transform.log
            # # ).Regular(
            # #     5, 3, 8, name="nMuon", label="nMuon"
            # ).Weight(),

            # Histogram for full method
            # Stores: 
            #   - number of tight muons in the event that are not in jets with 2 muons
            #   - number of inverse muons in the event that are not in jets with 2 muons
            #   - number of jets with 2 tight muons
            #   - number of jets with 2 inverse muons
            #   - number of jets with 1 tight and 1 inverse muon
            "full_method_hist": hist.Hist.new.Regular(
                8, 0, 8, name="nMuon_tight_l2", label="nMuon_tight_l2"
            ).Regular(
                8, 0, 8, name="nMuon_inverse_l2", label="nMuon_inverse_l2"
            ).Regular(
                4, 0, 4, name="nJet_2T_muons", label="nJet_2T_muons"
            ).Regular(
                4, 0, 4, name="nJet_2I_muons", label="nJet_2I_muons"
            ).Regular(
                4, 0, 4, name="nJet_1I1T_muons", label="nJet_1I1T_muons"
            ).Regular(
                8, 0, 8, name="nMuon", label="nMuon"
            ).Weight(),
            "check_pt_dependency": hist.Hist.new.Regular(
                9, 3, 300, name="pt", label="pt", transform=hist.axis.transform.log
            ).Regular(
                8, 0, 8, name="nMuon_tight", label="nMuon_tight"
            ).Regular(
                8, 0, 8, name="nMuon_inverse", label="nMuon_inverse"
            ).Regular(
                8, 0, 8, name="nMuon", label="nMuon"
            ).Weight(),
 
            "nMuon_tight_per_jet_vs_nMuon_inverse_per_jet": hist.Hist.new.Regular(
                3, 0, 3, name="nMuon_tight_per_jet", label="nMuon_tight_per_jet"
            ).Regular(
                3, 0, 3, name="nMuon_inverse_per_jet", label="nMuon_inverse_per_jet"
            ).Weight(),
            "num_muons_in_associated_jet_vs_nMuon": hist.Hist.new.Regular(
                8, 0, 8, name="num_muons_in_associated_jet", label="num_muons_in_associated_jet"
            ).Regular(
                5, 3, 8, name="nMuon", label="nMuon"
            ).Weight(),
            "nMuons_per_jet_vs_nMuon": hist.Hist.new.Regular(
                8, 0, 8, name="nMuons_per_jet", label="nMuons_per_jet"
            ).Regular(
                5, 3, 8, name="nMuon", label="nMuon"
            ).Weight(),
            "nMuon_inv_in_vs_nMuon_tight_in_vs_nMuon_inv_out_vs_nMuon_tight_out_vs_nMuon": hist.Hist.new.Regular(
                8, 0, 8, name="nMuon_inv_in", label="nMuon_inv_in"
            ).Regular(
                8, 0, 8, name="nMuon_tight_in", label="nMuon_tight_in"
            ).Regular(
                8, 0, 8, name="nMuon_inv_out", label="nMuon_inv_out"
            ).Regular(
                8, 0, 8, name="nMuon_tight_out", label="nMuon_tight_out"
            ).Regular(
                5, 3, 8, name="nMuon", label="nMuon"
            ).Weight(),

            # Better definition of the above histogram
            "nMuon_inv_0_nMuon_tight_0_through_4_vs_nMuon": hist.Hist.new.Regular(
                6, 0, 6, name="nMuon_inv_out_of_jet", label="nMuon_inv_out_of_jet"
            ).Regular(
                6, 0, 6, name="nMuon_tight_out_of_jet", label="nMuon_tight_out_of_jet"
            ).Regular(
                6, 0, 6, name="nMuon_inv_in_1muon_jet", label="nMuon_inv_in_1muon_jet"
            ).Regular(
                6, 0, 6, name="nMuon_tight_in_1muon_jet", label="nMuon_tight_in_1muon_jet"
            ).IntCategory(
                [0, 2, 4], name="nMuon_inv_in_2muon_jet", label="nMuon_inv_in_2muon_jet"
            ).IntCategory(
                [0, 2, 4], name="nMuon_tight_in_2muon_jet", label="nMuon_tight_in_2muon_jet"
            # ).IntCategory(
            #     [0, 3], name="nMuon_inv_in_3muon_jet", label="nMuon_inv_in_3muon_jet"
            # ).IntCategory(
            #     [0, 3], name="nMuon_tight_in_3muon_jet", label="nMuon_tight_in_3muon_jet"
            # ).IntCategory(
            #     [0, 4], name="nMuon_inv_in_4muon_jet", label="nMuon_inv_in_4muon_jet"
            # ).IntCategory(
            #     [0, 4], name="nMuon_tight_in_4muon_jet", label="nMuon_tight_in_4muon_jet"
            ).Regular(
                3, 3, 6, name="nMuon", label="nMuon"
            ).Weight(),

            "nMuon_inv_vs_nMuon_tight_vs_nMuon": hist.Hist.new.Regular(
                8, 0, 8, name="nMuon_inv", label="nMuon_inv"
            ).Regular(
                8, 0, 8, name="nMuon_tight", label="nMuon_tight"
            ).Regular(
                5, 3, 8, name="nMuon", label="nMuon"
            ).Weight(),
            # "nMuon_vs_nJet_with_muon": hist.Hist.new.Regular(
            #     5, 3, 8, name="nMuon", label="nMuon"
            # ).Regular(
            #     10, 0, 10, name="nJet_withMuon", label="nJet_withMuon"
            # ).Weight(),
            # "nTight_in_jet_vs_nInverse_in_jet_vs_nMuon": hist.Hist.new.Regular(
            #     4, 0, 4, name="nTight_in_jet", label="nTight_in_jet"
            # ).Regular(
            #     4, 0, 4, name="nInverse_in_jet", label="nInverse_in_jet"
            # ).Regular(
            #     5, 3, 8, name="nMuon", label="nMuon"
            # ).Weight(),
            # "btag_vs_muon_ip3d": hist.Hist.new.Bool(name="btag").Regular(
            #     100, 1e-6, 1, name="muon_ip3d", label="muon_ip3d", transform=hist.axis.transform.log
            # ).Weight(),
            # "btag_score_cut_vs_nTights_vs_nMuon": hist.Hist.new.Regular(
            #     50, 0, 1, name="btag_score_cut", label="btag_score_cut"
            # ).Regular(
            #     8, 0, 8, name="nTight", label="nTight"
            # ).Regular(
            #     5, 3, 8, name="nMuon", label="nMuon"
            # ).Weight(),
            # "ip3d_cut_vs_nTights_vs_nMuon": hist.Hist.new.Regular(
            #     50, 1e-4, 0.1, name="ip3d_cut", label="ip3d_cut", transform=hist.axis.transform.log
            # ).Regular(
            #     8, 0, 8, name="nTight", label="nTight"
            # ).Regular(
            #     5, 3, 8, name="nMuon", label="nMuon"
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
