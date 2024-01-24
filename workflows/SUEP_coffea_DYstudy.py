"""
SUEP_coffea.py
Coffea producer for SUEP analysis. Uses fastjet package to recluster large jets:
https://github.com/scikit-hep/fastjet
Chad Freer and Luca Lavezzo, 2021
"""
import gc
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
        try:
            if self.trigger == "TripleMu":
                if "TripleMu_5_3_3_Mass3p8_DZ" in events.HLT.fields:
                    trigger = events.HLT.TripleMu_5_3_3_Mass3p8_DZ == 1
                else:
                    if self.isMC:
                        raise ValueError(
                            "This 2018 file seems to have the 2017 trigger names"
                        )
                    trigger = ak.zeros_like(events.HLT.ZeroBias)
            elif self.trigger == "DoubleMu":
                # Alternative trigger path
                trigger = events.HLT.Mu37_TkMu27 == 1
        except:
            raise ValueError("The trigger path was not found in this file")
        events = events[trigger]
        return events

    def selected_muons(self, events):
        muons = events.Muon
        cut_muons = (
            (events.Muon.looseId)
            & (events.Muon.pt >= 10)
            & (abs(events.Muon.dxy) <= 0.02)
            & (abs(events.Muon.dz) <= 0.1)
            & (events.Muon.pfIsoId >= 2)
            & (abs(events.Muon.eta) < 2.4)
        )
        return muons[cut_muons]

    def selected_electrons(self, events):
        electrons = events.Electron
        cut_electrons = (
            (events.Electron.pt >= 10)
            & (events.Electron.mvaFall17V2Iso_WP90)
            & (
                abs(events.Electron.dxy)
                < (0.05 + 0.05 * (abs(events.Electron.eta) > 1.479))
            )
            & (abs(events.Electron.dz) < (0.10 + 0.10 * (events.Electron.eta > 1.479)))
            & ((abs(events.Electron.eta) < 1.444) | (abs(events.Electron.eta) > 1.566))
            & (abs(events.Electron.eta) < 2.5)
        )
        return electrons[cut_electrons]

    def find_OSSF_pairs(self, leptons):
        # Find all possible pairs
        all_pairs = ak.combinations(leptons, n=2, axis=1)
        l1, l2 = ak.unzip(all_pairs)
        # Find the pairs that have the same flavor
        cut = (l1.pdgId + l2.pdgId) == 0
        # Get the pairs
        return l1[cut], l2[cut]

    def ht(self, events):
        jet_Cut = (events.Jet.pt > 20) & (abs(events.Jet.eta) < 2.4)
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
        # Get the muons and apply basic cleaning
        # NOTE: check if we should use looser requirements for the initial nMuon counting
        muons = events.Muon
        clean_muons = (
            (events.Muon.mediumId)
            & (events.Muon.pt > 3)
            & (abs(events.Muon.eta) < 2.4)
            & (abs(events.Muon.dxy) <= 0.02)
            & (abs(events.Muon.dz) <= 0.1)
        )
        muons = muons[clean_muons]
        select_by_muons = (ak.num(muons, axis=-1) > 1) & (ak.num(muons, axis=-1) < 5)
        events = events[select_by_muons]
        muons = muons[select_by_muons]

        # Common selection to make the trigger paths match
        # select_by_muons = (muons.pt[:, 0] > 37) & (muons.pt[:, 1] > 27)
        # events = events[select_by_muons]
        # muons = muons[select_by_muons]

        # Need to reapply the mass cut
        # if len(muons) > 0:
        # idx = ak.argsort(muons.pt, ascending=False)
        # muon_pairs = ak.combinations(muons, n=2, axis=-1)
        # muon_pairs_args = ak.argcombinations(muons, n=2, axis=1)
        # mu1, mu2 = ak.unzip(muon_pairs)
        # arg1, arg2 = ak.unzip(muon_pairs_args)
        # mass_cut = (mu1 + mu2).mass > 3.8
        # not_passing_muons = ak.concatenate(
        #     [arg1[~mass_cut], arg2[~mass_cut]], axis=1
        # )
        # builder = ak.ArrayBuilder()
        # SUEP_utils.loop_over_arr(idx, not_passing_muons, builder)
        # muon_mask = ak.copy(builder.snapshot())
        # muons = muons[muon_mask]
        # del builder, muon_mask
        # gc.collect()
        # select_by_muons = ak.num(muons, axis=-1) > 2
        # events = events[select_by_muons]
        # muons = muons[select_by_muons]

        # # Finally, blind the data
        # select_by_muons = ak.num(muons, axis=-1) < 5
        # events = events[select_by_muons]
        # muons = muons[select_by_muons]

        # Cuts for this CR - apply only to hardest muon for now
        # leading_muons = ak.firsts(muons)
        # final_selection = (
        #    (ak.fill_none(leading_muons.pt, 0) > 30)
        #    & (ak.fill_none(leading_muons.miniPFRelIso_all, 0) < 0.1)
        #    & (abs(ak.fill_none(leading_muons.dxy, 0)) < 0.003)
        # )
        final_selection = (
            (muons.pt > 25)
            & (muons.miniPFRelIso_all < 0.1)
            & (abs(muons.dxy) < 0.003)
            & (abs(muons.dz) < 0.01)
        )
        muons = muons[final_selection]

        # Make sure there is at least one muon in the event after the cuts
        # select_by_muons_high = ak.num(muons, axis=-1) > 0
        select_by_muons_high = ak.any(final_selection, axis=-1)
        events = events[select_by_muons_high]
        muons = muons[select_by_muons_high]

        return events, muons

    def fill_histograms(self, events, muons, output):
        dataset = events.metadata["dataset"]

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

        # Find leading OSSF lepton pair
        mu1, mu2 = self.find_OSSF_pairs(self.selected_muons(events))
        dimuon_pt = ak.fill_none(ak.pad_none(mu1.pt + mu2.pt, 1, axis=1), 0)
        argmax_dimuon_pt = ak.argmax(dimuon_pt, axis=-1)

        HT = self.ht(events)
        # remove lepton pt from HT
        HT_sub = (
            HT
            - ak.firsts(
                ak.fill_none(ak.pad_none(mu1.matched_jet.pt, 1, axis=1), 0)[
                    ak.singletons(argmax_dimuon_pt)
                ]
            )
            - ak.firsts(
                ak.fill_none(ak.pad_none(mu1.matched_jet.pt, 1, axis=1), 0)[
                    ak.singletons(argmax_dimuon_pt)
                ]
            )
        )

        # Fill the histograms
        # Per event histograms
        output[dataset]["histograms"]["ht"].fill(
            HT,
            HT_sub,
            weight=weights,
        )
        nGenJets = np.zeros(len(events))
        if self.isMC:
            nGenJets = ak.num(events.GenJet)
        output[dataset]["histograms"]["nJet_vs_nGenJet_vs_nMuon"].fill(
            ak.num(self.clean_jets(events.Jet)),
            nGenJets,
            ak.num(muons),
            weight=weights,
        )
        LHE_Njets = np.zeros(len(events))
        LHE_Nglu = np.zeros(len(events))
        LHE_Nuds = np.zeros(len(events))
        LHE_Nc = np.zeros(len(events))
        LHE_Nb = np.zeros(len(events))
        if self.isMC and "LHE" in events.fields:
            LHE_Njets = events.LHE.Njets
            LHE_Nglu = events.LHE.Nglu
            LHE_Nuds = events.LHE.Nuds
            LHE_Nc = events.LHE.Nc
            LHE_Nb = events.LHE.Nb
        output[dataset]["histograms"][
            "LHE_Njets_vs_Nglu_vs_Nuds_vs_Nc_vs_Nb_vs_nMuon"
        ].fill(
            LHE_Njets,
            LHE_Nglu,
            LHE_Nuds,
            LHE_Nc,
            LHE_Nb,
            ak.num(muons),
            weight=weights,
        )
        muons_genPartFlav_v = ak.zeros_like(muons.pt, dtype=int)
        if self.isMC:
            muons_genPartFlav_v = muons.genPartFlav
        nMuon_nonprompt_reco = ak.num(muons) - ak.sum(
            (muons.pt > 25)
            & (muons.miniPFRelIso_all < 0.1)
            & (abs(muons.dxy) < 0.003)
            & (abs(muons.dz) < 0.01),
            axis=-1,
        )
        output[dataset]["histograms"][
            "LHE_Njets_vs_nMuon_vs_nMuon_nonprompt_vs_nMuon_nonprompt_reco"
        ].fill(
            LHE_Njets,
            ak.num(muons),
            ak.sum(muons_genPartFlav_v != 1, axis=-1),
            nMuon_nonprompt_reco,
            weight=weights,
        )
        output[dataset]["histograms"][
            "nMuon_unmatched_vs_nMuon_prompt_vs_nMuon_light_vs_nMuon_c_vs_nMuon_b_vs_nMuon_tau"
        ].fill(
            ak.sum(muons_genPartFlav_v == 0, axis=-1),
            ak.sum(muons_genPartFlav_v == 1, axis=-1),
            ak.sum(muons_genPartFlav_v == 3, axis=-1),
            ak.sum(muons_genPartFlav_v == 4, axis=-1),
            ak.sum(muons_genPartFlav_v == 5, axis=-1),
            ak.sum(muons_genPartFlav_v == 15, axis=-1),
            weight=weights,
        )
        # Per muon histograms
        output[dataset]["histograms"]["muon_pt_linear_vs_genPartFlav_vs_nMuon"].fill(
            ak.flatten(muons.pt),
            muons_genPartFlav,
            nMuons,
            weight=weights_per_muon,
        )
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
            ak.flatten(ak.where(muons.jetPtRelv2 > 1e-2, muons.jetPtRelv2, 1e-2)),
            muons_genPartFlav,
            nMuons,
            weight=weights_per_muon,
        )
        output[dataset]["histograms"][
            "muon_matched_jetRelIso_vs_genPartFlav_vs_nMuon"
        ].fill(
            ak.flatten(ak.where(muons.jetRelIso > 1e-5, muons.jetRelIso, 1e-5)),
            muons_genPartFlav,
            nMuons,
            weight=weights_per_muon,
        )

        # Muon pair opening angles
        muons_vectors = ak.zip(
            {
                "px": muons.px,
                "py": muons.py,
                "pz": muons.pz,
            },
            with_name="Momentum3D",
        )
        pairs = ak.combinations(
            muons_vectors, n=2, axis=-1, highlevel=True, with_name="Momentum3D"
        )
        mu1, mu2 = ak.unzip(pairs)
        weights_per_pair = ak.flatten(ak.broadcast_arrays(weights, mu1.px)[0])
        nMuon_per_pair = ak.flatten(ak.broadcast_arrays(ak.num(muons), mu1.px)[0])
        output[dataset]["histograms"]["all_pairs_deltaangle_vs_nMuon"].fill(
            ak.flatten(mu1.deltaangle(mu2)),
            nMuon_per_pair,
            weight=weights_per_pair,
        )

        # Do stuff with all muon pairs
        all_pairs = ak.combinations(muons, n=2, axis=-1)
        mu1, mu2 = ak.unzip(all_pairs)
        weights_per_pair = ak.flatten(ak.broadcast_arrays(weights, mu1.pt)[0])
        output[dataset]["histograms"]["all_pairs_dR"].fill(
            ak.flatten(mu1.delta_r(mu2)),
            weight=weights_per_pair,
        )

        # opposite_charges = mu1.charge != mu2.charge
        # mu1_os = mu1[opposite_charges]
        # mu2_os = mu2[opposite_charges]
        # os_pairs = mu1 + mu2

        # Leading dimuon pair histograms
        # Steps:
        #  - Get the leading muon
        #  - Filter the the other muons to keep only opposite sign pairs
        #  - Get the leading OS dimuon pair
        #  - Fill the histograms
        weights_less_that_two_muons = weights[ak.num(muons) < 2]
        weights = weights[ak.num(muons) > 1]
        muons = muons[ak.num(muons) > 1]
        leading_muon = ak.zip(
            {
                "pt": muons[:, 0].pt,
                "eta": muons[:, 0].eta,
                "phi": muons[:, 0].phi,
                "mass": muons[:, 0].mass,
                "charge": muons[:, 0].charge,
            },
            with_name="Momentum4D",
        )
        leading_dimuon_deltaR = muons[:, 0].delta_r(muons[:, 1])
        output[dataset]["histograms"]["leading_dimuon_pair_deltaR"].fill(
            leading_dimuon_deltaR,
            weight=weights,
        )
        # Fill -0.5 for events with less than 2 muons
        output[dataset]["histograms"]["leading_dimuon_pair_deltaR"].fill(
            np.full_like(weights_less_that_two_muons, -0.5),
            weight=weights_less_that_two_muons,
        )
        output[dataset]["histograms"]["leading_muon_iso_vs_subleading_muon_iso"].fill(
            muons[:, 0].miniPFRelIso_all,
            muons[:, 1].miniPFRelIso_all,
            weight=weights,
        )

        other_muons = muons[:, 1:]
        leading_charge_broadcasted = ak.broadcast_arrays(
            leading_muon.charge, other_muons.pt
        )[0]
        os_other_muons = other_muons[leading_charge_broadcasted != other_muons.charge]
        leading_muon = leading_muon[ak.num(os_other_muons) > 0]
        weights_no_os_muons = weights[ak.num(os_other_muons) == 0]

        weights = weights[ak.num(os_other_muons) > 0]
        os_other_muons = os_other_muons[ak.num(os_other_muons) > 0]
        os_subleading_muon = ak.zip(
            {
                "pt": os_other_muons[:, 0].pt,
                "eta": os_other_muons[:, 0].eta,
                "phi": os_other_muons[:, 0].phi,
                "mass": os_other_muons[:, 0].mass,
                "charge": os_other_muons[:, 0].charge,
            },
            with_name="Momentum4D",
        )
        leading_os_dimuon_pair = leading_muon + os_subleading_muon
        output[dataset]["histograms"]["leading_os_dimuon_pair_pt_vs_mass"].fill(
            leading_os_dimuon_pair.pt,
            leading_os_dimuon_pair.mass,
            weight=weights,
        )
        output[dataset]["histograms"]["leading_os_dimuon_pair_log_pt_vs_mass"].fill(
            leading_os_dimuon_pair.pt,
            leading_os_dimuon_pair.mass,
            weight=weights,
        )
        # Fill 0 for events with less than 2 OS muons
        not_passing_weights = ak.concatenate(
            [
                weights_less_that_two_muons,
                weights_no_os_muons,
            ]
        )
        output[dataset]["histograms"]["leading_os_dimuon_pair_pt_vs_mass"].fill(
            np.zeros_like(not_passing_weights),
            np.zeros_like(not_passing_weights),
            weight=not_passing_weights,
        )
        output[dataset]["histograms"]["leading_os_dimuon_pair_log_pt_vs_mass"].fill(
            np.ones_like(not_passing_weights),
            np.ones_like(not_passing_weights),
            weight=not_passing_weights,
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

        # Keep only 0 jet case for DY
        if "DY0JetsToLL_M-50_TuneCP5" in dataset:
            events = events[events.LHE.Njets == 0]

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
                "nMu==3",
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
            )
            .Reg(
                100,
                1,
                1e4,
                name="ht_subtracted",
                label="ht_subtracted",
                transform=hist.axis.transform.log,
            )
            .Weight(),
            "nJet_vs_nGenJet_vs_nMuon": hist.Hist.new.Reg(
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
            .Reg(
                6,
                0,
                6,
                name="nMuon",
                label="nMuon",
            )
            .Weight(),
            "LHE_Njets_vs_Nglu_vs_Nuds_vs_Nc_vs_Nb_vs_nMuon": hist.Hist.new.Reg(
                6,
                0,
                6,
                name="LHE_Njets",
                label="LHE_Njets",
            )
            .Reg(
                6,
                0,
                6,
                name="LHE_Nglu",
                label="LHE_Nglu",
            )
            .Reg(
                6,
                0,
                6,
                name="LHE_Nuds",
                label="LHE_Nuds",
            )
            .Reg(
                6,
                0,
                6,
                name="LHE_Nc",
                label="LHE_Nc",
            )
            .Reg(
                6,
                0,
                6,
                name="LHE_Nb",
                label="LHE_Nb",
            )
            .Reg(
                6,
                0,
                6,
                name="nMuon",
                label="nMuon",
            )
            .Weight(),
            "LHE_Njets_vs_nMuon_vs_nMuon_nonprompt_vs_nMuon_nonprompt_reco": hist.Hist.new.Reg(
                5,
                0,
                5,
                name="LHE_Njets",
                label="LHE_Njets",
            )
            .Reg(
                5,
                0,
                5,
                name="nMuon",
                label="nMuon",
            )
            .Reg(
                5,
                0,
                5,
                name="nMuon_nonprompt",
                label="nMuon_nonprompt",
            )
            .Reg(
                5,
                0,
                5,
                name="nMuon_nonprompt_reco",
                label="nMuon_nonprompt_reco",
            )
            .Weight(),
            "nMuon_unmatched_vs_nMuon_prompt_vs_nMuon_light_vs_nMuon_c_vs_nMuon_b_vs_nMuon_tau": hist.Hist.new.Reg(
                5,
                0,
                5,
                name="nMuon_unmatched",
                label="nMuon_unmatched",
            )
            .Reg(
                5,
                0,
                5,
                name="nMuon_prompt",
                label="nMuon_prompt",
            )
            .Reg(
                5,
                0,
                5,
                name="nMuon_light",
                label="nMuon_light",
            )
            .Reg(
                5,
                0,
                5,
                name="nMuon_c",
                label="nMuon_c",
            )
            .Reg(
                5,
                0,
                5,
                name="nMuon_b",
                label="nMuon_b",
            )
            .Reg(
                5,
                0,
                5,
                name="nMuon_tau",
                label="nMuon_tau",
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
            "muon_pt_linear_vs_genPartFlav_vs_nMuon": hist.Hist.new.Regular(
                50,
                0,
                200,
                name="Muon_pt",
                label="Muon_pt",
            )
            .IntCategory(
                [0, 1, 3, 4, 5, 15], name="Muon_genPartFlav", label="Muon_genPartFlav"
            )
            .Reg(8, 0, 8, name="nMuon", label="nMuon")
            .Weight(),
            "muon_pt_vs_genPartFlav_vs_nMuon": hist.Hist.new.Regular(
                50,
                1,
                1000,
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
                -3,
                3,
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
                1e-6,
                1e-2,
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
                1e-1,
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
                1,
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
                1e-2,
                1e3,
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
                1e-5,
                1e2,
                name="matched_jetRelIso",
                label="matched_jetRelIso",
                transform=hist.axis.transform.log,
            )
            .IntCategory(
                [0, 1, 3, 4, 5, 15], name="Muon_genPartFlav", label="Muon_genPartFlav"
            )
            .Reg(8, 0, 8, name="nMuon", label="nMuon")
            .Weight(),
            "all_pairs_deltaangle_vs_nMuon": hist.Hist.new.Reg(
                50,
                0,
                3.2,
                name="all_pairs_deltaangle",
                label="all_pairs_deltaangle",
            )
            .Reg(8, 0, 8, name="nMuon", label="nMuon")
            .Weight(),
            "all_pairs_dR": hist.Hist.new.Reg(
                100,
                0,
                6,
                name="all_pairs_dR",
                label="all_pairs_dR",
            ).Weight(),
            "leading_dimuon_pair_deltaR": hist.Hist.new.Reg(
                100,
                -1,
                6,
                name="leading_dimuon_pair_deltaR",
                label="leading_dimuon_pair_deltaR",
            ).Weight(),
            "leading_muon_iso_vs_subleading_muon_iso": hist.Hist.new.Regular(
                100,
                1e-4,
                1e3,
                name="Muon_miniPFRelIso_all leading",
                label="Muon_miniPFRelIso_all leading",
                transform=hist.axis.transform.log,
            )
            .Regular(
                100,
                1e-4,
                1e3,
                name="Muon_miniPFRelIso_all subleading",
                label="Muon_miniPFRelIso_all subleading",
                transform=hist.axis.transform.log,
            )
            .Weight(),
            "leading_os_dimuon_pair_pt_vs_mass": hist.Hist.new.Reg(
                150,
                0,
                300,
                name="leading_os_dimuon_pair_pt",
                label="leading_os_dimuon_pair_pt",
            )
            .Reg(
                150,
                0,
                300,
                name="leading_os_dimuon_pair_mass",
                label="leading_os_dimuon_pair_mass",
            )
            .Weight(),
            "leading_os_dimuon_pair_log_pt_vs_mass": hist.Hist.new.Reg(
                100,
                1,
                1e3,
                name="leading_os_dimuon_pair_pt",
                label="leading_os_dimuon_pair_pt",
                transform=hist.axis.transform.log,
            )
            .Reg(
                100,
                1,
                1e3,
                name="leading_os_dimuon_pair_mass",
                label="leading_os_dimuon_pair_mass",
                transform=hist.axis.transform.log,
            )
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
