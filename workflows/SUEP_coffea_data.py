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
from workflows.CMS_corrections.jetmet_utils import apply_jecs
from workflows.CMS_corrections.PartonShower_utils import GetPSWeights
from workflows.CMS_corrections.Prefire_utils import GetPrefireWeights
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

    def jet_awkward(self, Jets):
        """
        Create awkward array of jets. Applies basic selections.
        Returns: awkward array of dimensions (events x jets x 4 momentum)
        """
        Jets_awk = ak.zip(
            {
                "pt": Jets.pt,
                "eta": Jets.eta,
                "phi": Jets.phi,
                "mass": Jets.mass,
            }
        )
        jet_awk_Cut = (Jets.pt > 30) & (abs(Jets.eta) < 2.4)
        Jets_correct = Jets_awk[jet_awk_Cut]
        return Jets_correct

    def eventSelection(self, events):
        """
        Applies trigger, returns events.
        Default is PFHT triggers. Can use selection variable for customization.
        """
        # NOTE: Might be a good idea to make this a 'match-case' statement
        # once we can move to Python 3.10 for good.
        if self.trigger == "TripleMu":
            if self.era == 2016:
                trigger = events.HLT.TripleMu_5_3_3 == 1
            elif self.era == 2017:
                trigger = events.HLT.TripleMu_5_3_3_Mass3p8to60_DZ == 1
            elif self.era == 2018:
                try:
                    trigger = events.HLT.TripleMu_5_3_3_Mass3p8_DZ == 1
                except:
                    print("Warning; This file seems to have the 2017 trigger names")
                    trigger = events.HLT.TripleMu_5_3_3_Mass3p8to60_DZ == 1
            else:
                raise ValueError("Invalid era")
            events = events[trigger]
        else:
            raise ValueError("Invalid trigger")
        return events

    def muon_filter(
        self, events, nMuons=None, pt_limit=None, avpt_cut=None, blind=False
    ):
        """
        Filter events after the TripleMu trigger.
        Cleans muons and electrons.
        Requires at least nMuons with mediumId, pt, dxy, dz, and eta cuts.
        """
        muons = events.Muon
        electrons = events.Electron
        clean_muons = (
            (events.Muon.mediumId)
            & (events.Muon.pt > 3)
            & (abs(events.Muon.dxy) <= 0.02)
            & (abs(events.Muon.dz) <= 0.1)
            & (abs(events.Muon.eta) < 2.4)
        )
        if pt_limit is not None:
            clean_muons = clean_muons & (events.Muon.pt < pt_limit)
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
        if blind:
            select_by_muons = ak.num(muons, axis=-1) == 3
            events = events[select_by_muons]
            muons = muons[select_by_muons]
            electrons = electrons[select_by_muons]
            return events, electrons, muons
        if nMuons is not None:
            select_by_muons = ak.num(muons, axis=-1) >= nMuons
            events = events[select_by_muons]
            muons = muons[select_by_muons]
            electrons = electrons[select_by_muons]
        if avpt_cut is not None:
            avpt = ak.mean(muons.pt, axis=-1)
            events = events[avpt < avpt_cut]
            muons = muons[avpt < avpt_cut]
            electrons = electrons[avpt < avpt_cut]
        return events, electrons, muons

    def getGenTracks(self, events):
        genParts = events.GenPart
        genParts = ak.zip(
            {
                "pt": genParts.pt,
                "eta": genParts.eta,
                "phi": genParts.phi,
                "mass": genParts.mass,
                "pdgID": genParts.pdgId,
            },
            with_name="Momentum4D",
        )
        return genParts

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
                "mass": ak.zeros_like(events.lostTracks.pt),
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

    def count_events(
        self,
        events,
        nMuons=None,
        pt_limit=None,
        interiso_cut=None,
        avpt_cut=None,
        blind=False,
    ):
        # first apply the muon filter
        events, electrons, muons = self.muon_filter(
            events, nMuons=nMuons, pt_limit=pt_limit, avpt_cut=avpt_cut, blind=blind
        )
        # then calculate the inter-isolation
        if interiso_cut is not None:
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
                    "charge": electrons.pdgId / (-11),
                },
                with_name="Momentum4D",
            )
            leptons = ak.concatenate([muonsCollection, electronsCollection], axis=-1)
            leading_muons = leptons[:, 0]
            I15 = SUEP_utils.inter_isolation(leading_muons, leptons, dR=1.6)
            events = events[I15 > interiso_cut]
        return events

    def fill_histograms(self, events, muons, tracks, output):
        dataset = events.metadata["dataset"]

        # These arrays need to be broadcasted to the per muon dims from per event dims
        nMuons = ak.flatten(ak.broadcast_arrays(ak.num(muons), muons.pt)[0])
        weights = np.ones(len(events))
        if self.isMC:
            weights = events.genWeight
        weights_per_muon = ak.flatten(ak.broadcast_arrays(weights, muons.pt)[0])

        output[dataset]["histograms"]["muon_pt_vs_nMuon"].fill(
            ak.flatten(muons.pt), nMuons, weight=weights_per_muon
        )

        mT_MET = SUEP_utils.transverse_mass(muons, events.MET)
        mT = np.sqrt(muons.E**2 - muons.pz**2)
        output[dataset]["histograms"]["muon_METmt_vs_nMuon"].fill(
            ak.flatten(mT_MET), nMuons, weight=weights_per_muon
        )
        output[dataset]["histograms"]["muon_METmt_vs_nMuon"].fill(
            ak.flatten(mT_MET), nMuons, weight=weights_per_muon
        )
        output[dataset]["histograms"]["muon_mt_vs_nMuon"].fill(
            ak.flatten(mT), nMuons, weight=weights_per_muon
        )
        output[dataset]["histograms"]["muon_mt_vs_nMuon"].fill(
            ak.flatten(mT), nMuons, weight=weights_per_muon
        )
        output[dataset]["histograms"]["muon_pt_vs_nMuon"].fill(
            ak.flatten(muons.pt), nMuons, weight=weights_per_muon
        )
        output[dataset]["histograms"]["muon_dxy_vs_miniPFRelIso_vs_nMuon"].fill(
            ak.flatten(abs(muons.dxy)),
            ak.flatten(muons.miniPFRelIso_all),
            nMuons,
            weight=weights_per_muon,
        )
        output[dataset]["histograms"]["muon_pt_vs_miniPFRelIso_vs_nMuon"].fill(
            ak.flatten(muons.pt),
            ak.flatten(muons.miniPFRelIso_all),
            nMuons,
            weight=weights_per_muon,
        )
        output[dataset]["histograms"]["muon_dz_vs_nMuon"].fill(
            ak.flatten(abs(muons.dz)),
            nMuons,
            weight=weights_per_muon,
        )
        output[dataset]["histograms"]["muon_ip3d_vs_nMuon"].fill(
            ak.flatten(muons.ip3d),
            nMuons,
            weight=weights_per_muon,
        )
        output[dataset]["histograms"]["nTrack_vs_nMuon"].fill(
            ak.num(tracks),
            ak.num(muons),
            weight=weights,
        )
        output[dataset]["histograms"]["nTrack_log_vs_nMuon"].fill(
            ak.num(tracks),
            ak.num(muons),
            weight=weights,
        )

        return

    def analysis(self, events, output, do_syst=False, col_label=""):
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
        weights = np.ones(len(events))
        if self.isMC:
            weights = events.genWeight

        # Fill the cutflow columns for trigger
        output[dataset]["cutflow"].fill(
            len(events) * ["trigger"],
            weight=weights,
        )

        # Calculate various cutflow variables
        events_n3 = self.count_events(events, blind=True)
        weights_n3 = np.ones(len(events_n3))
        if self.isMC:
            weights_n3 = events_n3.genWeight
        output[dataset]["cutflow"].fill(
            len(events_n3) * ["nMu==3"],
            weight=weights_n3,
        )

        # fill the histograms
        events, electrons, muons = self.muon_filter(events, blind=True)
        tracks, Cleaned_cands = self.getTracks(events)
        self.fill_histograms(events, muons, tracks, output)

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
            # Muon pt histogram - per muon entry
            # Add an axis for nMuon
            "muon_pt_vs_nMuon": hist.Hist.new.Reg(
                150, 0, 150, name="muon_pt", label="Muon p_{T} [GeV]"
            )
            .Reg(10, 0, 10, name="nMuon", label="nMuon")
            .Weight(),
            # Transverse mass histogram between muons and MET
            "muon_METmt_vs_nMuon": hist.Hist.new.Reg(
                150, 0, 150, name="mt", label="m_{T} [GeV]"
            )
            .Reg(10, 0, 10, name="nMuon", label="nMuon")
            .Weight(),
            "muon_mt_vs_nMuon": hist.Hist.new.Reg(
                150, 0, 150, name="mt", label="m_{T} [GeV]"
            )
            .Reg(10, 0, 10, name="nMuon", label="nMuon")
            .Weight(),
            "muon_pt_vs_nMuon": hist.Hist.new.Reg(
                150, 0, 150, name="pt", label="p_{T} [GeV]"
            )
            .Reg(10, 0, 10, name="nMuon", label="nMuon")
            .Weight(),
            "muon_METmt_vs_nMuon": hist.Hist.new.Reg(
                150, 0, 150, name="mt", label="m_{T} [GeV]"
            )
            .Reg(10, 0, 10, name="nMuon", label="nMuon")
            .Weight(),
            "muon_mt_vs_nMuon": hist.Hist.new.Reg(
                150, 0, 150, name="mt", label="m_{T} [GeV]"
            )
            .Reg(10, 0, 10, name="nMuon", label="nMuon")
            .Weight(),
            "muon_dz_vs_nMuon": hist.Hist.new.Regular(
                200, 0, 0.1, name="Muon_dz", label="Muon_dz"
            )
            .Reg(10, 0, 10, name="nMuon", label="nMuon")
            .Weight(),
            "muon_ip3d_vs_nMuon": hist.Hist.new.Regular(
                200, 0, 0.1, name="Muon_ip3d", label="Muon_ip3d"
            )
            .Reg(10, 0, 10, name="nMuon", label="nMuon")
            .Weight(),
            "muon_dxy_vs_miniPFRelIso_vs_nMuon": hist.Hist.new.Regular(
                50,
                1e-4,
                1e-1,
                name="Muon_dxy",
                label="Muon_dxy",
                transform=hist.axis.transform.log,
            )
            .Regular(
                50,
                1e-3,
                1e3,
                name="Muon_miniPFRelIso_all",
                label="Muon_miniPFRelIso_all",
                transform=hist.axis.transform.log,
            )
            .Reg(10, 0, 10, name="nMuon", label="nMuon")
            .Weight(),
            "muon_pt_vs_miniPFRelIso_vs_nMuon": hist.Hist.new.Regular(
                50,
                1,
                1e3,
                name="pt",
                label="p_{T} [GeV]",
                transform=hist.axis.transform.log,
            )
            .Regular(
                50,
                1e-3,
                1e3,
                name="Muon_miniPFRelIso_all",
                label="Muon_miniPFRelIso_all",
                transform=hist.axis.transform.log,
            )
            .Reg(10, 0, 10, name="nMuon", label="nMuon")
            .Weight(),
            "nTrack_vs_nMuon": hist.Hist.new.Reg(
                100,
                0,
                200,
                name="nTrack",
                label="nTrack",
            )
            .Reg(10, 0, 10, name="nMuon", label="nMuon")
            .Weight(),
            "nTrack_log_vs_nMuon": hist.Hist.new.Reg(
                50,
                1,
                1e3,
                name="nTrack",
                label="nTrack",
                transform=hist.axis.transform.log,
            )
            .Reg(10, 0, 10, name="nMuon", label="nMuon")
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

        # run the analysis with the track systematics applied
        if self.isMC and self.do_syst:
            self.analysis(events, output, do_syst=True, col_label="_track_down")

        # run the analysis
        self.analysis(events, output)

        return output

    def postprocess(self, accumulator):
        pass
