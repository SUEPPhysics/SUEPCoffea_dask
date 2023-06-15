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
import workflows.ZH_utils as ZH_utils

# Importing CMS corrections
from workflows.CMS_corrections.golden_jsons_utils import applyGoldenJSON
from workflows.CMS_corrections.jetmet_utils import apply_jecs
from workflows.CMS_corrections.PartonShower_utils import GetPSWeights
from workflows.CMS_corrections.Prefire_utils import GetPrefireWeights
from workflows.CMS_corrections.track_killing_utils import track_killing
from workflows.pandas_accumulator import pandas_accumulator

# Set vector behavior
vector.register_awkward()


class SUEP_cluster(processor.ProcessorABC):
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
        accum: Optional[bool] = None,
        trigger: Optional[str] = None,
        debug: Optional[bool] = None,
    ) -> None:
        self._flag = flag
        self.output_location = output_location
        self.do_syst = do_syst
        self.gensumweight = 1.0
        self.scouting = scouting
        self.era = int(era)
        self.isMC = bool(isMC)
        self.sample = sample
        self.syst_var, self.syst_suffix = (
            (syst_var, f"_sys_{syst_var}") if do_syst and syst_var else ("", "")
        )
        self.weight_syst = weight_syst
        self.do_inf = do_inf
        self.prefixes = {"SUEP": "SUEP"}
        self.doOF = False
        self.accum = accum
        self.trigger = trigger
        self.debug = debug

        if self.do_inf:
            # ML settings
            self.batch_size = 1024

            # GNN settings
            # model names and configs should be in data/GNN/
            self.dgnn_model_names = [
                "single_l5_bPfcand_S1_SUEPtracks"
            ]  # Name for output
            self.configs = ["config.yml"]  # config paths
            self.obj = "bPFcand"
            self.coords = "cyl"

            # SSD settings
            self.ssd_models = []  # Add to this list. There will be an output for each
            self.eta_pix = 280
            self.phi_pix = 360
            self.eta_span = (-2.5, 2.5)
            self.phi_span = (-np.pi, np.pi)
            self.eta_scale = self.eta_pix / (self.eta_span[1] - self.eta_span[0])
            self.phi_scale = self.phi_pix / (self.phi_span[1] - self.phi_span[0])

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
        if self.scouting != 1:
            if self.era == 2016:
                trigger = events.HLT.PFHT900 == 1
            else:
                trigger = events.HLT.PFHT1050 == 1
            if self.trigger == "TripleMu":
                if self.era == 2016:
                    trigger = events.HLT.TripleMu_5_3_3 == 1
                elif self.era == 2017:
                    trigger = events.HLT.TripleMu_5_3_3_Mass3p8to60_DZ == 1
                elif self.era == 2018:
                    trigger = events.HLT.TripleMu_5_3_3_Mass3p8_DZ == 1
                else:
                    raise ValueError("Invalid era")
            events = events[trigger]
        return events

    def muon_filter(self, events, nMuons=4):
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
        select_by_muons = ak.num(muons, axis=-1) >= nMuons
        events = events[select_by_muons]
        muons = muons[select_by_muons]
        electrons = electrons[select_by_muons]
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

    def getScoutingTracks(self, events):
        Cands = ak.zip(
            {
                "pt": events.PFcand.pt,
                "eta": events.PFcand.eta,
                "phi": events.PFcand.phi,
                "mass": events.PFcand.mass,
            },
            with_name="Momentum4D",
        )
        cut = (
            (events.PFcand.pt >= 0.75)
            & (abs(events.PFcand.eta) <= 2.5)
            & (events.PFcand.vertex == 0)
            & (events.PFcand.q != 0)
        )
        Cleaned_cands = Cands[cut]
        tracks = ak.packed(Cleaned_cands)
        return tracks, Cleaned_cands

    def storeEventVars(
        self,
        events,
        output,
        tracks,
        electrons,
        muons,
        ak_inclusive_jets,
        ak_inclusive_cluster,
        out_label="",
    ):
        dataset = events.metadata["dataset"]

        # muon inter-isolation
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

        # select out ak4jets
        ak4jets = self.jet_awkward(events.Jet)

        # work on JECs and systematics
        prefix = ""
        if self.accum:
            if "casa" in self.accum:
                prefix = "dask-worker-space/"
        jets_c = apply_jecs(
            isMC=self.isMC,
            Sample=self.sample,
            era=self.era,
            events=events,
            prefix=prefix,
        )
        jets_jec = self.jet_awkward(jets_c)
        if self.isMC:
            jets_jec_JERUp = self.jet_awkward(jets_c["JER"].up)
            jets_jec_JERDown = self.jet_awkward(jets_c["JER"].down)
            jets_jec_JESUp = self.jet_awkward(jets_c["JES_jes"].up)
            jets_jec_JESDown = self.jet_awkward(jets_c["JES_jes"].down)
        # For data set these all to nominal so we can plot without switching all of the names
        else:
            jets_jec_JERUp = jets_jec
            jets_jec_JERDown = jets_jec
            jets_jec_JESUp = jets_jec
            jets_jec_JESDown = jets_jec

        # save per event variables to a dataframe
        output[dataset]["vars"]["ntracks" + out_label] = ak.num(tracks).to_list()
        output[dataset]["vars"]["ngood_fastjets" + out_label] = ak.num(
            ak_inclusive_jets
        ).to_list()
        if out_label == "":
            output[dataset]["vars"]["events" + out_label] = events.event.to_list()
            output[dataset]["vars"][
                "luminosityBlock" + out_label
            ] = events.luminosityBlock.to_list()
            output[dataset]["vars"][
                "genWeight" + out_label
            ] = events.genWeight.to_list()
            output[dataset]["vars"]["ht" + out_label] = ak.sum(
                ak4jets.pt, axis=-1
            ).to_list()
            output[dataset]["vars"]["ht_JEC" + out_label] = ak.sum(
                jets_jec.pt, axis=-1
            ).to_list()
            output[dataset]["vars"]["ht_JEC" + out_label + "_JER_up"] = ak.sum(
                jets_jec_JERUp.pt, axis=-1
            ).to_list()
            output[dataset]["vars"]["ht_JEC" + out_label + "_JER_down"] = ak.sum(
                jets_jec_JERDown.pt, axis=-1
            ).to_list()
            output[dataset]["vars"]["ht_JEC" + out_label + "_JES_up"] = ak.sum(
                jets_jec_JESUp.pt, axis=-1
            ).to_list()
            output[dataset]["vars"]["ht_JEC" + out_label + "_JES_down"] = ak.sum(
                jets_jec_JESDown.pt, axis=-1
            ).to_list()

            if self.era == 2016 and self.scouting == 0:
                output[dataset]["vars"]["HLT_PFHT900" + out_label] = events.HLT.PFHT900
            elif self.scouting == 0:
                output[dataset]["vars"][
                    "HLT_PFHT1050" + out_label
                ] = events.HLT.PFHT1050
            output[dataset]["vars"]["ngood_ak4jets" + out_label] = ak.num(
                ak4jets
            ).to_list()
            if self.scouting == 1:
                output[dataset]["vars"]["PV_npvs" + out_label] = ak.num(events.Vertex.x)
            else:
                if self.isMC:
                    output[dataset]["vars"][
                        "Pileup_nTrueInt" + out_label
                    ] = events.Pileup.nTrueInt
                    GetPSWeights(events, output[dataset])  # Parton Shower weights
                    GetPrefireWeights(self, events, output[dataset])  # Prefire weights
                output[dataset]["vars"]["PV_npvs" + out_label] = events.PV.npvs
                output[dataset]["vars"]["PV_npvsGood" + out_label] = events.PV.npvsGood

        # get gen SUEP kinematics
        SUEP_genMass = len(events) * [0]
        SUEP_genPt = len(events) * [0]
        SUEP_genEta = len(events) * [0]
        SUEP_genPhi = len(events) * [0]

        if self.isMC and not self.scouting:
            genParts = self.getGenTracks(events)
            genSUEP = genParts[(abs(genParts.pdgID) == 25)]

            # we need to grab the last SUEP in the chain for each event
            SUEP_genMass = [g[-1].mass if len(g) > 0 else 0 for g in genSUEP]
            SUEP_genPt = [g[-1].pt if len(g) > 0 else 0 for g in genSUEP]
            SUEP_genPhi = [g[-1].phi if len(g) > 0 else 0 for g in genSUEP]
            SUEP_genEta = [g[-1].eta if len(g) > 0 else 0 for g in genSUEP]

        output[dataset]["vars"]["SUEP_genMass" + out_label] = SUEP_genMass
        output[dataset]["vars"]["SUEP_genPt" + out_label] = SUEP_genPt
        output[dataset]["vars"]["SUEP_genEta" + out_label] = SUEP_genEta
        output[dataset]["vars"]["SUEP_genPhi" + out_label] = SUEP_genPhi

        # nMuons
        output[dataset]["vars"]["nMuons" + out_label] = ak.num(events.Muon).to_list()
        output[dataset]["vars"]["nMuons_highPurity" + out_label] = ak.sum(
            muons.highPurity, axis=-1
        ).to_list()
        output[dataset]["vars"]["nMuons_isPFcand" + out_label] = ak.sum(
            muons.isPFcand, axis=-1
        ).to_list()
        output[dataset]["vars"]["nMuons_looseId" + out_label] = ak.sum(
            muons.looseId, axis=-1
        ).to_list()
        output[dataset]["vars"]["nMuons_mediumId" + out_label] = ak.sum(
            muons.mediumId, axis=-1
        ).to_list()
        output[dataset]["vars"]["nMuons_tightId" + out_label] = ak.sum(
            muons.tightId, axis=-1
        ).to_list()
        output[dataset]["vars"]["nMuons_triggerIdLoose" + out_label] = ak.sum(
            muons.triggerIdLoose, axis=-1
        ).to_list()
        output[dataset]["vars"]["nMuons_isTracker" + out_label] = ak.sum(
            muons.isTracker, axis=-1
        ).to_list()
        output[dataset]["vars"]["muon_pt_mean" + out_label] = ak.mean(
            muons.pt, axis=-1
        ).to_list()
        output[dataset]["vars"]["muon_dxy_mean" + out_label] = ak.mean(
            abs(muons.dxy), axis=-1
        ).to_list()
        output[dataset]["vars"]["muon_dz_mean" + out_label] = ak.mean(
            abs(muons.dz), axis=-1
        ).to_list()
        output[dataset]["vars"]["muon_ip3d_mean" + out_label] = ak.mean(
            muons.ip3d, axis=-1
        ).to_list()
        output[dataset]["vars"]["muon_pt_leading" + out_label] = muons.pt[
            :, 0
        ].to_list()
        output[dataset]["vars"]["muon_pt_subleading" + out_label] = muons.pt[
            :, 1
        ].to_list()
        output[dataset]["vars"][
            "muon_miniPFRelIso_all_leading" + out_label
        ] = muons.miniPFRelIso_all[:, 0].to_list()
        output[dataset]["vars"][
            "muon_miniPFRelIso_all_subleading" + out_label
        ] = muons.miniPFRelIso_all[:, 1].to_list()
        output[dataset]["vars"]["muon_miniPFRelIso_all_mean" + out_label] = ak.mean(
            muons.miniPFRelIso_all, axis=-1
        ).to_list()
        output[dataset]["vars"]["muon_l_sl_deltaphi" + out_label] = abs(
            ak.materialized(muonsCollection[:, 0]).deltaphi(
                ak.materialized(muonsCollection[:, 1])
            )
        ).to_list()
        output[dataset]["vars"]["muon_l_sl_deltaeta" + out_label] = abs(
            ak.materialized(muonsCollection[:, 0]).deltaeta(
                ak.materialized(muonsCollection[:, 1])
            )
        ).to_list()
        output[dataset]["vars"]["muon_l_sl_deltaR" + out_label] = (
            ak.materialized(muonsCollection[:, 0])
            .deltaR(ak.materialized(muonsCollection[:, 1]))
            .to_list()
        )
        eigs_muons = SUEP_utils.sphericity(muons, 1.0)
        output[dataset]["vars"]["muon_S1" + out_label] = 1.5 * (
            eigs_muons[:, 1] + eigs_muons[:, 0]
        )
        eigs_tracks = SUEP_utils.sphericity(tracks, 1.0)
        output[dataset]["vars"]["event_S1" + out_label] = 1.5 * (
            eigs_tracks[:, 1] + eigs_tracks[:, 0]
        )

        leptons = ak.concatenate([muonsCollection, electronsCollection], axis=-1)
        leading_muons = leptons[:, 0]
        output[dataset]["vars"][
            "muon_interIsolation_0p2" + out_label
        ] = SUEP_utils.inter_isolation(leading_muons, leptons, dR=0.2).to_list()
        output[dataset]["vars"][
            "muon_interIsolation_0p4" + out_label
        ] = SUEP_utils.inter_isolation(leading_muons, leptons, dR=0.4).to_list()
        output[dataset]["vars"][
            "muon_interIsolation_0p8" + out_label
        ] = SUEP_utils.inter_isolation(leading_muons, leptons, dR=0.8).to_list()
        output[dataset]["vars"][
            "muon_interIsolation_1p6" + out_label
        ] = SUEP_utils.inter_isolation(leading_muons, leptons, dR=1.6).to_list()

        # Eta ring variables
        output[dataset]["vars"][
            "nMuons_eta_ring_0p2" + out_label
        ] = SUEP_utils.n_eta_ring(muonsCollection, 0.2).to_list()
        output[dataset]["vars"][
            "nMuons_eta_ring_0p4" + out_label
        ] = SUEP_utils.n_eta_ring(muonsCollection, 0.4).to_list()
        output[dataset]["vars"][
            "nMuons_eta_ring_0p8" + out_label
        ] = SUEP_utils.n_eta_ring(muonsCollection, 0.8).to_list()
        output[dataset]["vars"][
            "nMuons_eta_ring_1p6" + out_label
        ] = SUEP_utils.n_eta_ring(muonsCollection, 1.6).to_list()

    def initializeColumns(self, label=""):
        # need to add these to dataframe when no events pass to make the merging work
        # for some reason, initializing these as empty and then trying to fill them doesn't work
        self.columns_CL = [
            "SUEP_nconst_CL",
            "SUEP_ntracks_CL",
            "SUEP_pt_avg_CL",
            "SUEP_pt_avg_b_CL",
            "SUEP_S1_CL",
            "SUEP_rho0_CL",
            "SUEP_rho1_CL",
            "SUEP_pt_CL",
            "SUEP_eta_CL",
            "SUEP_phi_CL",
            "SUEP_mass_CL",
            "dphi_SUEP_ISR_CL",
        ]
        self.columns_CL_ISR = [c.replace("SUEP", "ISR") for c in self.columns_CL]

        self.columns_ML, self.columns_ML_ISR = [], []
        if self.do_inf:
            self.columns_ML = ["SUEP_" + m + "_GNN" for m in self.dgnn_model_names] + [
                "SUEP_S1_GNN",
                "SUEP_nconst_GNN",
            ]
            self.columns_ML += [m + "_ssd" for m in self.ssd_models]
            self.columns_ML_ISR = [c.replace("SUEP", "ISR") for c in self.columns_ML]

        self.columns = (
            self.columns_CL
            + self.columns_CL_ISR
            + self.columns_ML
            + self.columns_ML_ISR
        )

        # add a specific label to all columns
        for iCol in range(len(self.columns)):
            self.columns[iCol] = self.columns[iCol] + label

    def analysis(self, events, output, do_syst=False, col_label=""):
        #####################################################################################
        # ---- Trigger event selection
        # Cut based on ak4 jets to replicate the trigger
        #####################################################################################

        # get dataset name
        dataset = events.metadata["dataset"]

        # some cutflow stuff
        output[dataset]["cutflow"].fill(
            len(events) * ["all events"], weight=events.genWeight
        )
        output[dataset]["cutflow"].fill(
            ak.sum(events.HLT.PFHT430 == 1) * ["HLT_PFHT430"]
        )
        output[dataset]["cutflow"].fill(
            ak.sum(events.HLT.TripleMu_5_3_3_Mass3p8_DZ == 1) * ["HLT_TripleMu_5_3_3"],
            weight=events[events.HLT.TripleMu_5_3_3_Mass3p8_DZ == 1].genWeight,
        )

        # golden jsons for offline data
        if not self.isMC and self.scouting != 1:
            events = applyGoldenJSON(self, events)

        if self.trigger != "TripleMu":
            events, electrons, muons = ZH_utils.selectByLeptons(
                self, events, lepveto=True
            )
        events = self.eventSelection(events)

        # make sure we have at least 3 muons with loose ID
        if self.trigger == "TripleMu":
            events, electrons, muons = self.muon_filter(events, 4)
        output[dataset]["cutflow"].fill(
            len(events) * ["nMuon_mediumId >= 4"], weight=events.genWeight
        )
        n_muons = ak.num(muons)
        output[dataset]["cutflow"].fill(
            len(events[n_muons >= 6]) * ["nMuon_mediumId >= 6"],
            weight=events[n_muons >= 6].genWeight,
        )

        # output empty dataframe if no events pass trigger
        if len(events) == 0:
            return output

        #####################################################################################
        # ---- Track selection
        # Prepare the clean PFCand matched to tracks collection
        #####################################################################################

        if self.scouting == 1:
            tracks, Cleaned_cands = self.getScoutingTracks(events)
        else:
            tracks, Cleaned_cands = self.getTracks(events)

        if self.isMC and do_syst:
            tracks = track_killing(self, tracks)
            Cleaned_cands = track_killing(self, Cleaned_cands)

        #####################################################################################
        # ---- FastJet reclustering
        # The jet clustering part
        #####################################################################################

        ak_inclusive_jets, ak_inclusive_cluster = SUEP_utils.FastJetReclustering(
            tracks, r=1.5, min_pt=50
        )

        #####################################################################################
        # ---- Event level information
        #####################################################################################

        self.storeEventVars(
            events,
            output,
            tracks,
            electrons,
            muons,
            ak_inclusive_jets,
            ak_inclusive_cluster,
            out_label=col_label,
        )

        # indices of events in tracks, used to keep track which events pass selections
        indices = np.arange(0, len(tracks))

        # initialize the columns with all the variables that you want to fill
        self.initializeColumns(col_label)

        #####################################################################################
        # ---- Cut Based Analysis
        #####################################################################################

        # Check the effect of higher pT cuts on the SUEP and ISR jets
        (
            ak_inclusive_jets_highPt,
            ak_inclusive_cluster_highPt,
        ) = SUEP_utils.FastJetReclustering(tracks, r=1.5, min_pt=150)
        clusterCut_highPt = ak.num(ak_inclusive_jets_highPt, axis=1) > 1
        output[dataset]["cutflow"].fill(
            len(events[clusterCut_highPt]) * ["n_ak15 >= 2 (150 GeV)"],
            weight=events[clusterCut_highPt].genWeight,
        )

        # remove events with at least 2 clusters (i.e. need at least SUEP and ISR jets for IRM)
        clusterCut = ak.num(ak_inclusive_jets, axis=1) > 1
        ak_inclusive_cluster = ak_inclusive_cluster[clusterCut]
        ak_inclusive_jets = ak_inclusive_jets[clusterCut]
        tracks = tracks[clusterCut]
        indices = indices[clusterCut]
        events = events[clusterCut]
        output[dataset]["cutflow"].fill(
            len(events) * ["n_ak15 >= 2 (50 GeV)"],
            weight=events.genWeight,
        )

        # output file if no events pass selections, avoids errors later on
        if len(tracks) == 0:
            for c in self.columns:
                output[dataset]["vars"][c] = np.nan
            return output

        tracks, indices, topTwoJets = SUEP_utils.getTopTwoJets(
            self, tracks, indices, ak_inclusive_jets, ak_inclusive_cluster
        )
        SUEP_cand, ISR_cand, SUEP_cluster_tracks, ISR_cluster_tracks = topTwoJets

        # output file if no events pass selections, avoids errors later on
        if len(tracks) == 0:
            for c in self.columns:
                output[dataset]["vars"][c] = np.nan
            return output

        SUEP_utils.ClusterMethod(
            self,
            output,
            dataset,
            indices,
            tracks,
            SUEP_cand,
            ISR_cand,
            SUEP_cluster_tracks,
            ISR_cluster_tracks,
            do_inverted=True,
            out_label=col_label,
        )

        if self.do_inf:
            import workflows.ML_utils as ML_utils

            ML_utils.DGNNMethod(
                self,
                indices,
                SUEP_tracks=SUEP_cluster_tracks,
                SUEP_cand=SUEP_cand,
                ISR_tracks=ISR_cluster_tracks,
                ISR_cand=ISR_cand,
                out_label=col_label,
                do_inverted=True,
            )

    def process(self, events):
        dataset = events.metadata["dataset"]
        cutflow = hist.Hist.new.StrCategory(
            [
                "all events",
                "HLT_TripleMu_5_3_3",
                "nMuon_mediumId >= 4",
                "nMuon_mediumId >= 6",
                "n_ak15 >= 2 (50 GeV)",
                "n_ak15 >= 2 (150 GeV)",
            ],
            name="cutflow",
            label="cutflow",
        ).Weight()
        output = {
            dataset: {
                "cutflow": cutflow,
                "gensumweight": processor.value_accumulator(float, 0),
                "vars": pandas_accumulator(pd.DataFrame()),
            },
        }

        # gen weights
        if self.isMC and self.scouting == 1:
            self.gensumweight = ak.num(events.PFcand.pt, axis=0)
            output[dataset]["gensumweight"].add(self.gensumweight)
        elif self.isMC:
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
