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

# IO utils
import workflows.pandas_utils as pandas_utils

# Importing SUEP specific functions
import workflows.SUEP_utils as SUEP_utils
import workflows.ZH_utils as ZH_utils

# Importing CMS corrections
from workflows.CMS_corrections.golden_jsons_utils import applyGoldenJSON
from workflows.CMS_corrections.HEM_utils import jetHEMFilter
from workflows.CMS_corrections.jetmet_utils import apply_jecs
from workflows.CMS_corrections.PartonShower_utils import GetPSWeights
from workflows.CMS_corrections.Prefire_utils import GetPrefireWeights
from workflows.CMS_corrections.track_killing_utils import (
    scout_track_killing,
    track_killing,
)

# Set vector behavior
vector.register_awkward()


class SUEP_cluster(processor.ProcessorABC):
    def __init__(
        self,
        isMC: int,
        era: str,
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
    ) -> None:
        self._flag = flag
        self.output_location = output_location
        self.do_syst = do_syst
        self.gensumweight = 1.0
        self.scouting = scouting
        self.era = era.lower()
        self.isMC = isMC
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
        self.out_vars = pd.DataFrame()

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

        # Set up for the histograms
        self._accumulator = processor.dict_accumulator({})

    @property
    def accumulator(self):
        return self._accumulator

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
            },
            with_name="Momentum4D",
        )
        if self.scouting == 1:
            jet_awk_Cut = (Jets_awk.pt > 30) & (abs(Jets_awk.eta) < 2.6)
        else:
            jet_awk_Cut = (Jets_awk.pt > 30) & (abs(Jets_awk.eta) < 2.4)
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
            if self.trigger == "TripleMu":
                if self.era == "2016" or self.era == "2016apv":
                    trigger = events.HLT.TripleMu_5_3_3 == 1
                elif self.era == "2017":
                    trigger = events.HLT.TripleMu_5_3_3_Mass3p8to60_DZ == 1
                else:
                    trigger = events.HLT.TripleMu_5_3_3_Mass3p8_DZ == 1
            else:
                if self.era == "2016" or self.era == "2016apv":
                    trigger = events.HLT.PFHT900 == 1
                else:
                    trigger = events.HLT.PFHT1050 == 1
            events = events[trigger]
        else:
            # if self.era == "2016" or self.era == "2016apv":
            #    trigger = events.hltResult[:,3] == 1 # require trigger DST_HT410_PFScouting_v2
            # elif self.era == "2017":
            #    trigger = events.hltResult[:,5] == 1 # require trigger DST_HT410_PFScouting_v12
            # elif self.era == "2018":
            #    trigger = events.hltResult[:,7] == 1 # require trigger DST_HT410_PFScouting_v16
            trigger = events.scouting.trig == 1
            events = events[trigger]
        return events

    def selectByFilters(self, events):
        ### Apply MET filter selection (see https://twiki.cern.ch/twiki/bin/viewauth/CMS/MissingETOptionalFiltersRun2)
        if self.era == "2018" or self.era == "2017":
            cutAnyFilter = (
                (events.Flag.goodVertices)
                & (events.Flag.globalSuperTightHalo2016Filter)
                & (events.Flag.HBHENoiseFilter)
                & (events.Flag.HBHENoiseIsoFilter)
                & (events.Flag.EcalDeadCellTriggerPrimitiveFilter)
                & (events.Flag.BadPFMuonFilter)
                & (events.Flag.BadPFMuonDzFilter)
                & (events.Flag.eeBadScFilter)
                & (events.Flag.ecalBadCalibFilter)
            )
        if self.era == "2016" or self.era == "2016apv":
            cutAnyFilter = (
                (events.Flag.goodVertices)
                & (events.Flag.globalSuperTightHalo2016Filter)
                & (events.Flag.HBHENoiseFilter)
                & (events.Flag.HBHENoiseIsoFilter)
                & (events.Flag.EcalDeadCellTriggerPrimitiveFilter)
                & (events.Flag.BadPFMuonFilter)
                & (events.Flag.BadPFMuonDzFilter)
                & (events.Flag.eeBadScFilter)
            )
        return events[cutAnyFilter]

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
        if "2016" in self.era:
            Cands = ak.zip(
                {
                    "pt": events.offlineTrack.pt,
                    "eta": events.offlineTrack.eta,
                    "phi": events.offlineTrack.phi,
                    "mass": events.offlineTrack.mass,
                },
                with_name="Momentum4D",
            )
            cut = (
                (events.offlineTrack.pt >= 0.75)
                & (abs(events.offlineTrack.eta) <= 2.4)
                & (events.offlineTrack.quality == 1)
            )
        else:
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
                & (abs(events.PFcand.eta) <= 2.4)
                & (events.PFcand.vertex == 0)
                & (events.PFcand.q != 0)
            )
        Cleaned_cands = Cands[cut]
        tracks = ak.packed(Cleaned_cands)
        return tracks, Cleaned_cands

    def getLooseLeptons(self, events):
        if self.scouting == 1:
            looseMuons = ak.zip(
                {
                    "pt": events.Muon.pt,
                    "eta": events.Muon.eta,
                    "phi": events.Muon.phi,
                    "mass": events.Muon.mass,
                },
                with_name="Momentum4D",
            )

            looseElectrons = ak.zip(
                {
                    "pt": events.Electron.pt,
                    "eta": events.Electron.eta,
                    "phi": events.Electron.phi,
                    "mass": events.Electron.mass,
                },
                with_name="Momentum4D",
            )

            cutLooseMuons = (events.Muon.pt >= 1) & (abs(events.Muon.eta) < 2.4)
            cutLooseElectrons = (events.Electron.pt >= 1) & (
                abs(events.Electron.eta) < 2.5
            )
        else:
            looseMuons = ak.zip(
                {
                    "pt": events.Muon.pt,
                    "eta": events.Muon.eta,
                    "phi": events.Muon.phi,
                    "mass": events.Muon.mass,
                    "charge": events.Muon.pdgId / (-13),
                },
                with_name="Momentum4D",
            )

            looseElectrons = ak.zip(
                {
                    "pt": events.Electron.pt,
                    "eta": events.Electron.eta,
                    "phi": events.Electron.phi,
                    "mass": events.Electron.mass,
                    "charge": events.Electron.pdgId / (-11),
                },
                with_name="Momentum4D",
            )

            cutLooseMuons = (
                (events.Muon.looseId)
                & (events.Muon.pt >= 1)
                & (abs(events.Muon.dxy) <= 0.02)
                & (abs(events.Muon.dz) <= 0.1)
                & (abs(events.Muon.eta) < 2.4)
            )
            cutLooseElectrons = (
                (events.Electron.cutBased >= 1)
                & (events.Electron.pt >= 1)
                & (
                    abs(events.Electron.dxy)
                    < 0.05 + 0.05 * (abs(events.Electron.eta) > 1.479)
                )
                & (
                    abs(events.Electron.dz)
                    < 0.10 + 0.10 * (abs(events.Electron.eta) > 1.479)
                )
                & (
                    (abs(events.Electron.eta) < 1.444)
                    | (abs(events.Electron.eta) > 1.566)
                )
                & (abs(events.Electron.eta) < 2.5)
            )

        ### Apply the cuts
        # Object selection. selMuons contain only the events that are filtered by cutMuons criteria.
        looseMuons = looseMuons[cutLooseMuons]
        looseElectrons = looseElectrons[cutLooseElectrons]

        return looseElectrons, looseMuons

    def storeEventVars(
        self,
        events,
        tracks,
        ak_inclusive_jets,
        ak_inclusive_cluster,
        electrons,
        muons,
        out_label="",
    ):
        # select out ak4jets
        if self.scouting and "2016" in self.era:
            ak4jets = self.jet_awkward(events.OffJet)
        else:
            ak4jets = self.jet_awkward(events.Jet)

        # work on JECs and systematics
        prefix = ""
        if self.accum:
            if "dask" in self.accum:
                prefix = "dask-worker-space/"
        jets_c, met_c = apply_jecs(
            self, Sample=self.sample, events=events, prefix=prefix
        )
        jet_HEM_Cut, _ = jetHEMFilter(self, jets_c, events.run)
        jets_c = jets_c[jet_HEM_Cut]
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
        self.out_vars["event" + out_label] = events.event.to_list()
        self.out_vars["run" + out_label] = events.run

        if self.scouting == 1:
            self.out_vars["lumi" + out_label] = events.lumSec
        else:
            self.out_vars["luminosityBlock" + out_label] = events.luminosityBlock
        self.out_vars["ntracks" + out_label] = ak.num(tracks).to_list()
        self.out_vars["ngood_fastjets" + out_label] = ak.num(
            ak_inclusive_jets
        ).to_list()

        if out_label == "":
            self.out_vars["ht" + out_label] = ak.sum(ak4jets.pt, axis=-1).to_list()
            self.out_vars["ht_JEC" + out_label] = ak.sum(jets_jec.pt, axis=-1).to_list()
            self.out_vars["ht_JEC" + out_label + "_JER_up"] = ak.sum(
                jets_jec_JERUp.pt, axis=-1
            ).to_list()
            self.out_vars["ht_JEC" + out_label + "_JER_down"] = ak.sum(
                jets_jec_JERDown.pt, axis=-1
            ).to_list()
            self.out_vars["ht_JEC" + out_label + "_JES_up"] = ak.sum(
                jets_jec_JESUp.pt, axis=-1
            ).to_list()
            self.out_vars["ht_JEC" + out_label + "_JES_down"] = ak.sum(
                jets_jec_JESDown.pt, axis=-1
            ).to_list()
            self.out_vars["n_sel_electrons"] = ak.to_numpy(ak.num(electrons))
            self.out_vars["n_sel_muons"] = ak.to_numpy(ak.num(muons))
            self.out_vars["n_sel_leps"] = ak.to_numpy(ak.num(electrons)) + ak.to_numpy(
                ak.num(muons)
            )
            self.out_vars["ngood_ak4jets" + out_label] = ak.num(ak4jets).to_list()

            # store event weights for MC
            if self.isMC and self.scouting == 0:
                self.out_vars["genweight"] = events.genWeight
            elif self.isMC and self.scouting == 1:
                self.out_vars["genweight"] = [
                    1.0 for e in (len(events) * [0])
                ]  # create awkward array of ones

            if "2016" in self.era and self.scouting == 0:
                self.out_vars["HLT_PFHT900" + out_label] = events.HLT.PFHT900
            elif self.scouting == 0:
                self.out_vars["HLT_PFHT1050" + out_label] = events.HLT.PFHT1050

            if self.scouting == 1:
                if self.isMC:
                    self.out_vars["Pileup_nTrueInt" + out_label] = events.PU.num
                self.out_vars["PV_npvs" + out_label] = ak.num(events.Vertex.x)
            else:
                if self.isMC:
                    self.out_vars[
                        "Pileup_nTrueInt" + out_label
                    ] = events.Pileup.nTrueInt
                self.out_vars["PV_npvs" + out_label] = events.PV.npvs
                self.out_vars["PV_npvsGood" + out_label] = events.PV.npvsGood

            if self.isMC:
                psweights = GetPSWeights(self, events)  # Parton Shower weights
                if len(psweights) == 4:
                    self.out_vars["PSWeight_ISR_up" + out_label] = psweights[0]
                    self.out_vars["PSWeight_ISR_down" + out_label] = psweights[1]
                    self.out_vars["PSWeight_FSR_up" + out_label] = psweights[2]
                    self.out_vars["PSWeight_FSR_down" + out_label] = psweights[3]
                else:
                    self.out_vars["PSWeight" + out_label] = psweights
                GetPrefireWeights(self, events)  # Prefire weights

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

        if self.isMC and self.scouting and "SUEP" in self.sample:
            SUEP_genMass = events.scalar.mass
            SUEP_genPt = events.scalar.pt
            SUEP_genPhi = events.scalar.phi
            SUEP_genEta = events.scalar.eta

        self.out_vars["SUEP_genMass" + out_label] = SUEP_genMass
        self.out_vars["SUEP_genPt" + out_label] = SUEP_genPt
        self.out_vars["SUEP_genEta" + out_label] = SUEP_genEta
        self.out_vars["SUEP_genPhi" + out_label] = SUEP_genPhi

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

    def analysis(self, events, do_syst=False, col_label=""):
        #####################################################################################
        # ---- Trigger event selection
        # Cut based on ak4 jets to replicate the trigger
        #####################################################################################

        # golden jsons for offline data
        if self.isMC == 0:
            events = applyGoldenJSON(self, events)
        events, _, _ = ZH_utils.selectByLeptons(self, events, lepveto=True)
        events = self.eventSelection(events)
        if self.scouting != 1:
            events = self.selectByFilters(events)

        # output empty dataframe if no events pass trigger
        if len(events) == 0:
            print("No events passed trigger. Saving empty outputs.")
            if self.accum == "pandas_merger":
                self.out_vars = pd.DataFrame(["empty"], columns=["empty"])
            elif self.accum:
                self.initializeColumns(col_label)
                for c in self.columns:
                    self.out_vars[c] = np.nan
            return

        #####################################################################################
        # ---- Track selection
        # Prepare the clean PFCand matched to tracks collection
        #####################################################################################
        if self.scouting == 1:
            tracks, Cleaned_cands = self.getScoutingTracks(events)
        else:
            tracks, Cleaned_cands = self.getTracks(events)
        looseElectrons, looseMuons = self.getLooseLeptons(events)
        if self.isMC and do_syst and self.scouting == 1:
            tracks = scout_track_killing(self, tracks)
            Cleaned_cands = scout_track_killing(self, Cleaned_cands)

        if self.isMC and do_syst and self.scouting == 0:
            tracks = track_killing(self, tracks)
            Cleaned_cands = track_killing(self, Cleaned_cands)

        #####################################################################################
        # ---- FastJet reclustering
        # The jet clustering part
        #####################################################################################

        if self.scouting == 1:
            min_FastJet = 50
        else:
            min_FastJet = 150

        ak_inclusive_jets, ak_inclusive_cluster = SUEP_utils.FastJetReclustering(
            tracks, r=1.5, minPt=min_FastJet
        )

        #####################################################################################
        # ---- Event level information
        #####################################################################################

        self.storeEventVars(
            events,
            tracks,
            ak_inclusive_jets,
            ak_inclusive_cluster,
            looseElectrons,
            looseMuons,
            out_label=col_label,
        )

        # indices of events in tracks, used to keep track which events pass selections
        indices = np.arange(0, len(tracks))

        # initialize the columns with all the variables that you want to fill
        self.initializeColumns(col_label)

        #####################################################################################
        # ---- Cut Based Analysis
        #####################################################################################

        # remove events with less than 2 clusters (i.e. need at least SUEP and ISR jets for IRM)
        clusterCut = ak.num(ak_inclusive_jets, axis=1) > 1
        ak_inclusive_cluster = ak_inclusive_cluster[clusterCut]
        ak_inclusive_jets = ak_inclusive_jets[clusterCut]
        tracks = tracks[clusterCut]
        indices = indices[clusterCut]

        # output file if no events pass selections, avoids errors later on
        if len(tracks) == 0:
            print("No events pass clusterCut.")
            for c in self.columns:
                self.out_vars[c] = np.nan
            return

        tracks, indices, topTwoJets = SUEP_utils.getTopTwoJets(
            self, tracks, indices, ak_inclusive_jets, ak_inclusive_cluster
        )
        SUEP_cand, ISR_cand, SUEP_cluster_tracks, ISR_cluster_tracks = topTwoJets

        SUEP_utils.ClusterMethod(
            self,
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
        output = self.accumulator.identity()
        dataset = events.metadata["dataset"]

        # gen weights
        if self.isMC and self.scouting == 1:
            self.gensumweight = ak.num(events.PFcand.pt, axis=0)
        elif self.isMC:
            self.gensumweight = ak.sum(events.genWeight)

        # run the analysis with the track systematics applied
        if self.isMC and self.do_syst:
            self.analysis(events, do_syst=True, col_label="_track_down")

        # run the analysis
        self.analysis(events)

        # output result to dask dataframe accumulator
        if self.accum:
            if "dask" in self.accum:
                return self.out_vars

            # output result to iterative/futures accumulator
            if "iterative" in self.accum or "futures" in self.accum:
                # Convert output to the desired format when the accumulator is used
                for c in self.out_vars.columns:
                    output[c] = self.out_vars[c].to_list()
                output = {dataset: self.out_vars}
                return output

            if "pandas_merger" == self.accum:
                # save the out_vars object as a Pandas DataFrame
                pandas_utils.save_dfs(
                    self,
                    [self.out_vars],
                    ["vars"],
                    "ntuple_"
                    + events.behavior["__events_factory__"]._partition_key.replace(
                        "/", "_"
                    )
                    + ".hdf5",
                )
                return output

    def postprocess(self, accumulator):
        return accumulator
