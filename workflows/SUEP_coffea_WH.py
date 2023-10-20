"""
SUEP_coffea_WH.py
Coffea producer for SUEP WH analysis. Uses fastjet package to recluster large jets:
https://github.com/scikit-hep/fastjet
Pietro Lugato, Chad Freer, Luca Lavezzo, 2023
"""
from typing import Optional

import awkward as ak
import numpy as np
import pandas as pd
import vector
from coffea import processor

# IO utils
import workflows.pandas_utils as pandas_utils
from workflows.pandas_accumulator import pandas_accumulator

# Importing SUEP specific functions
import workflows.SUEP_utils as SUEP_utils
import workflows.WH_utils as WH_utils

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


class SUEP_cluster_WH(processor.ProcessorABC):
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
        self.prefixes = {"SUEP": "SUEP"}
        self.doOF = False
        self.accum = accum
        self.trigger = trigger
        self.out_vars = pd.DataFrame()

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

    def triggerSelection(self, events, output, out_label):
        """
        Applies trigger, returns events.
        Default is PFHT triggers. Can use selection variable for customization.
        """

        triggerSingleMuon = (
            events.HLT.IsoMu30
            | events.HLT.IsoMu27
            | events.HLT.IsoMu24
            | events.HLT.Mu50
        )
        triggerDoubleMuon = (
            events.HLT.Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass8
            | events.HLT.Mu19_TrkIsoVVL_Mu9_TrkIsoVVL_DZ_Mass8
            | events.HLT.Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8
            | events.HLT.Mu19_TrkIsoVVL_Mu9_TrkIsoVVL_DZ_Mass3p8
        )
        triggerEGamma = (
            events.HLT.Ele27_WPTight_Gsf
            | events.HLT.Ele32_WPTight_Gsf
            | events.HLT.Ele35_WPTight_Gsf
            | events.HLT.Ele38_WPTight_Gsf
            | events.HLT.Ele40_WPTight_Gsf
            | events.HLT.Ele115_CaloIdVT_GsfTrkIdT
            | events.HLT.Ele23_Ele12_CaloIdL_TrackIdL_IsoVL
            | events.HLT.Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ
            | events.HLT.Ele32_WPTight_Gsf_L1DoubleEG
            | events.HLT.DoubleEle27_CaloIdL_MW
            | events.HLT.DoubleEle25_CaloIdL_MW
            | events.HLT.DoubleEle33_CaloIdL_MW
            | events.HLT.DiEle27_WPTightCaloOnly_L1DoubleEG
            | events.HLT.Photon200
            | events.HLT.DoublePhoton70
        )

        # this is just for cutflow
        output["triggerSingleMuon"+out_label] += len(events[triggerSingleMuon])
        output["triggerDoubleMuon"+out_label] += len(events[triggerDoubleMuon])
        output["triggerEGamma"+out_label] += len(events[triggerEGamma])

        events = events[triggerDoubleMuon | triggerEGamma | triggerSingleMuon]

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

    def getTracks(self, events, lepton=None, leptonIsolation=None):
        Cands = ak.zip(
            {
                "pt": events.PFCands.trkPt,
                "eta": events.PFCands.trkEta,
                "phi": events.PFCands.trkPhi,
                "mass": events.PFCands.mass,
                # "pdgID": events.PFCands.pdgID
            },
            with_name="Momentum4D",
        )
        cut = (
            (events.PFCands.fromPV > 1)
            & (events.PFCands.trkPt >= 1)
            & (abs(events.PFCands.trkEta) <= 2.5)
            & (abs(events.PFCands.dz) < 0.05)
            # & (events.PFCands.dzErr < 0.05)
            & (abs(events.PFCands.d0) < 0.05)
            & (events.PFCands.puppiWeight > 0.1)
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
            & (events.lostTracks.pt >= 0.1)
            & (abs(events.lostTracks.eta) <= 2.5)
            & (abs(events.lostTracks.dz) < 0.05)
            & (abs(events.lostTracks.d0) < 0.05)
            # & (events.lostTracks.dzErr < 0.05)
            & (events.lostTracks.puppiWeight > 0.1)
        )
        Lost_Tracks_cands = LostTracks[cut]
        Lost_Tracks_cands = ak.packed(Lost_Tracks_cands)

        # select which tracks to use in the script
        # dimensions of tracks = events x tracks in event x 4 momenta
        tracks = ak.concatenate([Cleaned_cands, Lost_Tracks_cands], axis=1)

        if leptonIsolation:
            # Sorting out the tracks that overlap with the lepton
            tracks = tracks[
                (tracks.deltaR(lepton[:, 0]) >= leptonIsolation)
                # & (tracks.deltaR(lepton) >= 0.4)
            ]

        return tracks, Cleaned_cands

    def storeEventVars(
        self,
        events,
        tracks,
        ak_inclusive_jets,
        ak_inclusive_cluster,
        lepton,
        output,
        out_label="",
    ):
        # select out ak4jets
        ak4jets = self.jet_awkward(events.Jet)

        # work on JECs and systematics
        prefix = ""
        if self.accum:
            if "dask" in self.accum:
                prefix = "dask-worker-space/"
        jets_c = apply_jecs(
            self,
            Sample=self.sample,
            events=events,
            prefix=prefix,
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
        output['vars']["ntracks" + out_label] = ak.num(tracks).to_list()
        output['vars']["ngood_fastjets" + out_label] = ak.num(
            ak_inclusive_jets
        ).to_list()

        if out_label == "":
            output['vars']["ht" + out_label] = ak.sum(ak4jets.pt, axis=-1).to_list()
            output['vars']["ht_JEC" + out_label] = ak.sum(jets_jec.pt, axis=-1).to_list()
            output['vars']["ht_JEC" + out_label + "_JER_up"] = ak.sum(
                jets_jec_JERUp.pt, axis=-1
            ).to_list()
            output['vars']["ht_JEC" + out_label + "_JER_down"] = ak.sum(
                jets_jec_JERDown.pt, axis=-1
            ).to_list()
            output['vars']["ht_JEC" + out_label + "_JES_up"] = ak.sum(
                jets_jec_JESUp.pt, axis=-1
            ).to_list()
            output['vars']["ht_JEC" + out_label + "_JES_down"] = ak.sum(
                jets_jec_JESDown.pt, axis=-1
            ).to_list()

            # store event weights for MC
            if self.isMC and self.scouting == 0:
                output['vars']["genweight"] = events.genWeight
            elif self.isMC and self.scouting == 1:
                output['vars']["genweight"] = [
                    1.0 for e in (len(events) * [0])
                ]  # create awkward array of ones

            output['vars']["ngood_ak4jets" + out_label] = ak.num(ak4jets).to_list()

            if self.isMC:
                output['vars']["Pileup_nTrueInt" + out_label] = events.Pileup.nTrueInt
                GetPSWeights(self, events)  # Parton Shower weights
                GetPrefireWeights(self, events)  # Prefire weights
            output['vars']["PV_npvs" + out_label] = events.PV.npvs
            output['vars']["PV_npvsGood" + out_label] = events.PV.npvsGood

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

        output['vars']["SUEP_genMass" + out_label] = SUEP_genMass
        output['vars']["SUEP_genPt" + out_label] = SUEP_genPt
        output['vars']["SUEP_genEta" + out_label] = SUEP_genEta
        output['vars']["SUEP_genPhi" + out_label] = SUEP_genPhi

        # saving lepton kinematics

        output['vars']["lepton_pt" + out_label] = lepton.pt[:, 0]
        output['vars']["lepton_eta" + out_label] = lepton.eta[:, 0]
        output['vars']["lepton_phi" + out_label] = lepton.phi[:, 0]
        output['vars']["lepton_mass" + out_label] = lepton.mass[:, 0]


    def analysis(self, events, output, do_syst=False, out_label=""):
        #####################################################################################
        # ---- Basic event selection
        # Apply triggers, quality filters, MET, and one lepton selections.
        #####################################################################################

        output["total"+out_label] += len(events)

        # golden jsons for offline data
        if self.isMC == 0:
            events = applyGoldenJSON(self, events)

        output["goldenJSON"+out_label] += len(events)

        events = self.triggerSelection(events, output, out_label)
        output["all_triggers"+out_label] += len(events)

        events = self.selectByFilters(events)
        output["qualityFilters"+out_label] += len(events)

        # TODO: MET
        output["MET"+out_label] += len(events)

        events, selLeptons = WH_utils.selectByLeptons(self, events, lepveto=True)
        output["oneLepton"+out_label] += len(events)

        # output empty dataframe if no events pass basic event selection
        if len(events) == 0:
            print("No events passed basic event selection. Saving empty outputs.")
            return output

        #####################################################################################
        # ---- Track selection
        # Prepare the clean PFCand matched to tracks collection, imposing a dR > 0.4
        # cut on tracks from the selected lepton
        #####################################################################################

        tracks, _ = self.getTracks(
            events, lepton=selLeptons, leptonIsolation=0.4
        )

        if self.isMC and do_syst:
            tracks = track_killing(self, tracks)

        #####################################################################################
        # ---- FastJet reclustering
        # The jet clustering part
        #####################################################################################

        ak_inclusive_jets, ak_inclusive_cluster = SUEP_utils.FastJetReclustering(
            tracks, r=1.5, minPt=60
        )

        #####################################################################################
        # ---- Store event level information
        #####################################################################################

        self.storeEventVars(
            events,
            tracks,
            ak_inclusive_jets,
            ak_inclusive_cluster,
            lepton=selLeptons,
            output=output,
            out_label=out_label,
        )

        #####################################################################################
        # ---- Cut Based Analysis
        #####################################################################################

        # indices of events in tracks, used to keep track which events pass selections
        indices = np.arange(0, len(tracks))

        # remove events with less than 1 cluster (i.e. need at least SUEP candidate cluster)
        clusterCut = ak.num(ak_inclusive_jets, axis=1) > 0
        ak_inclusive_cluster = ak_inclusive_cluster[clusterCut]
        ak_inclusive_jets = ak_inclusive_jets[clusterCut]
        selLeptons = selLeptons[clusterCut]
        tracks = tracks[clusterCut]
        indices = indices[clusterCut]

        # output file if no events pass selections, avoids errors later on
        if len(tracks) == 0:
            print("No events pass clusterCut.")
            return output

        WH_utils.TopPTMethod(
            self,
            indices,
            tracks,
            ak_inclusive_jets,
            ak_inclusive_cluster,
            output=output,
            out_label=out_label,
        )

        return output

    def process(self, events):
        dataset = events.metadata["dataset"]

        output = processor.dict_accumulator({
            "gensumweight": processor.value_accumulator(float, 0),
            "total": processor.value_accumulator(float, 0),
            "goldenJSON": processor.value_accumulator(float, 0),
            "triggerSingleMuon": processor.value_accumulator(float, 0),
            "triggerDoubleMuon": processor.value_accumulator(float, 0),
            "triggerEGamma": processor.value_accumulator(float, 0),
            "all_triggers": processor.value_accumulator(float, 0),
            "oneLepton": processor.value_accumulator(float, 0),
            "qualityFilters": processor.value_accumulator(float, 0),
            "MET": processor.value_accumulator(float, 0),
            "vars": pandas_accumulator(pd.DataFrame()),
        })

        # gen weights
        output['gensumweight'] = ak.sum(events.genWeight)

        # run the analysis with the track systematics applied
        if self.isMC and self.do_syst:
            output.update({
                "total_track_down": processor.value_accumulator(float, 0),
                "goldenJSON_track_down": processor.value_accumulator(float, 0),
                "triggerSingleMuon_track_down": processor.value_accumulator(float, 0),
                "triggerDoubleMuon_track_down": processor.value_accumulator(float, 0),
                "triggerEGamma_track_down": processor.value_accumulator(float, 0),
                "all_triggers_track_down": processor.value_accumulator(float, 0),
                "oneLepton_track_down": processor.value_accumulator(float, 0),
                "qualityFilters_track_down": processor.value_accumulator(float, 0),
                "MET_track_down": processor.value_accumulator(float, 0),
            })
            output = self.analysis(
                events, output, do_syst=True, out_label="_track_down"
            )

        # run the analysis
        output = self.analysis(events, output)

        return {dataset: output}

    def postprocess(self, accumulator):
        return accumulator
