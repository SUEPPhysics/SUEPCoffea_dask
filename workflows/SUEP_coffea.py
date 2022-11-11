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

vector.register_awkward()

import workflows.pandas_utils as pandas_utils

# Importing SUEP specific functions
import workflows.SUEP_utils as SUEP_utils
import workflows.ZH_utils as ZH_utils

# Importing CMS corrections
from workflows.CMS_corrections.golden_jsons_utils import applyGoldenJSON
from workflows.CMS_corrections.jetmet_utils import apply_jecs
from workflows.CMS_corrections.PartonShower_utils import GetPSWeights
from workflows.CMS_corrections.track_killing_utils import track_killing


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
        accum: Optional[bool],
        output_location: Optional[str],
        trigger: Optional[str],
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
        self.accum = False if accum is False or None else True
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
            if self.trigger == "TripleMu":
                if self.era == 2018:
                    trigger = events.HLT.TripleMu_5_3_3_Mass3p8_DZ == 1
                elif self.era == 2017:
                    trigger = events.HLT.TripleMu_5_3_3_Mass3p8to60_DZ == 1
                else:
                    trigger = events.HLT.TripleMu_5_3_3 == 1
            else:
                if self.era == 2016:
                    trigger = events.HLT.PFHT900 == 1
                else:
                    trigger = events.HLT.PFHT1050 == 1
            events = events[trigger]

        return events

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
        self, events, tracks, ak_inclusive_jets, ak_inclusive_cluster, out_label=""
    ):

        # select out ak4jets
        ak4jets = self.jet_awkward(events.Jet)

        # work on JECs and systematics
        jets_c = apply_jecs(
            isMC=self.isMC, Sample=self.sample, era=self.era, events=events
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
                jets_jec.pt, axis=-1
            ).to_list()

            if self.era == 2016 and self.scouting == 0:
                self.out_vars["HLT_PFHT900" + out_label] = events.HLT.PFHT900
            elif self.scouting == 0:
                self.out_vars["HLT_PFHT1050" + out_label] = events.HLT.PFHT1050
            self.out_vars["ngood_ak4jets" + out_label] = ak.num(ak4jets).to_list()
            if self.scouting == 1:
                self.out_vars["PV_npvs" + out_label] = ak.num(events.Vertex.x)
            else:
                if self.isMC:
                    self.out_vars[
                        "Pileup_nTrueInt" + out_label
                    ] = events.Pileup.nTrueInt
                    GetPSWeights(self, events)  # Parton Shower weights
                    # GetPrefireWeights(self, events)#Prefire weights (commented out until A02 is ready)
                self.out_vars["PV_npvs" + out_label] = events.PV.npvs
                self.out_vars["PV_npvsGood" + out_label] = events.PV.npvsGood

        # get gen SUEP mass
        SUEP_genMass = len(events) * [0]
        SUEP_genPt = len(events) * [0]
        if self.isMC and not self.scouting:
            genParts = self.getGenTracks(events)
            genSUEP = genParts[(abs(genParts.pdgID) == 25)]
            # we need to grab the last SUEP in the chain for each event
            SUEP_genMass = [g[-1].mass if len(g) > 0 else 0 for g in genSUEP]
            SUEP_genPt = [g[-1].pt if len(g) > 0 else 0 for g in genSUEP]
        self.out_vars["SUEP_genMass" + out_label] = SUEP_genMass
        self.out_vars["SUEP_genPt" + out_label] = SUEP_genPt

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
            self.columns_ML = [m + "_GNN" for m in self.dgnn_model_names] + [
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
        if not self.isMC and self.scouting != 1:
            events = applyGoldenJSON(self, events)
        events, electrons, muons = ZH_utils.selectByLeptons(self, events, lepveto=True)
        events = self.eventSelection(events)

        # output empty dataframe if no events pass trigger
        if len(events) == 0:
            print("No events passed trigger. Saving empty outputs.")
            if self.accum is False:
                self.out_vars = pd.DataFrame(["empty"], columns=["empty"])
            else:
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

        if self.isMC and do_syst:
            tracks = track_killing(self, tracks)
            Cleaned_cands = track_killing(self, Cleaned_cands)

        #####################################################################################
        # ---- FastJet reclustering
        # The jet clustering part
        #####################################################################################

        ak_inclusive_jets, ak_inclusive_cluster = SUEP_utils.FastJetReclustering(
            tracks, r=1.5, minPt=150
        )

        #####################################################################################
        # ---- Event level information
        #####################################################################################

        self.storeEventVars(
            events, tracks, ak_inclusive_jets, ak_inclusive_cluster, out_label=col_label
        )

        # indices of events in tracks, used to keep track which events pass selections
        indices = np.arange(0, len(tracks))

        # initialize the columns with all the variables that you want to fill
        self.initializeColumns(col_label)

        #####################################################################################
        # ---- Cut Based Analysis
        #####################################################################################

        # remove events with at least 2 clusters (i.e. need at least SUEP and ISR jets for IRM)
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

        # save the out_vars object as a Pandas DataFrame
        if self.accum is False:
            pandas_utils.save_dfs(
                self,
                [self.out_vars],
                ["vars"],
                events.behavior["__events_factory__"]._partition_key.replace("/", "_")
                + ".hdf5",
            )
        else:
            # Convert output to the desired format when the accumulator is used
            for c in out_vars_keys:
                output[c] = self.out_vars[c].to_list()
            output = {dataset: self.out_vars}
        return output

    def postprocess(self, accumulator):
        return accumulator
