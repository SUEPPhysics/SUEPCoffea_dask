"""
SUEP_coffea.py
Coffea producer for SUEP analysis. Uses fastjet package to recluster large jets:
https://github.com/scikit-hep/fastjet
Chad Freer and Luca Lavezzo, 2021
"""
from coffea import processor
from typing import Optional
import awkward as ak
import pandas as pd
import numpy as np
import fastjet
import vector

vector.register_awkward()

# Importing SUEP specific functions
from workflows.pandas_utils import *
from workflows.math_utils import *


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
    ) -> None:
        self._flag = flag
        self.output_location = output_location
        self.do_syst = do_syst
        self.gensumweight = 1.0
        self.era = era
        self.isMC = isMC
        self.sample = sample
        self.syst_var, self.syst_suffix = (
            (syst_var, f"_sys_{syst_var}") if do_syst and syst_var else ("", "")
        )
        self.weight_syst = weight_syst
        self.prefixes = {"SUEP": "SUEP"}

        # Set up the image size and pixels
        self.eta_pix = 280
        self.phi_pix = 360
        self.eta_span = (-2.5, 2.5)
        self.phi_span = (-np.pi, np.pi)
        self.eta_scale = self.eta_pix / (self.eta_span[1] - self.eta_span[0])
        self.phi_scale = self.phi_pix / (self.phi_span[1] - self.phi_span[0])

        # Set up for the output arrays
        self._accumulator = processor.dict_accumulator({})

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, events):
        output = self.accumulator.identity()
        dataset = events.metadata["dataset"]

        if self.isMC:
            self.gensumweight = ak.sum(events.genWeight)

        # cut based on ak4 jets to replicate the trigger
        Jets = ak.zip(
            {
                "pt": events.Jet.pt,
                "eta": events.Jet.eta,
                "phi": events.Jet.phi,
                "mass": events.Jet.mass,
                "jetId": events.Jet.jetId,
            }
        )
        jetCut = (Jets.pt > 30) & (abs(Jets.eta) < 4.7)
        ak4jets = Jets[jetCut]
        ht = ak.sum(ak4jets.pt, axis=-1)

        # apply trigger selection
        trigger = events.HLT.TripleMu_5_3_3_Mass3p8_DZ == 1
        events = events[trigger]
        ak4jets = ak4jets[trigger]

        # output empty dataframe if no events pass trigger
        if len(events) == 0:
            print("No events passed trigger. Saving empty outputs.")
            out_vars = pd.DataFrame(["empty"], columns=["empty"])
            save_dfs(
                self,
                [out_vars],
                ["vars"],
                events.behavior["__events_factory__"]._partition_key.replace("/", "_")
                + ".hdf5",
            )
            return output

        #####################################################################################
        # ---- Track selection
        #####################################################################################

        # Prepare the clean PFCand matched to tracks collection
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
            & (events.PFCands.trkPt >= 0.7)
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
            & (events.lostTracks.pt >= 0.7)
            & (abs(events.lostTracks.eta) <= 1.0)
            & (abs(events.lostTracks.dz) < 10)
            & (events.lostTracks.dzErr < 0.05)
        )
        Lost_Tracks_cands = LostTracks[cut]
        Lost_Tracks_cands = ak.packed(Lost_Tracks_cands)

        # select which tracks to use in the script
        # dimensions of tracks = events x tracks in event x 4 momenta
        tracks = ak.concatenate([Cleaned_cands, Lost_Tracks_cands], axis=1)

        #####################################################################################
        # ---- FastJet reclustering
        #####################################################################################

        # minimum pT for reclustered jets
        minPt = 150

        # The jet clustering part
        jetdef = fastjet.JetDefinition(fastjet.antikt_algorithm, 1.5)
        cluster = fastjet.ClusterSequence(tracks, jetdef)

        # have to set min_pt = 0 and cut later to avoid some memory issues
        # FIXME: should try to understand this failure
        ak_inclusive_jets = ak.with_name(cluster.inclusive_jets(), "Momentum4D")
        ak_inclusive_cluster = ak.with_name(cluster.constituents(), "Momentum4D")

        # apply minimum pT cut
        minPtCut = ak_inclusive_jets.pt > minPt
        ak_inclusive_jets = ak_inclusive_jets[minPtCut]
        ak_inclusive_cluster = ak_inclusive_cluster[minPtCut]

        #####################################################################################
        # ---- Event level information
        #####################################################################################

        # from https://twiki.cern.ch/twiki/bin/view/CMS/JetID:
        # jetId==2 means: pass tight ID, fail tightLepVeto
        # jetId==6 means: pass tight and tightLepVeto ID.
        tightJetId = ak4jets.jetId > 2
        tight_ak4jets = ak4jets[tightJetId]
        looseJetId = ak4jets.jetId >= 2
        loose_ak4jets = ak4jets[looseJetId]

        # tracker jets
        trackerCut = abs(ak4jets.eta) < 2.4
        tracker_ak4jets = ak4jets[trackerCut]

        # save per event variables to a dataframe
        out_vars = pd.DataFrame()
        out_vars["ntracks"] = ak.num(tracks).to_list()
        out_vars["ngood_fastjets"] = ak.num(ak_inclusive_jets).to_list()
        out_vars["ht"] = ak.sum(ak4jets.pt, axis=-1).to_list()
        if self.era == 2016:
            out_vars["HLT_PFHT900"] = events.HLT.PFHT900
        else:
            out_vars["HLT_PFHT1050"] = events.HLT.PFHT1050
        out_vars["ngood_ak4jets"] = ak.num(ak4jets).to_list()
        out_vars["ngood_tracker_ak4jets"] = ak.num(tracker_ak4jets).to_list()
        out_vars["n_loose_ak4jets"] = ak.num(loose_ak4jets).to_list()
        out_vars["n_tight_ak4jets"] = ak.num(tight_ak4jets).to_list()
        out_vars["ht_loose"] = ak.sum(loose_ak4jets.pt, axis=-1).to_list()
        out_vars["ht_tight"] = ak.sum(tight_ak4jets.pt, axis=-1).to_list()
        out_vars["ht_tracker"] = ak.sum(tracker_ak4jets.pt, axis=-1).to_list()
        if self.isMC:
            out_vars["Pileup_nTrueInt"] = events.Pileup.nTrueInt
        out_vars["PV_npvs"] = events.PV.npvs
        out_vars["PV_npvsGood"] = events.PV.npvsGood

        # indices of events in tracks, used to keep track which events pass selections
        indices = np.arange(0, len(tracks))


        #####################################################################################
        # ---- Cut Based Analysis
        #####################################################################################

        # need to add these to dataframe when no events pass to make the merging work
        # for some reason, initializing these as empty and then trying to fill them doesn't work
        columns_IRM = [
            "SUEP_nconst_IRM",
            "SUEP_ntracks_IRM",
            "SUEP_pt_avg_IRM",
            "SUEP_pt_avg_b_IRM",
            "SUEP_pt_mean_scaled",
            "SUEP_S1_IRM",
            "SUEP_rho0_IRM",
            "SUEP_rho1_IRM",
            "SUEP_pt_IRM",
            "SUEP_eta_IRM",
            "SUEP_phi_IRM",
            "SUEP_mass_IRM",
            "dphi_SUEP_ISR_IRM",
        ]
        columns_CL = [c.replace("IRM", "CL") for c in columns_IRM]
        columns_CO = [c.replace("IRM", "CO") for c in columns_IRM]
        columns = columns_IRM + columns_CL + columns_CO
        if self.isMC:
            columns_CL_ISR = [
                c.replace("IRM", "CL".replace("SUEP", "ISR")) for c in columns_IRM
            ]
            columns_CO_ISR = [
                c.replace("IRM", "CO".replace("SUEP", "ISR")) for c in columns_IRM
            ]
            columns += columns_CL_ISR + columns_CO_ISR

        # remove events with at least 2 clusters (i.e. need at least SUEP and ISR jets for IRM)
        clusterCut = ak.num(ak_inclusive_jets, axis=1) > 1
        ak_inclusive_cluster = ak_inclusive_cluster[clusterCut]
        ak_inclusive_jets = ak_inclusive_jets[clusterCut]
        tracks = tracks[clusterCut]
        indices = indices[clusterCut]

        # output file if no events pass selections, avoids errors later on
        if len(tracks) == 0:
            print("No events pass clusterCut.")
            for c in columns:
                out_vars[c] = np.nan
            save_dfs(
                self,
                [out_vars],
                ["vars"],
                events.behavior["__events_factory__"]._partition_key.replace("/", "_")
                + ".hdf5",
            )
            return output

        # order the reclustered jets by pT (will take top 2 for ISR removal method)
        highpt_jet = ak.argsort(
            ak_inclusive_jets.pt, axis=1, ascending=False, stable=True
        )
        jets_pTsorted = ak_inclusive_jets[highpt_jet]
        clusters_pTsorted = ak_inclusive_cluster[highpt_jet]

        # at least 2 tracks in highest pt jet
        highpt_cands = clusters_pTsorted[:, 0]  # tracks for highest pt jet
        singletrackCut = ak.num(highpt_cands) > 1
        jets_pTsorted = jets_pTsorted[singletrackCut]
        clusters_pTsorted = clusters_pTsorted[singletrackCut]
        tracks = tracks[singletrackCut]
        indices = indices[singletrackCut]

        # number of constituents per jet, sorted by pT
        nconst_pTsorted = ak.num(clusters_pTsorted, axis=-1)

        # Top 2 pT jets. If jet1 has fewer tracks than jet2 then swap
        SUEP_cand = ak.where(
            nconst_pTsorted[:, 1] <= nconst_pTsorted[:, 0],
            jets_pTsorted[:, 0],
            jets_pTsorted[:, 1],
        )
        ISR_cand = ak.where(
            nconst_pTsorted[:, 1] > nconst_pTsorted[:, 0],
            jets_pTsorted[:, 0],
            jets_pTsorted[:, 1],
        )
        SUEP_cluster_tracks = ak.where(
            nconst_pTsorted[:, 1] <= nconst_pTsorted[:, 0],
            clusters_pTsorted[:, 0],
            clusters_pTsorted[:, 1],
        )
        ISR_cluster_tracks = ak.where(
            nconst_pTsorted[:, 1] > nconst_pTsorted[:, 0],
            clusters_pTsorted[:, 0],
            clusters_pTsorted[:, 1],
        )

        #####################################################################################
        # ---- ISR Removal Method (IRM)
        # In this method, we boost into the frame of the SUEP jet as selected previously
        # and select all tracks that are dphi > 1.6 from the ISR jet in this frame
        # to be the SUEP tracks. Variables such as sphericity are calculated using these.
        #####################################################################################

        # boost into frame of SUEP
        boost_SUEP = ak.zip(
            {
                "px": SUEP_cand.px * -1,
                "py": SUEP_cand.py * -1,
                "pz": SUEP_cand.pz * -1,
                "mass": SUEP_cand.mass,
            },
            with_name="Momentum4D",
        )
        ISR_cand_b = ISR_cand.boost_p4(boost_SUEP)
        tracks_b = tracks.boost_p4(boost_SUEP)

        # SUEP and IRM tracks as defined by IRS Removal Method (IRM):
        # all tracks outside/inside dphi 1.6 from ISR jet
        SUEP_tracks_b = tracks_b[abs(tracks_b.deltaphi(ISR_cand_b)) > 1.6]
        ISR_tracks_b = tracks_b[abs(tracks_b.deltaphi(ISR_cand_b)) <= 1.6]
        oneIRMtrackCut = ak.num(SUEP_tracks_b) > 1

        # output file if no events pass selections for ISR
        # avoids leaving this chunk without these columns
        if not any(oneIRMtrackCut):
            print("No events in ISR Removal Method, oneIRMtrackCut.")
            for c in columns_IRM:
                out_vars[c] = np.nan
        else:
            # remove the events left with one track
            SUEP_tracks_b_IRM = SUEP_tracks_b[oneIRMtrackCut]
            ISR_tracks_b_IRM = ISR_tracks_b[oneIRMtrackCut]
            SUEP_cand_IRM = SUEP_cand[oneIRMtrackCut]
            ISR_cand_IRM = ISR_cand[oneIRMtrackCut]
            tracks_IRM = tracks[oneIRMtrackCut]
            indices_IRM = indices[oneIRMtrackCut]

            out_vars.loc[indices_IRM, "SUEP_dphi_SUEP_ISR_IRM"] = ak.mean(
                abs(SUEP_cand_IRM.deltaphi(ISR_cand_IRM)), axis=-1
            )

            # SUEP jet variables
            eigs = sphericity(self, SUEP_tracks_b_IRM, 1.0)  # Set r=1.0 for IRC safe
            out_vars.loc[indices_IRM, "SUEP_nconst_IRM"] = ak.num(SUEP_tracks_b_IRM)
            out_vars.loc[indices_IRM, "SUEP_pt_avg_b_IRM"] = ak.mean(
                SUEP_tracks_b_IRM.pt, axis=-1
            )
            out_vars.loc[indices_IRM, "SUEP_pt_mean_scaled_IRM"] = ak.mean(
                SUEP_tracks_b_IRM.pt, axis=-1
            ) / ak.max(SUEP_tracks_b_IRM.pt, axis=-1)
            out_vars.loc[indices_IRM, "SUEP_S1_IRM"] = 1.5 * (eigs[:, 1] + eigs[:, 0])

            # unboost for these
            SUEP_tracks_IRM = SUEP_tracks_b_IRM.boost_p4(SUEP_cand_IRM)
            out_vars.loc[indices_IRM, "SUEP_pt_avg_IRM"] = ak.mean(
                SUEP_tracks_IRM.pt, axis=-1
            )
            deltaR = SUEP_tracks_IRM.deltaR(SUEP_cand_IRM)
            out_vars.loc[indices_IRM, "SUEP_rho0_IRM"] = rho(
                self, 0, SUEP_cand_IRM, SUEP_tracks_IRM, deltaR
            )
            out_vars.loc[indices_IRM, "SUEP_rho1_IRM"] = rho(
                self, 1, SUEP_cand_IRM, SUEP_tracks_IRM, deltaR
            )

            # redefine the jets using the tracks as selected by IRM
            SUEP = ak.zip(
                {
                    "px": ak.sum(SUEP_tracks_IRM.px, axis=-1),
                    "py": ak.sum(SUEP_tracks_IRM.py, axis=-1),
                    "pz": ak.sum(SUEP_tracks_IRM.pz, axis=-1),
                    "energy": ak.sum(SUEP_tracks_IRM.energy, axis=-1),
                },
                with_name="Momentum4D",
            )
            out_vars.loc[indices_IRM, "SUEP_pt_IRM"] = SUEP.pt
            out_vars.loc[indices_IRM, "SUEP_eta_IRM"] = SUEP.eta
            out_vars.loc[indices_IRM, "SUEP_phi_IRM"] = SUEP.phi
            out_vars.loc[indices_IRM, "SUEP_mass_IRM"] = SUEP.mass

        #####################################################################################
        # ---- Cluster Method (CL)
        # In this method, we use the tracks that were already clustered into the SUEP jet
        # to be the SUEP jet. Variables such as sphericity are calculated using these.
        #####################################################################################

        # no cut needed, but still define new variables for this method
        # so we don't mix things up
        indices_CL = indices
        SUEP_cand_CL = SUEP_cand
        ISR_cand_CL = ISR_cand

        # boost into frame of SUEP
        boost_SUEP = ak.zip(
            {
                "px": SUEP_cand_CL.px * -1,
                "py": SUEP_cand_CL.py * -1,
                "pz": SUEP_cand_CL.pz * -1,
                "mass": SUEP_cand_CL.mass,
            },
            with_name="Momentum4D",
        )

        # SUEP tracks for this method are defined to be the ones from the cluster
        # that was picked to be the SUEP jet
        SUEP_tracks_b_CL = SUEP_cluster_tracks.boost_p4(boost_SUEP)

        # SUEP jet variables
        eigs = sphericity(self, SUEP_tracks_b_CL, 1.0)  # Set r=1.0 for IRC safe
        out_vars.loc[indices_CL, "SUEP_nconst_CL"] = ak.num(SUEP_tracks_b_CL)
        out_vars.loc[indices_CL, "SUEP_pt_avg_b_CL"] = ak.mean(
            SUEP_tracks_b_CL.pt, axis=-1
        )
        out_vars.loc[indices_CL, "SUEP_pt_mean_scaled_CL"] = ak.mean(
            SUEP_tracks_b_CL.pt, axis=-1
        ) / ak.max(SUEP_tracks_b_CL.pt, axis=-1)
        out_vars.loc[indices_CL, "SUEP_S1_CL"] = 1.5 * (eigs[:, 1] + eigs[:, 0])

        # unboost for these
        SUEP_tracks_CL = SUEP_tracks_b_CL.boost_p4(SUEP_cand_CL)
        out_vars.loc[indices_CL, "SUEP_pt_avg_CL"] = ak.mean(SUEP_tracks_CL.pt, axis=-1)
        deltaR = SUEP_tracks_CL.deltaR(SUEP_cand_CL)
        out_vars.loc[indices_CL, "SUEP_rho0_CL"] = rho(
            self, 0, SUEP_cand_CL, SUEP_tracks_CL, deltaR
        )
        out_vars.loc[indices_CL, "SUEP_rho1_CL"] = rho(
            self, 1, SUEP_cand_CL, SUEP_tracks_CL, deltaR
        )

        out_vars.loc[indices_CL, "SUEP_pt_CL"] = SUEP_cand_CL.pt
        out_vars.loc[indices_CL, "SUEP_eta_CL"] = SUEP_cand_CL.eta
        out_vars.loc[indices_CL, "SUEP_phi_CL"] = SUEP_cand_CL.phi
        out_vars.loc[indices_CL, "SUEP_mass_CL"] = SUEP_cand_CL.mass

        # inverted selection
        if True:

            boost_ISR = ak.zip(
                {
                    "px": ISR_cand_CL.px * -1,
                    "py": ISR_cand_CL.py * -1,
                    "pz": ISR_cand_CL.pz * -1,
                    "mass": ISR_cand_CL.mass,
                },
                with_name="Momentum4D",
            )
            ISR_tracks_b_CL = ISR_cluster_tracks.boost_p4(boost_ISR)

            oneISRtrackCut = ak.num(ISR_tracks_b_CL) > 1

            # output file if no events pass selections for ISR
            # avoids leaving this chunk without these columns
            if not any(oneIRMtrackCut):
                print("No events in Inverted CL Removal Method, oneISRtrackCut.")
                for c in columns_CL_ISR:
                    out_vars[c] = np.nan
            else:

                # remove events with only one track in ISR
                indices_CL = indices[oneISRtrackCut]
                ISR_tracks_b_CL = ISR_tracks_b[oneISRtrackCut]
                ISR_cand_CL = ISR_cand[oneISRtrackCut]

                # ISR jet variables
                eigs = sphericity(self, ISR_tracks_b_CL, 1.0)  # Set r=1.0 for IRC safe
                out_vars.loc[indices_CL, "ISR_nconst_CL"] = ak.num(ISR_tracks_b_CL)
                out_vars.loc[indices_CL, "ISR_pt_avg_b_CL"] = ak.mean(
                    ISR_tracks_b_CL.pt, axis=-1
                )
                out_vars.loc[indices_CL, "ISR_pt_mean_scaled_CL"] = ak.mean(
                    ISR_tracks_b_CL.pt, axis=-1
                ) / ak.max(ISR_tracks_b_CL.pt, axis=-1)
                out_vars.loc[indices_CL, "ISR_S1_CL"] = 1.5 * (eigs[:, 1] + eigs[:, 0])

                # unboost for these
                ISR_tracks_CL = ISR_tracks_b_CL.boost_p4(ISR_cand_CL)
                out_vars.loc[indices_CL, "ISR_pt_avg_CL"] = ak.mean(
                    ISR_tracks_CL.pt, axis=-1
                )
                deltaR = ISR_tracks_CL.deltaR(ISR_cand_CL)
                out_vars.loc[indices_CL, "ISR_rho0_CL"] = rho(
                    self, 0, ISR_cand_CL, ISR_tracks_CL, deltaR
                )
                out_vars.loc[indices_CL, "ISR_rho1_CL"] = rho(
                    self, 1, ISR_cand_CL, ISR_tracks_CL, deltaR
                )

                out_vars.loc[indices_CL, "ISR_pt_CL"] = ISR_cand_CL.pt
                out_vars.loc[indices_CL, "ISR_eta_CL"] = ISR_cand_CL.eta
                out_vars.loc[indices_CL, "ISR_phi_CL"] = ISR_cand_CL.phi
                out_vars.loc[indices_CL, "ISR_mass_CL"] = ISR_cand_CL.mass

        #####################################################################################
        # ---- Cone Method (CO)
        # In this method, all tracks outside a cone of abs(deltaR) of 1.6 (in lab frame)
        # are the SUEP tracks, those inside the cone are ISR tracks.
        #####################################################################################

        # SUEP tracks are all tracks outside a deltaR cone around ISR
        ISR_cand_CO = ISR_cand
        SUEP_tracks_CO = tracks[abs(tracks.deltaR(ISR_cand_CO)) > 1.6]
        ISR_tracks_CO = tracks[abs(tracks.deltaR(ISR_cand_CO)) <= 1.6]
        oneCOtrackCut = ak.num(SUEP_tracks_CO) > 1

        # output file if no events pass selections for CO
        # avoids leaving this chunk without these columns
        if not any(oneCOtrackCut):
            print("No events in Cone Method, oneCOtrackCut.")
            for c in columns_CO:
                out_vars[c] = np.nan
            if self.isMC:
                for c in columns_CO_ISR:
                    out_vars[c] = np.nan
        else:
            # remove the events left with one track
            SUEP_tracks_CO = SUEP_tracks_CO[oneCOtrackCut]
            ISR_tracks_CO = ISR_tracks_CO[oneCOtrackCut]
            tracks_CO = tracks[oneCOtrackCut]
            indices_CO = indices[oneCOtrackCut]

            SUEP_cand_CO = ak.zip(
                {
                    "px": ak.sum(SUEP_tracks_CO.px, axis=-1),
                    "py": ak.sum(SUEP_tracks_CO.py, axis=-1),
                    "pz": ak.sum(SUEP_tracks_CO.pz, axis=-1),
                    "energy": ak.sum(SUEP_tracks_CO.energy, axis=-1),
                },
                with_name="Momentum4D",
            )

            # boost into frame of SUEP
            boost_SUEP = ak.zip(
                {
                    "px": SUEP_cand_CO.px * -1,
                    "py": SUEP_cand_CO.py * -1,
                    "pz": SUEP_cand_CO.pz * -1,
                    "mass": SUEP_cand_CO.mass,
                },
                with_name="Momentum4D",
            )

            SUEP_tracks_b_CO = SUEP_tracks_CO.boost_p4(boost_SUEP)

            # SUEP jet variables
            eigs = sphericity(self, SUEP_tracks_b_CO, 1.0)  # Set r=1.0 for IRC safe
            out_vars.loc[indices_CO, "SUEP_nconst_CO"] = ak.num(SUEP_tracks_b_CO)
            out_vars.loc[indices_CO, "SUEP_pt_avg_b_CO"] = ak.mean(
                SUEP_tracks_b_CO.pt, axis=-1
            )
            out_vars.loc[indices_CO, "SUEP_pt_mean_scaled_CO"] = ak.mean(
                SUEP_tracks_b_CO.pt, axis=-1
            ) / ak.max(SUEP_tracks_b_CO.pt, axis=-1)
            out_vars.loc[indices_CO, "SUEP_S1_CO"] = 1.5 * (eigs[:, 1] + eigs[:, 0])

            # unboost for these
            SUEP_tracks_CO = SUEP_tracks_b_CO.boost_p4(SUEP_cand_CO)
            out_vars.loc[indices_CO, "SUEP_pt_avg_CO"] = ak.mean(
                SUEP_tracks_CO.pt, axis=-1
            )
            deltaR = SUEP_tracks_CO.deltaR(SUEP_cand_CO)
            out_vars.loc[indices_CO, "SUEP_rho0_CO"] = rho(
                self, 0, SUEP_cand_CO, SUEP_tracks_CO, deltaR
            )
            out_vars.loc[indices_CO, "SUEP_rho1_CO"] = rho(
                self, 1, SUEP_cand_CO, SUEP_tracks_CO, deltaR
            )

            out_vars.loc[indices_CO, "SUEP_pt_CO"] = SUEP_cand_CO.pt
            out_vars.loc[indices_CO, "SUEP_eta_CO"] = SUEP_cand_CO.eta
            out_vars.loc[indices_CO, "SUEP_phi_CO"] = SUEP_cand_CO.phi
            out_vars.loc[indices_CO, "SUEP_mass_CO"] = SUEP_cand_CO.mass

            # inverted selection
            if self.isMC:

                oneCOISRtrackCut = ak.num(ISR_tracks_CO) > 1

                # output file if no events pass selections for ISR
                # avoids leaving this chunk without these columns
                if not any(oneCOISRtrackCut):
                    print("No events in Inverted CO Removal Method, oneCOISRtrackCut.")
                    for c in columns_CO_ISR:
                        out_vars[c] = np.nan
                else:

                    # remove events with one ISR track
                    ISR_tracks_CO = ISR_tracks_CO[oneCOISRtrackCut]
                    indices_CO = indices[oneCOISRtrackCut]

                    ISR_cand_CO = ak.zip(
                        {
                            "px": ak.sum(ISR_tracks_CO.px, axis=-1),
                            "py": ak.sum(ISR_tracks_CO.py, axis=-1),
                            "pz": ak.sum(ISR_tracks_CO.pz, axis=-1),
                            "energy": ak.sum(ISR_tracks_CO.energy, axis=-1),
                        },
                        with_name="Momentum4D",
                    )

                    boost_ISR = ak.zip(
                        {
                            "px": ISR_cand_CO.px * -1,
                            "py": ISR_cand_CO.py * -1,
                            "pz": ISR_cand_CO.pz * -1,
                            "mass": ISR_cand_CO.mass,
                        },
                        with_name="Momentum4D",
                    )

                    ISR_tracks_b_CO = ISR_tracks_CO.boost_p4(boost_ISR)

                    # ISR jet variables
                    eigs = sphericity(
                        self, ISR_tracks_b_CO, 1.0
                    )  # Set r=1.0 for IRC safe
                    out_vars.loc[indices_CO, "ISR_nconst_CO"] = ak.num(ISR_tracks_b_CO)
                    out_vars.loc[indices_CO, "ISR_pt_avg_b_CO"] = ak.mean(
                        ISR_tracks_b_CO.pt, axis=-1
                    )
                    out_vars.loc[indices_CO, "ISR_pt_mean_scaled_CO"] = ak.mean(
                        ISR_tracks_b_CO.pt, axis=-1
                    ) / ak.max(ISR_tracks_b_CO.pt, axis=-1)
                    out_vars.loc[indices_CO, "ISR_S1_CO"] = 1.5 * (
                        eigs[:, 1] + eigs[:, 0]
                    )

                    # unboost for these
                    ISR_tracks_CO = ISR_tracks_b_CO.boost_p4(ISR_cand_CO)
                    out_vars.loc[indices_CO, "ISR_pt_avg_CO"] = ak.mean(
                        ISR_tracks_CO.pt, axis=-1
                    )
                    deltaR = ISR_tracks_CO.deltaR(ISR_cand_CO)
                    out_vars.loc[indices_CO, "ISR_rho0_CO"] = rho(
                        self, 0, ISR_cand_CO, ISR_tracks_CO, deltaR
                    )
                    out_vars.loc[indices_CO, "ISR_rho1_CO"] = rho(
                        self, 1, ISR_cand_CO, ISR_tracks_CO, deltaR
                    )

                    out_vars.loc[indices_CO, "ISR_pt_CO"] = ISR_cand_CO.pt
                    out_vars.loc[indices_CO, "ISR_eta_CO"] = ISR_cand_CO.eta
                    out_vars.loc[indices_CO, "ISR_phi_CO"] = ISR_cand_CO.phi
                    out_vars.loc[indices_CO, "ISR_mass_CO"] = ISR_cand_CO.mass

        #####################################################################################
        # ---- Save outputs
        #####################################################################################

        save_dfs(
            self,
            [out_vars],
            ["vars"],
            events.behavior["__events_factory__"]._partition_key.replace("/", "_")
            + ".hdf5",
        )
        return output

    def postprocess(self, accumulator):
        return accumulator
