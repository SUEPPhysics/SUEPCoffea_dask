"""
SUEP_coffea_WH.py
Coffea producer for SUEP WH analysis. Uses fastjet package to recluster large jets:
https://github.com/scikit-hep/fastjet
Pietro Lugato, Chad Freer, Luca Lavezzo, Joey Reichert 2023
"""

import warnings

import awkward as ak
import numpy as np
import pandas as pd
import vector
from coffea import processor

warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

# Importing SUEP specific functions
import workflows.SUEP_utils as SUEP_utils
import workflows.WH_utils as WH_utils

# Importing CMS corrections
from workflows.CMS_corrections.btag_utils import btagcuts, doBTagWeights, getBTagEffs
from workflows.CMS_corrections.golden_jsons_utils import applyGoldenJSON
from workflows.CMS_corrections.HEM_utils import jetHEMFilter
from workflows.CMS_corrections.jetmet_utils import apply_jecs
from workflows.CMS_corrections.PartonShower_utils import GetPSWeights
from workflows.CMS_corrections.Prefire_utils import GetPrefireWeights
from workflows.CMS_corrections.track_killing_utils import track_killing

# IO utils
from workflows.utils.pandas_accumulator import pandas_accumulator

# Set vector behavior
vector.register_awkward()


class SUEP_cluster_WH(processor.ProcessorABC):
    def __init__(
        self,
        isMC: int,
        era: str,
        sample: str,
        do_syst: bool,
        flag: bool,
        output_location=None,
    ) -> None:
        self._flag = flag
        self.do_syst = do_syst
        self.era = str(era).lower()
        self.isMC = isMC
        self.sample = sample
        self.output_location = output_location
        self.scouting = 0

    def HighestPTMethod(
        self,
        indices,
        events,
        output,
        out_label=None,
    ):

        #####################################################################################
        # ---- Track selection
        # Prepare the clean PFCand matched to tracks collection, imposing a dR > 0.4
        # cut on tracks from the selected lepton.
        #####################################################################################

        tracks, _ = WH_utils.getTracks(events, lepton=self.lepton, leptonIsolation=0.4)
        if self.isMC and "track_down" in out_label:
            tracks = track_killing(self, tracks)

        # save tracks variables
        output["vars"].loc(indices, "ntracks" + out_label, ak.num(tracks).to_list())
        deltaPhi_tracks_W = np.abs(tracks.deltaphi(self.W))
        output["vars"].loc(
            indices,
            "ntracks_dPhiW0p2" + out_label,
            ak.num(deltaPhi_tracks_W[deltaPhi_tracks_W < 0.2], axis=1),
        )
        output["vars"].loc(
            indices,
            "trackspt_dPhiW0p2" + out_label,
            ak.sum(tracks.pt[deltaPhi_tracks_W < 0.2], axis=1),
        )
        deltaPhi_tracks_MET = np.abs(tracks.deltaphi(self.MET))
        output["vars"].loc(
            indices,
            "ntracks_dPhiMET0p2" + out_label,
            ak.num(deltaPhi_tracks_MET[deltaPhi_tracks_MET < 0.2], axis=1),
        )
        output["vars"].loc(
            indices,
            "trackspt_dPhiMET0p2" + out_label,
            ak.sum(tracks.pt[deltaPhi_tracks_MET < 0.2], axis=1),
        )

        #####################################################################################
        # ---- FastJet reclustering
        # The jet clustering part.
        #####################################################################################

        # make the ak15 clusters
        ak15jets, clusters = SUEP_utils.FastJetReclustering(tracks, r=1.5, minPt=60)

        # same some variables before making any selections on the ak15 clusters
        output["vars"].loc(
            indices, "ngood_fastjets" + out_label, ak.num(ak15jets).to_list()
        )

        #####################################################################################
        # ---- Highest pT Jet (PT)
        # SUEP defined as the highest pT jet. Cut on at least one ak15 cluster, and
        # SUEP candidate having at least 2 tracks.
        #####################################################################################

        # remove events with less than 1 cluster (i.e. need at least SUEP candidate cluster)
        clusterCut = ak.num(ak15jets, axis=1) > 0
        clusters = clusters[clusterCut]
        ak15jets = ak15jets[clusterCut]
        tracks = tracks[clusterCut]
        indices = indices[clusterCut]
        events = events[clusterCut]
        output["cutflow_oneCluster" + out_label] += ak.sum(events.genWeight)

        # output file if no events pass selections, avoids errors later on
        if len(tracks) == 0:
            print("No events pass clusterCut.")
            return

        # choose highest pT jet
        highpt_jet = ak.argsort(ak15jets.pt, axis=1, ascending=False, stable=True)
        ak15jets_pTsorted = ak15jets[highpt_jet]
        clusters_pTsorted = clusters[highpt_jet]
        SUEP_cand = ak15jets_pTsorted[:, 0]
        SUEP_cand_constituents = clusters_pTsorted[:, 0]
        other_AK15 = ak15jets_pTsorted[:, 1:]
        other_AK15_constituents = clusters_pTsorted[:, 1:]

        # at least 2 tracks
        singleTrackCut = ak.num(SUEP_cand_constituents) > 1
        SUEP_cand = SUEP_cand[singleTrackCut]
        SUEP_cand_constituents = SUEP_cand_constituents[singleTrackCut]
        tracks = tracks[singleTrackCut]
        indices = indices[singleTrackCut]
        events = events[singleTrackCut]
        other_AK15 = other_AK15[singleTrackCut]
        other_AK15_constituents = other_AK15_constituents[singleTrackCut]
        output["cutflow_twoTracksInCluster" + out_label] += ak.sum(events.genWeight)

        # output file if no events pass selections, avoids errors later on
        if len(indices) == 0:
            print("No events pass singleTrackCut.")
            return

        ######################################################################################
        # ---- SUEP kinematics
        # Store SUEP kinematics
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

        # SUEP tracks for this method are defined to be the ones from the cluster
        # that was picked to be the SUEP jet
        SUEP_cand_constituents_b = SUEP_cand_constituents.boost_p4(
            boost_SUEP
        )  ### boost the SUEP tracks to their restframe

        # SUEP candidate rest frame
        eigs = SUEP_utils.sphericity(
            SUEP_cand_constituents_b, 1.0
        )  # Set r=1.0 for IRC safe
        output["vars"].loc(
            indices,
            "SUEP_nconst_HighestPT" + out_label,
            ak.num(SUEP_cand_constituents_b),
        )
        output["vars"].loc(
            indices,
            "SUEP_pt_avg_b_HighestPT" + out_label,
            ak.mean(SUEP_cand_constituents_b.pt, axis=-1),
        )
        output["vars"].loc(
            indices, "SUEP_S1_HighestPT" + out_label, 1.5 * (eigs[:, 1] + eigs[:, 0])
        )

        # lab frame SUEP kinematics
        output["vars"].loc(
            indices,
            "SUEP_pt_avg_HighestPT" + out_label,
            ak.mean(SUEP_cand_constituents.pt, axis=-1),
        )
        output["vars"].loc(
            indices,
            "SUEP_highestPTtrack_HighestPT" + out_label,
            ak.max(SUEP_cand_constituents.pt, axis=-1),
        )
        output["vars"].loc(indices, "SUEP_pt_HighestPT" + out_label, SUEP_cand.pt)
        output["vars"].loc(indices, "SUEP_eta_HighestPT" + out_label, SUEP_cand.eta)
        output["vars"].loc(indices, "SUEP_phi_HighestPT" + out_label, SUEP_cand.phi)
        output["vars"].loc(indices, "SUEP_mass_HighestPT" + out_label, SUEP_cand.mass)

        # JEC corrected ak4jets inside SUEP cluster
        dR_ak4_SUEP = self.jets_jec[indices].deltaR(
            SUEP_cand
        )  # delta R between jets (selecting events that pass the HighestPT selections) and the SUEP cluster
        ak4jets_inSUEPcluster = self.jets_jec[indices][dR_ak4_SUEP < 1.5]
        output["vars"].loc(
            indices,
            "ak4jets_inSUEPcluster_n_HighestPT",
            ak.num(ak4jets_inSUEPcluster, axis=1),
        )
        output["vars"].loc(
            indices,
            "ak4jets_inSUEPcluster_pt_HighestPT",
            ak.sum(ak4jets_inSUEPcluster.pt, axis=1),
        )
        ak4jets_inSUEPcluster_ptargsort = ak.argsort(
            ak4jets_inSUEPcluster.pt, axis=1, ascending=False, stable=True
        )  # sort by pt to save some of these jets
        ak4jets_inSUEPcluster_ptsort = ak4jets_inSUEPcluster[
            ak4jets_inSUEPcluster_ptargsort
        ]
        for i in range(2):
            output["vars"].loc(
                indices,
                "ak4jet" + str(i + 1) + "_inSUEPcluster_pt_HighestPT",
                ak.fill_none(
                    ak.pad_none(
                        ak4jets_inSUEPcluster_ptsort.pt, i + 1, axis=1, clip=True
                    ),
                    -999,
                )[:, i],
            )
            output["vars"].loc(
                indices,
                "ak4jet" + str(i + 1) + "_inSUEPcluster_phi_HighestPT",
                ak.fill_none(
                    ak.pad_none(
                        ak4jets_inSUEPcluster_ptsort.phi, i + 1, axis=1, clip=True
                    ),
                    -999,
                )[:, i],
            )
            output["vars"].loc(
                indices,
                "ak4jet" + str(i + 1) + "_inSUEPcluster_eta_HighestPT",
                ak.fill_none(
                    ak.pad_none(
                        ak4jets_inSUEPcluster_ptsort.eta, i + 1, axis=1, clip=True
                    ),
                    -999,
                )[:, i],
            )
            output["vars"].loc(
                indices,
                "ak4jet" + str(i + 1) + "_inSUEPcluster_mass_HighestPT",
                ak.fill_none(
                    ak.pad_none(
                        ak4jets_inSUEPcluster_ptsort.mass, i + 1, axis=1, clip=True
                    ),
                    -999,
                )[:, i],
            )

        # leading ak4jet tracks
        leadingAK4_phi = ak.fill_none(
            ak.pad_none(ak4jets_inSUEPcluster_ptsort.phi, i + 1, axis=1, clip=True),
            -999,
        )[:, 0]
        leadingAK4_eta = ak.fill_none(
            ak.pad_none(ak4jets_inSUEPcluster_ptsort.eta, i + 1, axis=1, clip=True),
            -999,
        )[:, 0]
        dR_tracks_leadingAK4 = tracks.deltaR(
            ak.zip(
                {
                    "phi": leadingAK4_phi,
                    "eta": leadingAK4_eta,
                    "pt": ak.ones_like(leadingAK4_eta),
                },
                with_name="Momentum4D",
            )
        )
        leadingAK4_tracks = tracks[dR_tracks_leadingAK4 < 0.4]
        output["vars"].loc(
            indices,
            "leadingAK4_inSUEPcluster_ntracks_HighestPT",
            ak.num(leadingAK4_tracks, axis=1),
        )
        output["vars"].loc(
            indices,
            "leadingAK4_inSUEPcluster_scalarpt_HighestPT",
            ak.sum(leadingAK4_tracks.pt, axis=-1),
        )
        output["vars"].loc(
            indices,
            "leadingAK4_inSUEPcluster_vectorpt_HighestPT",
            (
                ak.sum(leadingAK4_tracks.px, axis=-1) ** 2
                + ak.sum(leadingAK4_tracks.py, axis=-1) ** 2
            )
            ** 0.5,
        )

        # jets outside of the SUEP cluster
        output["vars"].loc(
            indices,
            "minDeltaR_ak4jets_outsideSUEPcluster_HighestPT",
            ak.fill_none(ak.min(np.abs(dR_ak4_SUEP[dR_ak4_SUEP > 1.5]), axis=1), -999),
        )
        output["vars"].loc(
            indices,
            "maxDeltaR_ak4jets_outsideSUEPcluster_HighestPT",
            ak.fill_none(ak.max(np.abs(dR_ak4_SUEP[dR_ak4_SUEP > 1.5]), axis=1), -999),
        )

        # select tracks outside the AK15 SUEP cluster
        tracks_outside_SUEP = tracks[tracks.deltaR(SUEP_cand) > 1.5]
        twoTracksOutsideSUEP = ak.num(tracks_outside_SUEP) > 1
        tracks_outside_SUEP = tracks_outside_SUEP[twoTracksOutsideSUEP]
        nonSUEP_eigs = SUEP_utils.sphericity(tracks_outside_SUEP, 1.0)
        output["vars"].loc(
            indices[twoTracksOutsideSUEP], "nonSUEP_eig0_HighestPT", nonSUEP_eigs[:, 0]
        )
        output["vars"].loc(
            indices[twoTracksOutsideSUEP], "nonSUEP_eig1_HighestPT", nonSUEP_eigs[:, 1]
        )
        output["vars"].loc(
            indices[twoTracksOutsideSUEP], "nonSUEP_eig2_HighestPT", nonSUEP_eigs[:, 2]
        )

        # other AK15 jets info
        output["vars"].loc(
            indices, "otherAK15_pt_HighestPT", ak.sum(other_AK15.pt, axis=1)
        )
        other_AK15_nconst = ak.num(other_AK15_constituents, axis=-1)
        mostNumerousAK15 = other_AK15[
            ak.argmax(other_AK15_nconst, axis=-1, keepdims=True)
        ]
        output["vars"].loc(
            indices,
            "otherAK15_maxConst_pt_HighestPT",
            mostNumerousAK15.pt.to_numpy(allow_missing=True),
        )
        output["vars"].loc(
            indices,
            "otherAK15_maxConst_eta_HighestPT",
            mostNumerousAK15.eta.to_numpy(allow_missing=True),
        )
        output["vars"].loc(
            indices,
            "otherAK15_maxConst_phi_HighestPT",
            mostNumerousAK15.phi.to_numpy(allow_missing=True),
        )
        output["vars"].loc(
            indices,
            "otherAK15_maxConst_nconst_HighestPT",
            ak.max(other_AK15_nconst, axis=-1).to_numpy(allow_missing=True),
        )

        # WH system
        WH_system = SUEP_cand + self.W[indices]
        output["vars"].loc(indices, "WH_system_mass_HighestPT", WH_system.mass)
        output["vars"].loc(indices, "WH_system_pt_HighestPT", WH_system.pt)
        output["vars"].loc(indices, "WH_system_phi_HighestPT", WH_system.phi)
        WH_system_PuppiMET = SUEP_cand + self.W_PuppiMET[indices]
        output["vars"].loc(
            indices, "WH_system_PuppiMET_mass_HighestPT", WH_system_PuppiMET.mass
        )
        output["vars"].loc(
            indices, "WH_system_PuppiMET_pt_HighestPT", WH_system_PuppiMET.pt
        )
        output["vars"].loc(
            indices, "WH_system_PuppiMET_phi_HighestPT", WH_system_PuppiMET.phi
        )

    def storeEventVars(
        self,
        events,
        output,
    ):
        """
        Store event variables in the output dictionary.
        """

        # general event vars
        if self.isMC:
            output["vars"]["genweight"] = events.genWeight.to_list()
            if "GenModel" in dir(
                events
            ):  # SUEP central samples have different genModels in each file
                output["vars"]["genModel"] = WH_utils.getGenModel(events)

        output["vars"]["event"] = events.event.to_list()
        output["vars"]["run"] = events.run
        output["vars"]["luminosityBlock"] = events.luminosityBlock
        output["vars"]["PV_npvs"] = events.PV.npvs
        output["vars"]["PV_npvsGood"] = events.PV.npvsGood

        if self.isMC:
            output["vars"]["LHE_Vpt"] = events.LHE.Vpt
            output["vars"]["LHE_HT"] = events.LHE.HT

        # select out ak4jets
        uncorrected_ak4jets = WH_utils.getAK4Jets(events.Jet, isMC=self.isMC)
        jets_c, met_c = apply_jecs(
            self,
            Sample=self.sample,
            events=events,
            prefix="",
        )
        jet_HEM_Cut, _ = jetHEMFilter(self, jets_c, events.run)
        jets_c = jets_c[jet_HEM_Cut]
        self.jets_jec = WH_utils.getAK4Jets(jets_c, self.lepton, self.isMC)
        output["vars"]["ngood_ak4jets"] = ak.num(self.jets_jec).to_list()

        # ht
        output["vars"]["ht"] = ak.sum(uncorrected_ak4jets.pt, axis=-1).to_list()
        output["vars"]["ht_JEC"] = ak.sum(self.jets_jec.pt, axis=-1).to_list()
        if self.isMC and self.do_syst:
            jets_jec_JERUp = WH_utils.getAK4Jets(
                jets_c["JER"].up, self.lepton, self.isMC
            )
            jets_jec_JERDown = WH_utils.getAK4Jets(
                jets_c["JER"].down, self.lepton, self.isMC
            )
            jets_jec_JESUp = WH_utils.getAK4Jets(
                jets_c["JES_jes"].up, self.lepton, self.isMC
            )
            jets_jec_JESDown = WH_utils.getAK4Jets(
                jets_c["JES_jes"].down, self.lepton, self.isMC
            )

            output["vars"]["ht_JEC" + "_JER_up"] = ak.sum(
                jets_jec_JERUp.pt, axis=-1
            ).to_list()
            output["vars"]["ht_JEC" + "_JER_down"] = ak.sum(
                jets_jec_JERDown.pt, axis=-1
            ).to_list()
            output["vars"]["ht_JEC" + "_JES_up"] = ak.sum(
                jets_jec_JESUp.pt, axis=-1
            ).to_list()
            output["vars"]["ht_JEC" + "_JES_down"] = ak.sum(
                jets_jec_JESDown.pt, axis=-1
            ).to_list()

        # saving number of bjets for different definitions (higher or lower requirements on b-likeliness) - see btag_utils.py
        # btag function requests eras as integers (used again for btag weights)
        if self.era == "2016apv":
            era_int = 2015
        else:
            era_int = int(self.era)

        output["vars"]["nBLoose"] = ak.sum(
            (self.jets_jec.btag >= btagcuts("Loose", era_int)), axis=1
        )[:]
        output["vars"]["nBMedium"] = ak.sum(
            (self.jets_jec.btag >= btagcuts("Medium", era_int)), axis=1
        )[:]
        output["vars"]["nBTight"] = ak.sum(
            (self.jets_jec.btag >= btagcuts("Tight", era_int)), axis=1
        )[:]

        # saving kinematic variables for three leading pT jets
        highpt_jet = ak.argsort(self.jets_jec.pt, axis=1, ascending=False, stable=True)
        jets_pTsorted = self.jets_jec[highpt_jet]
        for i in range(3):
            output["vars"]["jet" + str(i + 1) + "_pt"] = ak.fill_none(
                ak.pad_none(jets_pTsorted.pt, i + 1, axis=1, clip=True), -999
            )[:, i]
            output["vars"]["jet" + str(i + 1) + "_phi"] = ak.fill_none(
                ak.pad_none(jets_pTsorted.phi, i + 1, axis=1, clip=True), -999
            )[:, i]
            output["vars"]["jet" + str(i + 1) + "_eta"] = ak.fill_none(
                ak.pad_none(jets_pTsorted.eta, i + 1, axis=1, clip=True), -999
            )[:, i]
            output["vars"]["jet" + str(i + 1) + "_qgl"] = ak.fill_none(
                ak.pad_none(jets_pTsorted.qgl, i + 1, axis=1, clip=True), -999
            )[:, i]
            output["vars"]["jet" + str(i + 1) + "_mass"] = ak.fill_none(
                ak.pad_none(jets_pTsorted.mass, i + 1, axis=1, clip=True), -999
            )[:, i]

        # saving kinematic variables for the leading b-tagged jet
        highbtag_jet = ak.argsort(
            self.jets_jec.btag, axis=1, ascending=False, stable=True
        )
        jets_btag_sorted = self.jets_jec[highbtag_jet]
        output["vars"]["bjet_pt"] = ak.fill_none(
            ak.pad_none(jets_btag_sorted.pt, 1, axis=1, clip=True), -999
        )[:, 0]
        output["vars"]["bjet_phi"] = ak.fill_none(
            ak.pad_none(jets_btag_sorted.phi, 1, axis=1, clip=True), -999
        )[:, 0]
        output["vars"]["bjet_eta"] = ak.fill_none(
            ak.pad_none(jets_btag_sorted.eta, 1, axis=1, clip=True), -999
        )[:, 0]
        output["vars"]["bjet_qgl"] = ak.fill_none(
            ak.pad_none(jets_pTsorted.qgl, 1, axis=1, clip=True), -999
        )[:, 0]
        output["vars"]["bjet_btag"] = ak.fill_none(
            ak.pad_none(jets_pTsorted.btag, 1, axis=1, clip=True), -999
        )[:, 0]

        # save MET
        self.MET = WH_utils.make_MET_4v(events.MET)

        # saving kinematic variables for the deltaphi(min(jet,MET)) jet
        self.jets_jec.deltaPhiMET = WH_utils.MET_delta_phi(self.jets_jec, events.MET)
        sorted_deltaphiMET_jets = self.jets_jec[
            ak.argsort(self.jets_jec.deltaPhiMET, axis=1, ascending=True)
        ]
        output["vars"]["minDeltaPhiMETJet_pt"] = ak.fill_none(
            ak.pad_none(sorted_deltaphiMET_jets.pt, 1, axis=1, clip=True), -999
        )[:, 0]
        output["vars"]["minDeltaPhiMETJet_phi"] = ak.fill_none(
            ak.pad_none(sorted_deltaphiMET_jets.phi, 1, axis=1, clip=True), -999
        )[:, 0]
        output["vars"]["minDeltaPhiMETJet_eta"] = ak.fill_none(
            ak.pad_none(sorted_deltaphiMET_jets.eta, 1, axis=1, clip=True), -999
        )[:, 0]
        output["vars"]["minDeltaPhiMETJet_qgl"] = ak.fill_none(
            ak.pad_none(sorted_deltaphiMET_jets.qgl, 1, axis=1, clip=True), -999
        )[:, 0]

        # saving MET variables
        output["vars"]["CaloMET_pt"] = events.CaloMET.pt
        output["vars"]["CaloMET_phi"] = events.CaloMET.phi
        output["vars"]["CaloMET_sumEt"] = events.CaloMET.sumEt
        output["vars"]["PuppiMET_pt"] = events.PuppiMET.pt
        output["vars"]["PuppiMET_phi"] = events.PuppiMET.phi
        output["vars"]["PuppiMET_sumEt"] = events.PuppiMET.sumEt
        output["vars"]["MET_pt"] = events.MET.pt
        output["vars"]["MET_phi"] = events.MET.phi
        output["vars"]["MET_sumEt"] = events.MET.sumEt

        # deprecated
        """
        output["vars"]["ChsMET_pt"] = events.ChsMET.pt
        output["vars"]["ChsMET_phi"] = events.ChsMET.phi
        output["vars"]["ChsMET_sumEt"] = events.ChsMET.sumEt
        output["vars"]["TkMET_pt"] = events.TkMET.pt
        output["vars"]["TkMET_phi"] = events.TkMET.phi
        output["vars"]["TkMET_sumEt"] = events.TkMET.sumEt
        output["vars"]["RawMET_pt"] = events.RawMET.pt
        output["vars"]["RawMET_phi"] = events.RawMET.phi
        output["vars"]["RawMET_sumEt"] = events.RawMET.sumEt
        output["vars"]["RawPuppiMET_pt"] = events.RawPuppiMET.pt
        output["vars"]["RawPuppiMET_phi"] = events.RawPuppiMET.phi
        output["vars"]["MET_JEC_pt"] = met_c.pt
        output["vars"]["MET_JEC_sumEt"] = met_c.sumEt
        """

        # corrections on MET
        if self.isMC and self.do_syst:

            output["vars"]["PuppiMET_pt_JER_up"] = events.PuppiMET.ptJERUp
            output["vars"]["PuppiMET_pt_JER_down"] = events.PuppiMET.ptJERDown
            output["vars"]["PuppiMET_pt_JES_up"] = events.PuppiMET.ptJESUp
            output["vars"]["PuppiMET_pt_JES_down"] = events.PuppiMET.ptJESDown
            output["vars"]["PuppiMET_phi_JER_up"] = events.PuppiMET.phiJERUp
            output["vars"]["PuppiMET_phi_JER_down"] = events.PuppiMET.phiJERDown
            output["vars"]["PuppiMET_phi_JES_up"] = events.PuppiMET.phiJESUp
            output["vars"]["PuppiMET_phi_JES_down"] = events.PuppiMET.phiJESDown
            output["vars"]["MET_JEC_pt_JER_up"] = met_c.JER.up.pt
            output["vars"]["MET_JEC_pt_JER_down"] = met_c.JER.up.pt
            output["vars"]["MET_JEC_pt_JES_up"] = met_c.JES_jes.up.pt
            output["vars"]["MET_JEC_pt_JES_down"] = met_c.JES_jes.down.pt
            output["vars"][
                "MET_JEC_pt_UnclusteredEnergy_up"
            ] = met_c.MET_UnclusteredEnergy.up.pt
            output["vars"][
                "MET_JEC_pt_UnclusteredEnergy_down"
            ] = met_c.MET_UnclusteredEnergy.down.pt
            output["vars"]["MET_JEC_phi"] = met_c.phi
            output["vars"]["MET_JEC_phi_JER_up"] = met_c.JER.up.phi
            output["vars"]["MET_JEC_phi_JER_down"] = met_c.JER.down.phi
            output["vars"]["MET_JEC_phi_JES_up"] = met_c.JES_jes.up.phi
            output["vars"]["MET_JEC_phi_JES_down"] = met_c.JES_jes.down.phi
            output["vars"][
                "MET_JEC_phi_UnclusteredEnergy_up"
            ] = met_c.MET_UnclusteredEnergy.up.phi
            output["vars"][
                "MET_JEC_phi_UnclusteredEnergy_down"
            ] = met_c.MET_UnclusteredEnergy.down.phi

        if self.isMC:
            output["vars"]["Pileup_nTrueInt"] = events.Pileup.nTrueInt
            psweights = GetPSWeights(self, events)  # Parton Shower weights
            if len(psweights) == 4:
                output["vars"]["PSWeight_ISR_up"] = psweights[0]
                output["vars"]["PSWeight_ISR_down"] = psweights[1]
                output["vars"]["PSWeight_FSR_up"] = psweights[2]
                output["vars"]["PSWeight_FSR_down"] = psweights[3]
            else:
                output["vars"]["PSWeight"] = psweights

            bTagWeights = doBTagWeights(
                events, self.jets_jec, era_int, "L", do_syst=self.do_syst
            )  # Does not change selection
            output["vars"]["bTagWeight"] = bTagWeights["central"][:]  # BTag weights

            prefireweights = GetPrefireWeights(self, events)  # Prefire weights
            output["vars"]["prefire_nom"] = prefireweights[0]
            output["vars"]["prefire_up"] = prefireweights[1]
            output["vars"]["prefire_down"] = prefireweights[2]

        # get gen SUEP kinematics
        SUEP_genMass = ak.Array(len(events) * [0])
        SUEP_genPt = ak.Array(len(events) * [0])
        SUEP_genEta = ak.Array(len(events) * [0])
        SUEP_genPhi = ak.Array(len(events) * [0])
        darkphis = ak.Array(len(events) * [0])
        cleaned_darkphis = ak.Array(len(events) * [0])
        if self.isMC:
            genParts = WH_utils.getGenPart(events)
            genSUEP = genParts[(abs(genParts.pdgID) == 25)]

            # we need to grab the last SUEP in the chain for each event
            SUEP_genMass = [g[-1].mass if len(g) > 0 else 0 for g in genSUEP]
            SUEP_genPt = [g[-1].pt if len(g) > 0 else 0 for g in genSUEP]
            SUEP_genPhi = [g[-1].phi if len(g) > 0 else 0 for g in genSUEP]
            SUEP_genEta = [g[-1].eta if len(g) > 0 else 0 for g in genSUEP]

            # grab the daughters of the scalar
            darkphis = WH_utils.getGenDarkPseudoscalars(events)
            cleaned_darkphis = darkphis[abs(darkphis.eta) < 2.5]

        output["vars"]["SUEP_genMass"] = SUEP_genMass
        output["vars"]["SUEP_genPt"] = SUEP_genPt
        output["vars"]["SUEP_genEta"] = SUEP_genEta
        output["vars"]["SUEP_genPhi"] = SUEP_genPhi
        output["vars"]["n_darkphis"] = ak.num(darkphis, axis=-1)
        output["vars"]["n_darkphis_inTracker"] = ak.num(cleaned_darkphis, axis=-1)

        # saving tight lepton kinematics
        output["vars"]["lepton_pt"] = self.lepton.pt
        output["vars"]["lepton_eta"] = self.lepton.eta
        output["vars"]["lepton_phi"] = self.lepton.phi
        output["vars"]["lepton_mass"] = self.lepton.mass
        output["vars"]["lepton_flavor"] = self.lepton.pdgID
        output["vars"]["lepton_ID"] = self.lepton.ID
        output["vars"]["lepton_IDMVA"] = self.lepton.IDMVA
        output["vars"]["lepton_iso"] = self.lepton.iso
        output["vars"]["lepton_isoMVA"] = self.lepton.isoMVA
        output["vars"]["lepton_miniIso"] = self.lepton.miniIso
        output["vars"]["lepton_dxy"] = self.lepton.dxy
        output["vars"]["lepton_dz"] = self.lepton.dz

        # other loose leptons
        looseMuons, looseElectrons, looseLeptons = WH_utils.getLooseLeptons(events)
        self.looseLeptons = looseLeptons
        output["vars"]["nLooseLeptons"] = ak.num(looseLeptons).to_list()
        output["vars"]["nLooseMuons"] = ak.num(looseMuons).to_list()
        output["vars"]["nLooseElectrons"] = ak.num(looseElectrons).to_list()
        highpt_leptons = ak.argsort(
            looseLeptons.pt, axis=1, ascending=False, stable=True
        )
        looseLeptons_pTsorted = looseLeptons[highpt_leptons]
        for i in range(3):
            output["vars"]["looseLepton" + str(i + 1) + "_pt"] = ak.fill_none(
                ak.pad_none(looseLeptons_pTsorted.pt, i + 1, axis=1, clip=True), -999
            )[:, i]
            output["vars"]["looseLepton" + str(i + 1) + "_phi"] = ak.fill_none(
                ak.pad_none(looseLeptons_pTsorted.phi, i + 1, axis=1, clip=True), -999
            )[:, i]
            output["vars"]["looseLepton" + str(i + 1) + "_eta"] = ak.fill_none(
                ak.pad_none(looseLeptons_pTsorted.eta, i + 1, axis=1, clip=True), -999
            )[:, i]
            output["vars"]["looseLepton" + str(i + 1) + "_flavor"] = ak.fill_none(
                ak.pad_none(looseLeptons_pTsorted.pdgID, i + 1, axis=1, clip=True), -999
            )[:, i]

        # saving W information
        self.W = WH_utils.make_Wt_4v(self.lepton, events.MET)
        self.W_PuppiMET = WH_utils.make_Wt_4v(self.lepton, events.PuppiMET)
        self.W_CaloMET = WH_utils.make_Wt_4v(self.lepton, events.CaloMET)
        output["vars"]["W_pt"] = self.W.pt
        output["vars"]["W_phi"] = self.W.phi
        output["vars"]["W_mt"] = WH_utils.calc_W_mt(self.lepton, events.MET)
        output["vars"]["W_pt_PuppiMET"] = self.W_PuppiMET.pt
        output["vars"]["W_phi_PuppiMET"] = self.W_PuppiMET.phi
        output["vars"]["W_mt_PuppiMET"] = WH_utils.calc_W_mt(
            self.lepton, events.PuppiMET
        )
        output["vars"]["W_pt_CaloMET"] = self.W_CaloMET.pt
        output["vars"]["W_phi_CaloMET"] = self.W_CaloMET.phi
        output["vars"]["W_mt_CaloMET"] = WH_utils.calc_W_mt(self.lepton, events.CaloMET)

        # save genW for MC
        if self.isMC:
            genW = WH_utils.getGenW(events)
            output["vars"]["genW_pt"] = ak.fill_none(
                ak.pad_none(genW.pt, 1, axis=1, clip=True), -999
            )[:, 0]
            output["vars"]["genW_phi"] = ak.fill_none(
                ak.pad_none(genW.phi, 1, axis=1, clip=True), -999
            )[:, 0]
            output["vars"]["genW_eta"] = ak.fill_none(
                ak.pad_none(genW.eta, 1, axis=1, clip=True), -999
            )[:, 0]
            output["vars"]["genW_mass"] = ak.fill_none(
                ak.pad_none(genW.mass, 1, axis=1, clip=True), -999
            )[:, 0]

        # photon information
        photons = WH_utils.getPhotons(events, self.isMC)
        output["vars"]["nphotons"] = ak.num(photons).to_list()

        # saving min, max delta R, phi, eta between jets
        jet_combinations = ak.combinations(
            self.jets_jec, 2, fields=["jet1", "jet2"], axis=-1
        )
        jet_combinations_deltaR = np.abs(
            jet_combinations["jet1"].deltaR(jet_combinations["jet2"])
        )
        jet_combinations_deltaPhi = np.abs(
            jet_combinations["jet1"].deltaphi(jet_combinations["jet2"])
        )
        jet_combinations_deltaEta = np.abs(
            jet_combinations["jet1"].deltaeta(jet_combinations["jet2"])
        )
        output["vars"]["minDeltaRJets"] = ak.fill_none(
            ak.min(jet_combinations_deltaR, axis=-1), -999
        )
        output["vars"]["maxDeltaRJets"] = ak.fill_none(
            ak.max(jet_combinations_deltaR, axis=-1), -999
        )
        output["vars"]["minDeltaPhiJets"] = ak.fill_none(
            ak.min(jet_combinations_deltaPhi, axis=-1), -999
        )
        output["vars"]["maxDeltaPhiJets"] = ak.fill_none(
            ak.max(jet_combinations_deltaPhi, axis=-1), -999
        )
        output["vars"]["minDeltaEtaJets"] = ak.fill_none(
            ak.min(jet_combinations_deltaEta, axis=-1), -999
        )
        output["vars"]["maxDeltaEtaJets"] = ak.fill_none(
            ak.max(jet_combinations_deltaEta, axis=-1), -999
        )

        # saving min, max delta R, phi, eta between any jet and the tight lepton
        jet_lepton_combinations_deltaR = np.abs(self.jets_jec.deltaR(self.lepton))
        jet_lepton_combinations_deltaPhi = np.abs(self.jets_jec.deltaphi(self.lepton))
        jet_lepton_combinations_deltaEta = np.abs(self.jets_jec.deltaeta(self.lepton))
        output["vars"]["minDeltaRJetLepton"] = ak.fill_none(
            ak.min(jet_lepton_combinations_deltaR, axis=-1), -999
        )
        output["vars"]["maxDeltaRJetLepton"] = ak.fill_none(
            ak.max(jet_lepton_combinations_deltaR, axis=-1), -999
        )
        output["vars"]["minDeltaPhiJetLepton"] = ak.fill_none(
            ak.min(jet_lepton_combinations_deltaPhi, axis=-1), -999
        )
        output["vars"]["maxDeltaPhiJetLepton"] = ak.fill_none(
            ak.max(jet_lepton_combinations_deltaPhi, axis=-1), -999
        )
        output["vars"]["minDeltaEtaJetLepton"] = ak.fill_none(
            ak.min(jet_lepton_combinations_deltaEta, axis=-1), -999
        )
        output["vars"]["maxDeltaEtaJetLepton"] = ak.fill_none(
            ak.max(jet_lepton_combinations_deltaEta, axis=-1), -999
        )

        # saving min delta phi between any jet and the W
        jet_W_deltaPhi = np.abs(self.jets_jec.deltaphi(self.W))
        output["vars"]["minDeltaPhiJetW"] = ak.fill_none(
            ak.min(jet_W_deltaPhi, axis=-1), -999
        )

    def analysis(self, events, output, out_label=""):

        #####################################################################################
        # ---- Basic event selection
        # Define the events that we will use.
        # Apply triggers, golden JSON, quality filters, and orthogonality selections.
        #####################################################################################

        output["cutflow_total" + out_label] += ak.sum(events.genWeight)

        if self.isMC == 0:
            events = applyGoldenJSON(self, events)
            events.genWeight = np.ones(len(events))  # dummy value for data

        output["cutflow_goldenJSON" + out_label] += ak.sum(events.genWeight)

        events = WH_utils.genSelection(events, self.sample)
        output["cutflow_genCuts" + out_label] += ak.sum(events.genWeight)

        events = WH_utils.triggerSelection(
            events, self.sample, self.era, self.isMC, output, out_label
        )
        output["cutflow_allTriggers" + out_label] += ak.sum(events.genWeight)

        events = WH_utils.qualityFiltersSelection(events, self.era)
        output["cutflow_qualityFilters" + out_label] += ak.sum(events.genWeight)

        events = WH_utils.orthogonalitySelection(events)
        output["cutflow_orthogonality" + out_label] += ak.sum(events.genWeight)

        events = events[ak.num(WH_utils.getAK4Jets(events.Jet, isMC=self.isMC)) > 0]
        output["cutflow_oneAK4jet" + out_label] += ak.sum(events.genWeight)

        # output file if no events pass selections, avoids errors later on
        if len(events) == 0:
            print("No events passed basic event selection. Saving empty outputs.")
            return output

        #####################################################################################
        # ---- Lepton selection
        # Define the lepton objects and apply single lepton selection.
        #####################################################################################

        _, _, tightLeptons = WH_utils.getTightLeptons(events)

        # require exactly one tight lepton
        leptonSelection = ak.num(tightLeptons) == 1
        events = events[leptonSelection]
        tightLeptons = tightLeptons[leptonSelection]
        self.lepton = tightLeptons[:, 0]
        output["cutflow_oneLepton" + out_label] += ak.sum(events.genWeight)

        # output file if no events pass selections, avoids errors later on
        if len(events) == 0:
            print("No events pass oneLepton.")
            return output

        #####################################################################################
        # ---- Store event level information
        #####################################################################################

        # these only need to be saved once, as they shouldn't change even with track killing
        if out_label == "":
            self.storeEventVars(
                events,
                output=output,
            )

        #####################################################################################
        # ---- SUEP definition and analysis
        #####################################################################################

        # indices of events, used to keep track which events pass selections for each method
        # and only fill those rows of the DataFrame (e.g. track killing).
        # from now on, if any cuts are applied, the indices should be updated, and the df
        # should be filled with the updated indices.
        indices = np.arange(0, len(events))

        self.HighestPTMethod(
            indices,
            events,
            output=output,
            out_label=out_label,
        )

        return output

    def process(self, events):
        dataset = events.metadata["dataset"]

        output = processor.dict_accumulator(
            {
                "gensumweight": processor.value_accumulator(float, 0),
                "cutflow_total": processor.value_accumulator(float, 0),
                "cutflow_goldenJSON": processor.value_accumulator(float, 0),
                "cutflow_genCuts": processor.value_accumulator(float, 0),
                "cutflow_triggerSingleMuon": processor.value_accumulator(float, 0),
                "cutflow_triggerDoubleMuon": processor.value_accumulator(float, 0),
                "cutflow_triggerEGamma": processor.value_accumulator(float, 0),
                "cutflow_allTriggers": processor.value_accumulator(float, 0),
                "cutflow_orthogonality": processor.value_accumulator(float, 0),
                "cutflow_oneLepton": processor.value_accumulator(float, 0),
                "cutflow_qualityFilters": processor.value_accumulator(float, 0),
                "cutflow_oneAK4jet": processor.value_accumulator(float, 0),
                "cutflow_oneCluster": processor.value_accumulator(float, 0),
                "cutflow_twoTracksInCluster": processor.value_accumulator(float, 0),
                "vars": pandas_accumulator(pd.DataFrame()),
            }
        )

        # gen weights
        if self.isMC:
            output["gensumweight"] += ak.sum(events.genWeight)
        else:
            genWeight = np.ones(len(events))
            events = ak.with_field(events, genWeight, "genWeight")

        # run the analysis
        output = self.analysis(events, output)

        # run the analysis with the track systematics applied
        if self.isMC and self.do_syst:
            output.update(
                {
                    "cutflow_total_track_down": processor.value_accumulator(float, 0),
                    "cutflow_goldenJSON_track_down": processor.value_accumulator(
                        float, 0
                    ),
                    "cutflow_genCuts_track_down": processor.value_accumulator(float, 0),
                    "cutflow_triggerSingleMuon_track_down": processor.value_accumulator(
                        float, 0
                    ),
                    "cutflow_triggerDoubleMuon_track_down": processor.value_accumulator(
                        float, 0
                    ),
                    "cutflow_triggerEGamma_track_down": processor.value_accumulator(
                        float, 0
                    ),
                    "cutflow_allTriggers_track_down": processor.value_accumulator(
                        float, 0
                    ),
                    "cutflow_orthogonality_track_down": processor.value_accumulator(
                        float, 0
                    ),
                    "cutflow_oneLepton_track_down": processor.value_accumulator(
                        float, 0
                    ),
                    "cutflow_qualityFilters_track_down": processor.value_accumulator(
                        float, 0
                    ),
                    "cutflow_oneAK4jet_track_down": processor.value_accumulator(
                        float, 0
                    ),
                    "cutflow_oneCluster_track_down": processor.value_accumulator(
                        float, 0
                    ),
                    "cutflow_twoTracksInCluster_track_down": processor.value_accumulator(
                        float, 0
                    ),
                }
            )
            output = self.analysis(events, output, out_label="_track_down")

        return {dataset: output}

    def postprocess(self, accumulator):
        return accumulator
