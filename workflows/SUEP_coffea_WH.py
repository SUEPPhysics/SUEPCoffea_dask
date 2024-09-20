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
from hist import Hist

warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

# Importing SUEP specific functions
import workflows.SUEP_utils as SUEP_utils
import workflows.WH_utils as WH_utils

# Importing CMS corrections
from workflows.CMS_corrections.btag_utils import btagcuts, doBTagWeights, getBTagEffs
from workflows.CMS_corrections.golden_jsons_utils import applyGoldenJSON
from workflows.CMS_corrections.HEM_utils import jetHEMFilter, METHEMFilter
from workflows.CMS_corrections.PartonShower_utils import GetPSWeights
from workflows.CMS_corrections.Prefire_utils import GetPrefireWeights
from workflows.CMS_corrections.track_killing_utils import track_killing
from workflows.CMS_corrections.jetvetomap_utils import JetVetoMap
from workflows.CMS_corrections.jetmet_utils import applyJECStoJets

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
        CRQCD:bool=False,
        VRGJ:bool=False,
        dropNonMethodEvents:bool=False,
        storeJetsInfo:bool=False,
    ) -> None:
        self._flag = flag
        self.do_syst = do_syst
        self.era = str(era).lower()
        self.isMC = isMC
        self.sample = sample
        self.output_location = output_location
        self.scouting = 0
        self.CRQCD = CRQCD
        self.VRGJ = VRGJ
        self.dropNonMethodEvents = dropNonMethodEvents
        self.storeJetsInfo = storeJetsInfo # for the b-tag efficiencies

    def HighestPTMethod(
        self,
        events,
        output,
        out_label=None,
    ):

        # indices of events, used to keep track which events pass selections for each method
        # and only fill those rows of the DataFrame (e.g. track killing).
        # from now on, if any cuts are applied, the indices should be updated, and the df
        # should be filled with the updated indices.
        indices = np.arange(0, len(events))

        #####################################################################################
        # ---- Track selection
        # Prepare the clean PFCand matched to tracks collection, imposing a dR > 0.4
        # cut on tracks from the selected lepton.
        #####################################################################################

        tracks, pfcands, lost_tracks = WH_utils.getTracks(
            events,
            iso_object=events.WH_lepton if not self.VRGJ else events.WH_gamma,
            isolation_deltaR=0.4
        )
        if self.isMC and "track_down" in out_label:
            tracks = track_killing(self, tracks)
        events = ak.with_field(events, tracks, "WH_tracks")

        # save tracks variables
        output["vars"].loc(indices, "ntracks" + out_label, ak.num(events.WH_tracks).to_list())
        output["vars"].loc(indices, "npfcands" + out_label, ak.num(pfcands).to_list())
        output["vars"].loc(indices, "nlosttracks" + out_label, ak.num(lost_tracks).to_list())

        #####################################################################################
        # ---- FastJet reclustering
        # The jet clustering part.
        #####################################################################################

        # make the ak15 clusters
        ak15jets, clusters = SUEP_utils.FastJetReclustering(events.WH_tracks, r=1.5, minPt=60)
        events = ak.with_field(events, ak15jets, "WH_ak15jets")
        events = ak.with_field(events, clusters, "WH_ak15clusters")

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
        clusterCut = ak.num(events.WH_ak15clusters, axis=1) > 0
        indices = indices[clusterCut]
        events = events[clusterCut]
        output["cutflow_oneCluster" + out_label] += ak.sum(events.genWeight)

        # output file if no events pass selections, avoids errors later on
        if len(events) == 0:
            print("\n\nNo events pass clusterCut.\n\n")
            return

        # choose highest pT jet
        highpt_jet = ak.argsort(events.WH_ak15jets.pt, axis=1, ascending=False, stable=True)
        ak15jets_pTsorted = events.WH_ak15jets[highpt_jet]
        clusters_pTsorted = events.WH_ak15clusters[highpt_jet]
        events = ak.with_field(events, ak15jets_pTsorted[:, 0], "WH_SUEP_cand")
        events = ak.with_field(events, clusters_pTsorted[:, 0], "WH_SUEP_cand_constituents")
        events = ak.with_field(events, ak15jets_pTsorted[:, 1:], "WH_other_AK15")
        events = ak.with_field(events, clusters_pTsorted[:, 1:], "WH_other_AK15_constituents")

        # at least 2 tracks
        singleTrackCut = ak.num(events.WH_SUEP_cand_constituents) > 1
        indices = indices[singleTrackCut]
        events = events[singleTrackCut]
        output["cutflow_twoTracksInCluster" + out_label] += ak.sum(events.genWeight)

        # output file if no events pass selections, avoids errors later on
        if len(events) == 0:
            print("\n\nNo events pass singleTrackCut.\n\n")
            return

        ######################################################################################
        # ---- SUEP kinematics
        # Store SUEP kinematics
        #####################################################################################

        # boost into frame of SUEP
        boost_SUEP = ak.zip(
            {
                "px": events.WH_SUEP_cand.px * -1,
                "py": events.WH_SUEP_cand.py * -1,
                "pz": events.WH_SUEP_cand.pz * -1,
                "mass": events.WH_SUEP_cand.mass,
            },
            with_name="Momentum4D",
        )

        # SUEP tracks for this method are defined to be the ones from the cluster
        # that was picked to be the SUEP jet
        SUEP_cand_constituents_b = events.WH_SUEP_cand_constituents.boost_p4(
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
            ak.mean(events.WH_SUEP_cand_constituents.pt, axis=-1),
        )
        output["vars"].loc(
            indices,
            "SUEP_highestPTtrack_HighestPT" + out_label,
            ak.max(events.WH_SUEP_cand_constituents.pt, axis=-1),
        )
        output["vars"].loc(indices, "SUEP_pt_HighestPT" + out_label, events.WH_SUEP_cand.pt)
        output["vars"].loc(indices, "SUEP_eta_HighestPT" + out_label, events.WH_SUEP_cand.eta)
        output["vars"].loc(indices, "SUEP_phi_HighestPT" + out_label, events.WH_SUEP_cand.phi)
        output["vars"].loc(indices, "SUEP_mass_HighestPT" + out_label, events.WH_SUEP_cand.mass)

        # JEC corrected ak4jets inside SUEP cluster
        dR_ak4_SUEP = events.WH_jets_jec.deltaR(
            events.WH_SUEP_cand
        )  # delta R between jets (selecting events that pass the HighestPT selections) and the SUEP cluster
        ak4jets_inSUEPcluster = events.WH_jets_jec[dR_ak4_SUEP < 1.5]
        output["vars"].loc(
            indices,
            "ak4jets_inSUEPcluster_n_HighestPT" + out_label,
            ak.num(ak4jets_inSUEPcluster, axis=1),
        )
        output["vars"].loc(
            indices,
            "ak4jets_inSUEPcluster_pt_HighestPT" + out_label,
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
                "ak4jet" + str(i + 1) + "_inSUEPcluster_pt_HighestPT" + out_label,
                ak.fill_none(
                    ak.pad_none(
                        ak4jets_inSUEPcluster_ptsort.pt, i + 1, axis=1, clip=True
                    ),
                    -999,
                )[:, i],
            )
            output["vars"].loc(
                indices,
                "ak4jet" + str(i + 1) + "_inSUEPcluster_phi_HighestPT" + out_label,
                ak.fill_none(
                    ak.pad_none(
                        ak4jets_inSUEPcluster_ptsort.phi, i + 1, axis=1, clip=True
                    ),
                    -999,
                )[:, i],
            )
            output["vars"].loc(
                indices,
                "ak4jet" + str(i + 1) + "_inSUEPcluster_eta_HighestPT" + out_label,
                ak.fill_none(
                    ak.pad_none(
                        ak4jets_inSUEPcluster_ptsort.eta, i + 1, axis=1, clip=True
                    ),
                    -999,
                )[:, i],
            )
            output["vars"].loc(
                indices,
                "ak4jet" + str(i + 1) + "_inSUEPcluster_mass_HighestPT" + out_label,
                ak.fill_none(
                    ak.pad_none(
                        ak4jets_inSUEPcluster_ptsort.mass, i + 1, axis=1, clip=True
                    ),
                    -999,
                )[:, i],
            )

        # highest nconst non SUEP candidate (validate that highest pT ~= highest nconst)
        other_AK15_nconst = ak.num(events.WH_other_AK15_constituents, axis=-1)
        mostNumerousAK15 = events.WH_other_AK15[
            ak.argmax(other_AK15_nconst, axis=-1, keepdims=True)
        ]
        highestPT_otherAK15 = events.WH_other_AK15[
            ak.argmax(events.WH_other_AK15.pt, axis=-1, keepdims=True)
        ]
        output["vars"].loc(
            indices,
            "otherAK15_maxConst_pt_HighestPT",
            ak.fill_none(mostNumerousAK15.pt, 0).to_list(),
        )
        output["vars"].loc(
            indices,
            "otherAK15_maxConst_eta_HighestPT",
            ak.fill_none(mostNumerousAK15.eta, 0).to_list(),
        )
        output["vars"].loc(
            indices,
            "otherAK15_maxConst_phi_HighestPT",
            ak.fill_none(mostNumerousAK15.phi, 0).to_list(),
        )
        output["vars"].loc(
            indices,
            "otherAK15_maxConst_nconst_HighestPT",
            ak.fill_none(ak.max(other_AK15_nconst, axis=-1), 0).to_list(),
        )
        output["vars"].loc(
            indices,
            "otherAK15_maxPT_pt_HighestPT",
            ak.fill_none(highestPT_otherAK15.pt, 0),
        )
        output["vars"].loc(
            indices,
            "otherAK15_maxPT_eta_HighestPT",
            ak.fill_none(highestPT_otherAK15.eta, 0).to_list(),
        )
        output["vars"].loc(
            indices,
            "otherAK15_maxPT_phi_HighestPT",
            ak.fill_none(highestPT_otherAK15.phi, 0).to_list(),
        )

        # WH system
        if 'WH_W' in events.fields:
            VH_system = events.WH_SUEP_cand + events.WH_W
            output["vars"].loc(indices, "VH_system_mass_HighestPT", VH_system.mass)
            output["vars"].loc(indices, "VH_system_pt_HighestPT", VH_system.pt)
            output["vars"].loc(indices, "VH_system_phi_HighestPT", VH_system.phi)
            VH_system_PuppiMET = events.WH_SUEP_cand + events.WH_W_PuppiMET
            output["vars"].loc(
                indices, "VH_system_PuppiMET_mass_HighestPT", VH_system_PuppiMET.mass
            )
            output["vars"].loc(
                indices, "VH_system_PuppiMET_pt_HighestPT", VH_system_PuppiMET.pt
            )
            output["vars"].loc(
                indices, "VH_system_PuppiMET_phi_HighestPT", VH_system_PuppiMET.phi
            )
        if 'WH_gamma' in events.fields:
            GH_system = events.WH_SUEP_cand + events.WH_gamma
            output["vars"].loc(indices, "GH_system_mass_HighestPT", GH_system.mass)
            output["vars"].loc(indices, "GH_system_pt_HighestPT", GH_system.pt)
            output["vars"].loc(indices, "GH_system_phi_HighestPT", GH_system.phi)


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
        output["vars"]["ngood_ak4jets"] = ak.num(events.WH_jets_jec).to_list()
        output["vars"]["ht_JEC"] = ak.sum(events.WH_jets_jec.pt, axis=-1).to_list()

        # uncorrected jets
        ak4jets = WH_utils.getAK4Jets(events.Jet, events.run, iso=events.WH_lepton if not self.VRGJ else events.WH_gamma, isMC=self.isMC)
        output["vars"]["n_ak4jets"] = ak.num(ak4jets).to_list()
        output["vars"]["ht"] = ak.sum(ak4jets.pt, axis=-1).to_list()

        # gen variables
        if 'LHE' in events.fields:
            if 'Vpt' in events.LHE.fields:
                output["vars"]["LHE_Vpt"] = events.LHE.Vpt

        # saving number of bjets for different definitions (higher or lower requirements on b-likeliness) - see btag_utils.py
        # btag function requests eras as integers (used again for btag weights)
        if self.era == "2016apv":
            era_int = 2015
        else:
            era_int = int(self.era)
        output["vars"]["nBLoose"] = ak.sum(
            (events.WH_jets_jec.btag >= btagcuts("Loose", era_int)), axis=1
        )[:]
        output["vars"]["nBMedium"] = ak.sum(
            (events.WH_jets_jec.btag >= btagcuts("Medium", era_int)), axis=1
        )[:]
        output["vars"]["nBTight"] = ak.sum(
            (events.WH_jets_jec.btag >= btagcuts("Tight", era_int)), axis=1
        )[:]
        # store jet information (jet pt, eta, hadronFlavor, btag) in histogram for bjet eff calculation
        if self.isMC and self.storeJetsInfo:
            btag_category = ak.where(
              (events.WH_jets_jec.btag < btagcuts("Loose", era_int)),
                0,
                ak.where(
                    events.WH_jets_jec.btag < btagcuts("Tight", era_int),
                    1,
                    2
                )
            )
            maxnjets = ak.max(ak.num(events.WH_jets_jec, axis=1))
            output["vars"]["jets_btag_category"] = ak.fill_none(ak.pad_none(btag_category, maxnjets, axis=1, clip=True), -999)[:,:].to_list()
            output["vars"]["jets_pt"] = ak.fill_none(ak.pad_none(events.WH_jets_jec.pt,  maxnjets, axis=1, clip=True), 0.)[:,:].to_list()
            output["vars"]["jets_eta"] =ak.fill_none(ak.pad_none(events.WH_jets_jec.eta, maxnjets, axis=1, clip=True), -999)[:,:].to_list()
            output["vars"]["jets_hadronFlavor"] = ak.fill_none(ak.pad_none(events.WH_jets_jec.hadronFlavour, maxnjets, axis=1, clip=True), -1)[:,:].to_list()

        # saving kinematic variables for three leading pT jets
        highpt_jet = ak.argsort(events.WH_jets_jec.pt, axis=1, ascending=False, stable=True)
        jets_pTsorted = events.WH_jets_jec[highpt_jet]
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
            events.WH_jets_jec.btag, axis=1, ascending=False, stable=True
        )
        jets_btag_sorted = events.WH_jets_jec[highbtag_jet]
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

        # saving kinematic variables for the deltaphi(min(jet,WH MET)) jet
        jets_deltaPhiMET = WH_utils.MET_delta_phi(events.WH_jets_jec, events.WH_MET)
        sorted_deltaphiMET_jets = events.WH_jets_jec[
            ak.argsort(jets_deltaPhiMET, axis=1, ascending=True)
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
        output["vars"]["MET_significance"] = events.MET.significance
        output["vars"]["MET_covXX"] = events.MET.covXX
        output["vars"]["MET_covXY"] = events.MET.covXY
        output["vars"]["MET_covYY"] = events.MET.covYY
        output["vars"]["MET_sumPtUnclustered"] = events.MET.sumPtUnclustered
        output["vars"]["WH_MET_pt"] = events.WH_MET.pt
        output["vars"]["WH_MET_phi"] = events.WH_MET.phi
        output["vars"]["WH_MET_sumEt"] = events.WH_MET.sumEt

        # systematic variations
        if self.isMC and self.do_syst:

            output["vars"]["PuppiMET_pt_JER_up"] = events.PuppiMET.ptJERUp
            output["vars"]["PuppiMET_pt_JER_down"] = events.PuppiMET.ptJERDown
            output["vars"]["PuppiMET_pt_JES_up"] = events.PuppiMET.ptJESUp
            output["vars"]["PuppiMET_pt_JES_down"] = events.PuppiMET.ptJESDown
            output["vars"]["PuppiMET_phi_JER_up"] = events.PuppiMET.phiJERUp
            output["vars"]["PuppiMET_phi_JER_down"] = events.PuppiMET.phiJERDown
            output["vars"]["PuppiMET_phi_JES_up"] = events.PuppiMET.phiJESUp
            output["vars"]["PuppiMET_phi_JES_down"] = events.PuppiMET.phiJESDown
            # TODO CorrectedMETFactory is broken! The following cannot be trusted!
            #output["vars"]["MET_JEC_pt_JER_up"] = events.WH_MET.JER.up.pt
            #output["vars"]["MET_JEC_pt_JER_down"] = events.WH_MET.JER.up.pt
            #output["vars"]["MET_JEC_pt_JES_up"] = events.WH_MET.JES_jes.up.pt
            #output["vars"]["MET_JEC_pt_JES_down"] = events.WH_MET.JES_jes.down.pt
            #output["vars"][
            #    "MET_JEC_pt_UnclusteredEnergy_up"
            #] = events.WH_MET.MET_UnclusteredEnergy.up.pt
            #output["vars"][
            #    "MET_JEC_pt_UnclusteredEnergy_down"
            #] = events.WH_MET.MET_UnclusteredEnergy.down.pt
            #output["vars"]["MET_JEC_phi"] = events.WH_MET.phi
            #output["vars"]["MET_JEC_phi_JER_up"] = events.WH_MET.JER.up.phi
            #output["vars"]["MET_JEC_phi_JER_down"] = events.WH_MET.JER.down.phi
            #output["vars"]["MET_JEC_phi_JES_up"] = events.WH_MET.JES_jes.up.phi
            #output["vars"]["MET_JEC_phi_JES_down"] = events.WH_MET.JES_jes.down.phi
            #output["vars"][
            #    "MET_JEC_phi_UnclusteredEnergy_up"
            #] = events.WH_MET.MET_UnclusteredEnergy.up.phi
            #output["vars"][
            #    "MET_JEC_phi_UnclusteredEnergy_down"
            #] = events.WH_MET.MET_UnclusteredEnergy.down.phi

        # event weights
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

            for metaCR in ["sr", "crwj", "crtt"]:
                bTagWeights = doBTagWeights(
                    events.WH_jets_jec, era_int, wps="TL", channel="wh_"+metaCR, do_syst=self.do_syst
                )  
                for var in bTagWeights.keys():
                    output["vars"]["bTagWeight_" + var + "_" + metaCR] = bTagWeights[var]

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
        if 'WH_lepton' in events.fields:
            output["vars"]["lepton_pt"] = events.WH_lepton.pt
            output["vars"]["lepton_eta"] = events.WH_lepton.eta
            output["vars"]["lepton_phi"] = events.WH_lepton.phi
            output["vars"]["lepton_mass"] = events.WH_lepton.mass
            output["vars"]["lepton_flavor"] = events.WH_lepton.pdgID
            output["vars"]["lepton_ID"] = events.WH_lepton.ID
            output["vars"]["lepton_IDMVA"] = events.WH_lepton.IDMVA
            output["vars"]["lepton_iso"] = events.WH_lepton.iso
            output["vars"]["lepton_isoMVA"] = events.WH_lepton.isoMVA
            output["vars"]["lepton_miniIso"] = events.WH_lepton.miniIso
            output["vars"]["lepton_dxy"] = events.WH_lepton.dxy
            output["vars"]["lepton_dz"] = events.WH_lepton.dz
            output["vars"]["lepton_pfIsoId"] = events.WH_lepton.pfIsoId

            # saving min, max delta R, phi, eta between any jet and the tight lepton
            jet_lepton_combinations_deltaR = np.abs(events.WH_jets_jec.deltaR(events.WH_lepton))
            jet_lepton_combinations_deltaPhi = np.abs(events.WH_jets_jec.deltaphi(events.WH_lepton))
            jet_lepton_combinations_deltaEta = np.abs(events.WH_jets_jec.deltaeta(events.WH_lepton))
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

        # other loose leptons
        looseMuons, looseElectrons, looseLeptons = WH_utils.getLooseLeptons(events)
        output["vars"]["nLooseLeptons"] = ak.num(looseLeptons).to_list()
        output["vars"]["nLooseMuons"] = ak.num(looseMuons).to_list()
        output["vars"]["nLooseElectrons"] = ak.num(looseElectrons).to_list()
        
        # loose not tight leptons
        _, _, looseNotTightLeptons = WH_utils.getLooseNotTightLeptons(events)
        highpt_looseNotTightLeptons = ak.argsort(
            looseNotTightLeptons.pt, axis=1, ascending=False, stable=True
        )
        looseNotTightLeptons_pTsorted = looseNotTightLeptons[highpt_looseNotTightLeptons]
        for i in range(3):
            output["vars"]["looseNotTightLepton" + str(i + 1) + "_pt"] = ak.fill_none(
                ak.pad_none(looseNotTightLeptons_pTsorted.pt, i + 1, axis=1, clip=True), -999
            )[:, i]
            output["vars"]["looseNotTightLepton" + str(i + 1) + "_phi"] = ak.fill_none(
                ak.pad_none(looseNotTightLeptons_pTsorted.phi, i + 1, axis=1, clip=True), -999
            )[:, i]
            output["vars"]["looseNotTightLepton" + str(i + 1) + "_eta"] = ak.fill_none(
                ak.pad_none(looseNotTightLeptons_pTsorted.eta, i + 1, axis=1, clip=True), -999
            )[:, i]
            output["vars"]["looseNotTightLepton" + str(i + 1) + "_flavor"] = ak.fill_none(
                ak.pad_none(looseNotTightLeptons_pTsorted.pdgID, i + 1, axis=1, clip=True), -999
            )[:, i]

        if 'nCRQCDleptons' in events.fields:
            output["vars"]["nCRQCDleptons"] = events.nCRQCDleptons
        
        # saving W information
        if 'WH_W' in events.fields:
            events = ak.with_field(events, WH_utils.make_Wt_4v(events.WH_lepton, events.PuppiMET), "WH_W_PuppiMET")
            events = ak.with_field(events, WH_utils.make_Wt_4v(events.WH_lepton, events.CaloMET), "WH_W_CaloMET")
            output["vars"]["W_pt"] = events.WH_W.pt
            output["vars"]["W_phi"] = events.WH_W.phi
            output["vars"]["W_mt"] = WH_utils.calc_W_mt(events.WH_lepton, events.WH_MET)
            output["vars"]["W_pt_PuppiMET"] = events.WH_W_PuppiMET.pt
            output["vars"]["W_phi_PuppiMET"] = events.WH_W_PuppiMET.phi
            output["vars"]["W_mt_PuppiMET"] = WH_utils.calc_W_mt(
                events.WH_lepton, events.PuppiMET
            )
            output["vars"]["W_pt_CaloMET"] = events.WH_W_CaloMET.pt
            output["vars"]["W_phi_CaloMET"] = events.WH_W_CaloMET.phi
            output["vars"]["W_mt_CaloMET"] = WH_utils.calc_W_mt(events.WH_lepton, events.CaloMET)

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

            # saving min delta phi between any jet and the W
            jet_W_deltaPhi = np.abs(events.WH_jets_jec.deltaphi(events.WH_W))
            output["vars"]["minDeltaPhiJetW"] = ak.fill_none(
                ak.min(jet_W_deltaPhi, axis=-1), -999
            )

        # photon information
        photons = WH_utils.getPhotons(events, self.isMC)
        output["vars"]["nphotons"] = ak.num(photons).to_list()

        # tight photon information
        if 'WH_gamma' in events.fields or self.VRGJ:

            output["vars"]["WH_gammaTriggerBits"] = events.WH_gammaTriggerBits
            if not self.isMC: output["vars"]["WH_gammaTriggerUnprescaleWeight"] = events.WH_gammaTriggerUnprescaleWeight
            if 'QCD_HT' in self.sample or 'QCD_Pt' in self.sample: output["vars"]["WH_no_doubleCountedPhotons"] = events.WH_no_doubleCountedPhotons
            if self.isMC: output = WH_utils.storeGenPhotonStuff(events, output)

            output["vars"]["photon_pt"] = events.WH_gamma.pt
            output["vars"]["photon_eta"] = events.WH_gamma.eta
            output["vars"]["photon_phi"] = events.WH_gamma.phi
            output["vars"]["photon_pixelSeed"] = events.WH_gamma.pixelSeed
            output["vars"]["photon_electronVeto"] = events.WH_gamma.electronVeto
            output["vars"]["photon_mvaID"] = events.WH_gamma.mvaID
            output["vars"]["photon_pfRelIso03_all"] = events.WH_gamma.pfRelIso03_all
            output["vars"]["photon_hoe"] = events.WH_gamma.hoe
            output["vars"]["photon_r9"] = events.WH_gamma.r9
            output["vars"]["photon_isScEtaEB"] = events.WH_gamma.isScEtaEB
            output["vars"]["photon_isScEtaEE"] = events.WH_gamma.isScEtaEE
            output["vars"]["photon_cutBased"] = events.WH_gamma.cutBased
            output["vars"]["photon_sieie"] = events.WH_gamma.sieie

            # saving min, max delta R, phi, eta between any jet and the tight photon
            jet_photon_combinations_deltaR = np.abs(events.WH_jets_jec.deltaR(events.WH_gamma))
            jet_photon_combinations_deltaPhi = np.abs(events.WH_jets_jec.deltaphi(events.WH_gamma))
            jet_photon_combinations_deltaEta = np.abs(events.WH_jets_jec.deltaeta(events.WH_gamma))
            output["vars"]["minDeltaRJetPhoton"] = ak.fill_none(
                ak.min(jet_photon_combinations_deltaR, axis=-1), -999
            )
            output["vars"]["maxDeltaRJetPhoton"] = ak.fill_none(
                ak.max(jet_photon_combinations_deltaR, axis=-1), -999
            )
            output["vars"]["minDeltaPhiJetPhoton"] = ak.fill_none(
                ak.min(jet_photon_combinations_deltaPhi, axis=-1), -999
            )
            output["vars"]["maxDeltaPhiJetPhoton"] = ak.fill_none(
                ak.max(jet_photon_combinations_deltaPhi, axis=-1), -999
            )
            output["vars"]["minDeltaEtaJetPhoton"] = ak.fill_none(
                ak.min(jet_photon_combinations_deltaEta, axis=-1), -999
            )
            output["vars"]["maxDeltaEtaJetPhoton"] = ak.fill_none(
                ak.max(jet_photon_combinations_deltaEta, axis=-1), -999
            )

        # saving min, max delta R, phi, eta between jets
        jet_combinations = ak.combinations(
            events.WH_jets_jec, 2, fields=["jet1", "jet2"], axis=-1
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

        # store tau information
        WH_utils.storeTausInfo(events, output)

        # store Z information for DY
        if self.isMC and 'DYJetsToLL' in self.sample:
            WH_utils.storeGenZAndDaughtersInfo(events, output)

        # store W information for W+jets
        if self.isMC and 'WJets' in self.sample:
            WH_utils.storeGenWInfo(events, output)

        return events
    

    def analysis(self, events, output, out_label=""):

        #####################################################################################
        # ---- Basic event selection
        # Define the events that we will use.
        # Apply triggers, golden JSON, quality filters, and orthogonality selections.
        #####################################################################################

        output["cutflow_total" + out_label] += ak.sum(events.genWeight)

        if self.isMC == 0:
            events = applyGoldenJSON(self, events)
        output["cutflow_goldenJSON" + out_label] += ak.sum(events.genWeight)

        events = WH_utils.genSelection(events, self.sample)
        output["cutflow_genCuts" + out_label] += ak.sum(events.genWeight)

        if not self.VRGJ:
            events = WH_utils.triggerSelection(
                events, self.sample, self.era, self.isMC, output, out_label
            )
            output["cutflow_allTriggers" + out_label] += ak.sum(events.genWeight)
        
        events = WH_utils.qualityFiltersSelection(events, self.era)
        output["cutflow_qualityFilters" + out_label] += ak.sum(events.genWeight)

        if self.VRGJ:
            events = WH_utils.VRGJOrthogonalitySelection(events, era=self.era)
        else:
            events = WH_utils.orthogonalitySelection(events)
        output["cutflow_orthogonality" + out_label] += ak.sum(events.genWeight)

        # output file if no events pass selections, avoids errors later on
        if len(events) == 0:
            print("\n\nNo events pass basic event selection.\n\n")
            return events, output

        #####################################################################################
        # ---- Lepton selection
        # Define the lepton objects and apply single lepton selection.
        # (For gamma+jets CR, apply photon selection.)
        #####################################################################################

        if not self.CRQCD and not self.VRGJ: 
            events = WH_utils.oneTightLeptonSelection(events)
            output["cutflow_oneTightLepton" + out_label] += ak.sum(events.genWeight)
        elif self.VRGJ:
            events = WH_utils.onePhotonSelection(events, self.isMC)
            output["cutflow_onePhoton" + out_label] += ak.sum(events.genWeight)
            events = WH_utils.prescaledGammaTriggersSelection(events, self.era, bool(self.isMC))
            output["cutflow_allTriggers" + out_label] += ak.sum(events.genWeight)
            events = WH_utils.doubleCountingGenPhotonsSelection(events, self.sample)
            output["cutflow_doublePhotons" + out_label] += ak.sum(events.genWeight)
        elif self.CRQCD: 
            events = WH_utils.CRQCDSelection(events)
            output["cutflow_oneLooseLepton" + out_label] += ak.sum(events.genWeight)

        # TODO do we apply an electron filter here too?
        # _, eventEleHEMCut = jetHEMFilter(self, events.WH_lepton, events.run)

        # output file if no events pass selections, avoids errors later on
        if len(events) == 0:
            print("\n\nNo events pass one lepton / photon.\n\n")
            return events, output

        #####################################################################################
        # ---- Jets
        # Grab corrected ak4jets, apply HEM filter, and require at least one ak4jet.
        #####################################################################################

        jets_factory = applyJECStoJets(self.sample, self.isMC, self.era, events, events.Jet, jer=self.isMC)  
        jets_jec = WH_utils.getAK4Jets(jets_factory, events.run, iso=events.WH_lepton if not self.VRGJ else events.WH_gamma, isMC=self.isMC)
        events = ak.with_field(events, jets_factory, "WH_jets_factory")
        events = ak.with_field(events, jets_jec, "WH_jets_jec")
        events = events[ak.num(events.WH_jets_jec) > 0]
        output["cutflow_oneAK4jet" + out_label] += ak.sum(events.genWeight)

        # TODO do we apply HEMcut to all jets (currently done in getAK4jets) or to the events?

        # TODO do we want this? (if so, should go in getAK4jets? or before we give the jets to the JEC corrector?)
        # _, eventJetVetoCut = JetVetoMap(events.WH_jets_jec, self.era)

        #####################################################################################
        # ---- MET and W
        # Form the MET and W objects.
        #####################################################################################

        events = ak.with_field(events, events.PuppiMET, "WH_MET")
        if not self.VRGJ:
            events = ak.with_field(events, WH_utils.make_Wt_4v(events.WH_lepton, events.WH_MET), "WH_W")
            events = events[events.WH_MET.pt > 20]
            output["cutflow_MET20" + out_label] += ak.sum(events.genWeight)

        # TODO do we want this?
        # eventMETHEMCut = METHEMFilter(self, events.WH_MET, events.run)    

        #####################################################################################
        # ---- Store event level information
        #####################################################################################

        # these only need to be saved once, as they shouldn't change even with track killing
        if out_label == "":
            events = self.storeEventVars(
                events,
                output=output,
            )

        #####################################################################################
        # ---- SUEP definition and analysis
        #####################################################################################

        self.HighestPTMethod(
            events,
            output=output,
            out_label=out_label,
        )

        return events, output

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
                "cutflow_oneTightLepton": processor.value_accumulator(float, 0),
                "cutflow_oneLooseLepton": processor.value_accumulator(float, 0),
                "cutflow_onePhoton": processor.value_accumulator(float, 0),
                "cutflow_doublePhotons": processor.value_accumulator(float, 0),
                "cutflow_qualityFilters": processor.value_accumulator(float, 0),
                "cutflow_jetHEMcut": processor.value_accumulator(float, 0),
                "cutflow_electronHEMcut": processor.value_accumulator(float, 0),
                "cutflow_METHEMcut": processor.value_accumulator(float, 0),
                "cutflow_JetVetoMap": processor.value_accumulator(float, 0),
                "cutflow_MET20": processor.value_accumulator(float, 0),
                "cutflow_oneAK4jet": processor.value_accumulator(float, 0),
                "cutflow_oneCluster": processor.value_accumulator(float, 0),
                "cutflow_twoTracksInCluster": processor.value_accumulator(float, 0),
                "vars": pandas_accumulator(pd.DataFrame())
            }
        )

        # gen weights
        if self.isMC:
            output["gensumweight"] += ak.sum(events.genWeight)
        else:
            genWeight = np.ones(len(events))
            events = ak.with_field(events, genWeight, "genWeight")

        # run the analysis
        events, output = self.analysis(events, output)

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
                    "cutflow_oneTightLepton_track_down": processor.value_accumulator(
                        float, 0
                    ),
                    "cutflow_oneLooseLepton_track_down": processor.value_accumulator(
                        float, 0
                    ),
                    "cutflow_onePhoton_track_down": processor.value_accumulator(float, 0),
                    "cutflow_qualityFilters_track_down": processor.value_accumulator(
                        float, 0
                    ),
                    "cutflow_jetHEMcut_track_down": processor.value_accumulator(float, 0),
                    "cutflow_electronHEMcut_track_down": processor.value_accumulator(float, 0),
                    "cutflow_METHEMcut_track_down": processor.value_accumulator(float, 0),
                    "cutflow_JetVetoMap_track_down": processor.value_accumulator(float, 0),
                    "cutflow_MET20_track_down": processor.value_accumulator(float, 0),
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

            if len(events) > 0:
                self.HighestPTMethod(
                    events,
                    output=output,
                    out_label="_track_down",
                )

        if self.dropNonMethodEvents and len(events) > 0:
            # this is not very efficient, as we are processing a lot of events that we drop
            # it also assumes that we have SUEP_nconst for each method
            methods = [column.split("SUEP_nconst_")[-1] for column in output["vars"].columns if "SUEP_nconst_" in column]
            if len(methods) == 0:
                # no events passed any method
                selection = np.zeros(len(events), dtype=bool)
            else:
                selection = np.any([~output["vars"]["SUEP_nconst_" + method].isnull() for method in methods], axis=0)
            events = events[selection]
            output["vars"] = output["vars"][selection]

        return {dataset: output}

    def postprocess(self, accumulator):
        return accumulator
