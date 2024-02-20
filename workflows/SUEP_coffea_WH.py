"""
SUEP_coffea_WH.py
Coffea producer for SUEP WH analysis. Uses fastjet package to recluster large jets:
https://github.com/scikit-hep/fastjet
Pietro Lugato, Chad Freer, Luca Lavezzo, Joey Reichert 2023
"""

import awkward as ak
import numpy as np
import pandas as pd
import vector
from coffea import processor

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
from workflows.CMS_corrections.track_killing_utils import (
    track_killing
)

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
        output_location = None,
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
        tracks,
        jets,
        clusters,
        leptons,
        output,
        out_label=None,
    ):
        #####################################################################################
        # ---- Highest pT Jet (PT)
        # SUEP defined as the highest pT jet
        #####################################################################################

        # remove events with less than 1 cluster (i.e. need at least SUEP candidate cluster)
        clusterCut = ak.num(jets, axis=1) > 0
        clusters = clusters[clusterCut]
        jets = jets[clusterCut]
        leptons = leptons[clusterCut]
        tracks = tracks[clusterCut]
        indices = indices[clusterCut]
        events = events[clusterCut]
        output["cutflow_oneCluster" + out_label] += ak.sum(events.genWeight)

        # output file if no events pass selections, avoids errors later on
        if len(tracks) == 0:
            print("No events pass clusterCut.")
            return

        # choose highest pT jet
        highpt_jet = ak.argsort(jets.pt, axis=1, ascending=False, stable=True)
        jets_pTsorted = jets[highpt_jet]
        clusters_pTsorted = clusters[highpt_jet]
        SUEP_cand = jets_pTsorted[:, 0]
        SUEP_cand_constituents = clusters_pTsorted[:, 0]

        # at least 2 tracks
        singleTrackCut = ak.num(SUEP_cand_constituents) > 1
        SUEP_cand = SUEP_cand[singleTrackCut]
        SUEP_cand_constituents = SUEP_cand_constituents[singleTrackCut]
        tracks = tracks[singleTrackCut]
        indices = indices[singleTrackCut]
        events = events[singleTrackCut]
        output["cutflow_twoTracksInCluster" + out_label] += ak.sum(events.genWeight)

        # output file if no events pass selections, avoids errors later on
        if len(indices) == 0:
            print("No events pass singleTrackCut.")
            return

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
        SUEP_tracks_b = SUEP_cand_constituents.boost_p4(
            boost_SUEP
        )  ### boost the SUEP tracks to their restframe

        # SUEP jet variables
        eigs = SUEP_utils.sphericity(SUEP_tracks_b, 1.0)  # Set r=1.0 for IRC safe
        output["vars"].loc(
            indices, "SUEP_nconst_HighestPT" + out_label, ak.num(SUEP_tracks_b)
        )
        output["vars"].loc(
            indices,
            "SUEP_pt_avg_b_HighestPT" + out_label,
            ak.mean(SUEP_tracks_b.pt, axis=-1),
        )
        output["vars"].loc(
            indices, "SUEP_S1_HighestPT" + out_label, 1.5 * (eigs[:, 1] + eigs[:, 0])
        )

        # unboost for these
        SUEP_tracks = SUEP_tracks_b.boost_p4(SUEP_cand)
        output["vars"].loc(
            indices, "SUEP_pt_avg_HighestPT" + out_label, ak.mean(SUEP_tracks.pt, axis=-1)
        )
        output["vars"].loc(indices, "SUEP_pt_HighestPT" + out_label, SUEP_cand.pt)
        output["vars"].loc(indices, "SUEP_eta_HighestPT" + out_label, SUEP_cand.eta)
        output["vars"].loc(indices, "SUEP_phi_HighestPT" + out_label, SUEP_cand.phi)
        output["vars"].loc(indices, "SUEP_mass_HighestPT" + out_label, SUEP_cand.mass)

        # Calculate gen SUEP and candidate SUEP differences
        SUEP_genEta_diff_HighestPT = (
            output["vars"]["SUEP_eta_HighestPT" + out_label]
            - output["vars"]["SUEP_genEta" + out_label]
        )
        SUEP_genPhi_diff_HighestPT = (
            output["vars"]["SUEP_phi_HighestPT" + out_label]
            - output["vars"]["SUEP_genPhi" + out_label]
        )
        SUEP_genR_diff_HighestPT = (
            SUEP_genEta_diff_HighestPT**2 + SUEP_genPhi_diff_HighestPT**2
        ) ** 0.5
        output["vars"][
            "SUEP_deltaEtaGen_HighestPT" + out_label
        ] = SUEP_genEta_diff_HighestPT
        output["vars"][
            "SUEP_deltaPhiGen_HighestPT" + out_label
        ] = SUEP_genPhi_diff_HighestPT
        output["vars"]["SUEP_deltaRGen_HighestPT" + out_label] = SUEP_genR_diff_HighestPT
        output["vars"].loc(
            indices,
            "SUEP_deltaMassGen_HighestPT" + out_label,
            (SUEP_cand.mass - output["vars"]["SUEP_genMass" + out_label][indices]),
        )
        output["vars"].loc(
            indices,
            "SUEP_deltaPtGen_HighestPT" + out_label,
            (SUEP_cand.pt - output["vars"]["SUEP_genPt" + out_label][indices]),
        )

        # delta phi for SUEP and MET
        output["vars"].loc(
            indices,
            "deltaPhi_SUEP_CaloMET" + out_label,
            WH_utils.MET_delta_phi(SUEP_cand, events.CaloMET),
        )
        output["vars"].loc(
            indices,
            "deltaPhi_SUEP_PuppiMET" + out_label,
            WH_utils.MET_delta_phi(SUEP_cand, events.PuppiMET),
        )
        output["vars"].loc(
            indices,
            "deltaPhi_SUEP_MET" + out_label,
            WH_utils.MET_delta_phi(SUEP_cand, events.MET),
        )

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
        """
        Store event variables in the output dictionary.
        """

        # these are different for track_down
        output["vars"]["ntracks" + out_label] = ak.num(tracks).to_list()
        output["vars"]["ngood_fastjets" + out_label] = ak.num(
            ak_inclusive_jets
        ).to_list()
        
        if out_label == "track_down": return # only the above variables are different for track_down

        # general event vars
        if self.isMC: output["vars"]["genweight"] = events.genWeight
        output["vars"]["event" + out_label] = events.event.to_list()
        output["vars"]["run" + out_label] = events.run
        output["vars"]["luminosityBlock" + out_label] = events.luminosityBlock
        output["vars"]["PV_npvs" + out_label] = events.PV.npvs
        output["vars"]["PV_npvsGood" + out_label] = events.PV.npvsGood

        # select out ak4jets
        ak4jets = WH_utils.getAK4Jets(events.Jet, lepton)
        jets_c, met_c = apply_jecs(
            self,
            Sample=self.sample,
            events=events,
            prefix="",
        )
        jet_HEM_Cut, _ = jetHEMFilter(self, jets_c, events.run)
        jets_c = jets_c[jet_HEM_Cut]
        jets_jec = WH_utils.getAK4Jets(jets_c, lepton)
        output["vars"]["ngood_ak4jets" + out_label] = ak.num(ak4jets).to_list()

        # ht
        output["vars"]["ht" + out_label] = ak.sum(ak4jets.pt, axis=-1).to_list()
        output["vars"]["ht_JEC" + out_label] = ak.sum(
            jets_jec.pt, axis=-1
        ).to_list()
        if self.isMC and self.do_syst:
            jets_jec_JERUp = WH_utils.getAK4Jets(jets_c["JER"].up, lepton)
            jets_jec_JERDown = WH_utils.getAK4Jets(jets_c["JER"].down, lepton)
            jets_jec_JESUp = WH_utils.getAK4Jets(jets_c["JES_jes"].up, lepton)
            jets_jec_JESDown = WH_utils.getAK4Jets(jets_c["JES_jes"].down, lepton)

            output["vars"]["ht_JEC" + out_label + "_JER_up"] = ak.sum(
                jets_jec_JERUp.pt, axis=-1
            ).to_list()
            output["vars"]["ht_JEC" + out_label + "_JER_down"] = ak.sum(
                jets_jec_JERDown.pt, axis=-1
            ).to_list()
            output["vars"]["ht_JEC" + out_label + "_JES_up"] = ak.sum(
                jets_jec_JESUp.pt, axis=-1
            ).to_list()
            output["vars"]["ht_JEC" + out_label + "_JES_down"] = ak.sum(
                jets_jec_JESDown.pt, axis=-1
            ).to_list()

        # saving number of bjets for different definitions (higher or lower requirements on b-likeliness) - see btag_utils.py
        output["vars"]["nBLoose" + out_label] = ak.sum(
            (ak4jets.btag >= btagcuts("Loose", int(self.era))), axis=1
        )[:]
        output["vars"]["nBMedium" + out_label] = ak.sum(
            (ak4jets.btag >= btagcuts("Medium", int(self.era))), axis=1
        )[:]
        output["vars"]["nBTight" + out_label] = ak.sum(
            (ak4jets.btag >= btagcuts("Tight", int(self.era))), axis=1
        )[:]

        # saving kinematic variables for three leading pT jets
        highpt_jet = ak.argsort(ak4jets.pt, axis=1, ascending=False, stable=True)
        jets_pTsorted = ak4jets[highpt_jet]
        for i in range(3):
            output["vars"]["jet" + str(i + 1) + "_pT" + out_label] = ak.fill_none(
                ak.pad_none(jets_pTsorted.pt, i + 1, axis=1, clip=True), -999
            )[:, i]
            output["vars"]["jet" + str(i + 1) + "_phi" + out_label] = ak.fill_none(
                ak.pad_none(jets_pTsorted.phi, i + 1, axis=1, clip=True), -999
            )[:, i]
            output["vars"]["jet" + str(i + 1) + "_eta" + out_label] = ak.fill_none(
                ak.pad_none(jets_pTsorted.eta, i + 1, axis=1, clip=True), -999
            )[:, i]
            output["vars"]["jet" + str(i + 1) + "_qgl" + out_label] = ak.fill_none(
                ak.pad_none(jets_pTsorted.qgl, i + 1, axis=1, clip=True), -999
            )[:, i]

        # saving kinematic variables for the leading b-tagged jet
        highbtag_jet = ak.argsort(ak4jets.btag, axis=1, ascending=False, stable=True)
        jets_btag_sorted = ak4jets[highbtag_jet]
        output["vars"]["bjet_pt" + out_label] = ak.fill_none(
            ak.pad_none(jets_btag_sorted.pt, 1, axis=1, clip=True), 0.0
        )[:, 0]
        output["vars"]["bjet_phi" + out_label] = ak.fill_none(
            ak.pad_none(jets_btag_sorted.phi, 1, axis=1, clip=True), -1
        )[:, 0]
        output["vars"]["bjet_eta" + out_label] = ak.fill_none(
            ak.pad_none(jets_btag_sorted.eta, 1, axis=1, clip=True), -1
        )[:, 0]
        output["vars"]["bjet_qgl" + out_label] = ak.fill_none(
            ak.pad_none(jets_pTsorted.qgl, 1, axis=1, clip=True), -1.0
        )[:, 0]

        # saving MET variables
        output["vars"]["CaloMET_pt" + out_label] = events.CaloMET.pt
        output["vars"]["CaloMET_phi" + out_label] = events.CaloMET.phi
        output["vars"]["CaloMET_sumEt" + out_label] = events.CaloMET.sumEt
        output["vars"]["PuppiMET_pt" + out_label] = events.PuppiMET.pt
        output["vars"]["PuppiMET_phi" + out_label] = events.PuppiMET.phi
        output["vars"]["PuppiMET_sumEt" + out_label] = events.PuppiMET.sumEt
        output["vars"]["MET_pt" + out_label] = events.MET.pt
        output["vars"]["MET_phi" + out_label] = events.MET.phi
        output["vars"]["MET_sumEt" + out_label] = events.MET.sumEt

        # Will not be used for nominal analysis but keep around for studies
        """
        output["vars"]["ChsMET_pt" + out_label] = events.ChsMET.pt
        output["vars"]["ChsMET_phi" + out_label] = events.ChsMET.phi
        output["vars"]["ChsMET_sumEt" + out_label] = events.ChsMET.sumEt
        output["vars"]["TkMET_pt" + out_label] = events.TkMET.pt
        output["vars"]["TkMET_phi" + out_label] = events.TkMET.phi
        output["vars"]["TkMET_sumEt" + out_label] = events.TkMET.sumEt
        output["vars"]["RawMET_pt" + out_label] = events.RawMET.pt
        output["vars"]["RawMET_phi" + out_label] = events.RawMET.phi
        output["vars"]["RawMET_sumEt" + out_label] = events.RawMET.sumEt
        output["vars"]["RawPuppiMET_pt" + out_label] = events.RawPuppiMET.pt
        output["vars"]["RawPuppiMET_phi" + out_label] = events.RawPuppiMET.phi
        output["vars"]["MET_JEC_pt" + out_label] = met_c.pt
        output["vars"]["MET_JEC_sumEt" + out_label] = met_c.sumEt
        """

        # corrections on MET
        if self.isMC and self.do_syst:

            output["vars"]["PuppiMET_pt" + out_label + "_JER_up"] = events.PuppiMET.ptJERUp
            output["vars"][
                "PuppiMET_pt" + out_label + "_JER_down"
            ] = events.PuppiMET.ptJERDown
            output["vars"]["PuppiMET_pt" + out_label + "_JES_up"] = events.PuppiMET.ptJESUp
            output["vars"][
                "PuppiMET_pt" + out_label + "_JES_down"
            ] = events.PuppiMET.ptJESDown
            output["vars"]["PuppiMET_phi" + out_label + "_JER_up"] =  events.PuppiMET.phiJERUp
            output["vars"][
                "PuppiMET_phi" + out_label + "_JER_down"
            ] = events.PuppiMET.phiJERDown
            output["vars"]["PuppiMET_phi" + out_label + "_JES_up"] = events.PuppiMET.phiJESUp
            output["vars"][
                "PuppiMET_phi" + out_label + "_JES_down"
            ] = events.PuppiMET.phiJESDown
            output["vars"]["MET_JEC_pt" + out_label + "_JER_up"] = met_c.JER.up.pt
            output["vars"]["MET_JEC_pt" + out_label + "_JER_down"] =  met_c.JER.up.pt
            output["vars"]["MET_JEC_pt" + out_label + "_JES_up"] = met_c.JES_jes.up.pt
            output["vars"]["MET_JEC_pt" + out_label + "_JES_down"] = met_c.JES_jes.down.pt
            output["vars"][
                "MET_JEC_pt" + out_label + "_UnclusteredEnergy_up"
            ] = met_c.MET_UnclusteredEnergy.up.pt
            output["vars"][
                "MET_JEC_pt" + out_label + "_UnclusteredEnergy_down"
            ] = met_c.MET_UnclusteredEnergy.down.pt
            output["vars"]["MET_JEC_phi" + out_label] = met_c.phi
            output["vars"]["MET_JEC_phi" + out_label + "_JER_up"] = met_c.JER.up.phi
            output["vars"][
                "MET_JEC_phi" + out_label + "_JER_down"
            ] = met_c.JER.down.phi
            output["vars"]["MET_JEC_phi" + out_label + "_JES_up"] = met_c.JES_jes.up.phi
            output["vars"][
                "MET_JEC_phi" + out_label + "_JES_down"
            ] = met_c.JES_jes.down.phi
            output["vars"][
                "MET_JEC_phi" + out_label + "_UnclusteredEnergy_up"
            ] = met_c.MET_UnclusteredEnergy.up.phi
            output["vars"][
                "MET_JEC_phi" + out_label + "_UnclusteredEnergy_down"
            ] = met_c.MET_UnclusteredEnergy.down.phi

        if self.isMC:
            output["vars"]["Pileup_nTrueInt" + out_label] = events.Pileup.nTrueInt
            psweights = GetPSWeights(self, events)  # Parton Shower weights
            if len(psweights) == 4:
                output["vars"]["PSWeight_ISR_up" + out_label] = psweights[0]
                output["vars"]["PSWeight_ISR_down" + out_label] = psweights[1]
                output["vars"]["PSWeight_FSR_up" + out_label] = psweights[2]
                output["vars"]["PSWeight_FSR_down" + out_label] = psweights[3]
            else:
                output["vars"]["PSWeight" + out_label] = psweights

            bTagWeights = doBTagWeights(
                events, ak4jets, int(self.era), "L", do_syst=self.do_syst
            )  # Does not change selection
            output["vars"]["bTagWeight"] = bTagWeights["central"][:]  # BTag weights

            prefireweights = GetPrefireWeights(self, events)  # Prefire weights
            output["vars"]["prefire_nom"] = prefireweights[0]
            output["vars"]["prefire_up"] = prefireweights[1]
            output["vars"]["prefire_down"] = prefireweights[2]

        # get gen SUEP kinematics
        SUEP_genMass = len(events) * [0]
        SUEP_genPt = len(events) * [0]
        SUEP_genEta = len(events) * [0]
        SUEP_genPhi = len(events) * [0]

        if self.isMC:
            genParts = WH_utils.getGenTracks(events)
            genSUEP = genParts[(abs(genParts.pdgID) == 25)]

            # we need to grab the last SUEP in the chain for each event
            SUEP_genMass = [g[-1].mass if len(g) > 0 else 0 for g in genSUEP]
            SUEP_genPt = [g[-1].pt if len(g) > 0 else 0 for g in genSUEP]
            SUEP_genPhi = [g[-1].phi if len(g) > 0 else 0 for g in genSUEP]
            SUEP_genEta = [g[-1].eta if len(g) > 0 else 0 for g in genSUEP]

        output["vars"]["SUEP_genMass" + out_label] = SUEP_genMass
        output["vars"]["SUEP_genPt" + out_label] = SUEP_genPt
        output["vars"]["SUEP_genEta" + out_label] = SUEP_genEta
        output["vars"]["SUEP_genPhi" + out_label] = SUEP_genPhi

        # saving lepton kinematics
        output["vars"]["lepton_pt" + out_label] = lepton.pt[:, 0]
        output["vars"]["lepton_eta" + out_label] = lepton.eta[:, 0]
        output["vars"]["lepton_phi" + out_label] = lepton.phi[:, 0]
        output["vars"]["lepton_mass" + out_label] = lepton.mass[:, 0]
        output["vars"]["lepton_flavor" + out_label] = lepton.pdgID[:, 0]
        output["vars"]["lepton_ID" + out_label] = lepton.ID[:, 0]
        output["vars"]["lepton_IDMVA" + out_label] = lepton.IDMVA[:, 0]
        output["vars"]["lepton_iso" + out_label] = lepton.iso[:, 0]
        output["vars"]["lepton_isoMVA" + out_label] = lepton.isoMVA[:, 0]
        output["vars"]["lepton_miniIso" + out_label] = lepton.miniIso[:, 0]
        output["vars"]["lepton_dxy" + out_label] = lepton.dxy[:, 0]
        output["vars"]["lepton_dz" + out_label] = lepton.dz[:, 0]

        # W kinematics
        (
            W_mT_from_CaloMET,
            W_pT_from_CaloMET,
            W_phi_from_CaloMET,
        ) = WH_utils.W_kinematics(lepton, events.CaloMET)
        (
            W_mT_from_PuppiMET,
            W_pT_from_PuppiMET,
            W_phi_from_PuppiMET,
        ) = WH_utils.W_kinematics(lepton, events.PuppiMET)
        W_mT_from_MET, W_pT_from_MET, W_phi_from_MET = WH_utils.W_kinematics(
            lepton, events.MET
        )

        # W transverse mass for different METs -- zero mass for lepton, MET in Mt calculation
        output["vars"]["W_mT_from_CaloMET" + out_label] = W_mT_from_CaloMET
        output["vars"]["W_mT_from_PuppiMET" + out_label] = W_mT_from_PuppiMET
        output["vars"]["W_mT_from_MET" + out_label] = W_mT_from_MET

        output["vars"]["W_pT_from_CaloMET" + out_label] = W_pT_from_CaloMET
        output["vars"]["W_pT_from_PuppiMET" + out_label] = W_pT_from_PuppiMET
        output["vars"]["W_pT_from_MET" + out_label] = W_pT_from_MET

        output["vars"]["W_phi_from_CaloMET" + out_label] = W_phi_from_CaloMET
        output["vars"]["W_phi_from_PuppiMET" + out_label] = W_phi_from_PuppiMET
        output["vars"]["W_phi_from_MET" + out_label] = W_phi_from_MET

        # delta phi for lepton and different METs
        output["vars"]["deltaPhi_lepton_CaloMET_func" + out_label] = (
            WH_utils.MET_delta_phi(lepton, events.CaloMET)
        )
        output["vars"]["deltaPhi_lepton_PuppiMET_func" + out_label] = (
            WH_utils.MET_delta_phi(lepton, events.PuppiMET)
        )
        output["vars"]["deltaPhi_lepton_MET_func" + out_label] = WH_utils.MET_delta_phi(
            lepton, events.MET
        )

    def analysis(self, events, output, do_syst=False, out_label=""):

        #####################################################################################
        # ---- Basic event selection
        # Define the events that we will use.
        # Apply triggers, golden JSON, quality filters, and orthogonality selections.
        #####################################################################################

        output["cutflow_total" + out_label] += ak.sum(events.genWeight)

        if self.isMC == 0:
            events = applyGoldenJSON(self, events)
            events.genWeight = np.ones(len(events)) # dummy value for data
        
        output["cutflow_goldenJSON" + out_label] += ak.sum(events.genWeight)

        events = WH_utils.triggerSelection(events, self.era, self.isMC, output, out_label)
        output["cutflow_allTriggers" + out_label] += ak.sum(events.genWeight)

        events = WH_utils.qualityFiltersSelection(events, self.era)
        output["cutflow_qualityFilters" + out_label] += ak.sum(events.genWeight)

        events = WH_utils.orthogonalitySelection(events)
        output["cutflow_orthogonality" + out_label] += ak.sum(events.genWeight)

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
        leptonSelection = (ak.num(tightLeptons) == 1)
        events = events[leptonSelection]
        tightLeptons = tightLeptons[leptonSelection]
        output["cutflow_oneLepton" + out_label] += ak.sum(events.genWeight)

        # output file if no events pass selections, avoids errors later on
        if len(events) == 0:
            print("No events pass oneLepton.")
            return output

        #####################################################################################
        # ---- Track selection
        # Prepare the clean PFCand matched to tracks collection, imposing a dR > 0.4
        # cut on tracks from the selected lepton.
        #####################################################################################

        tracks, _ = WH_utils.getTracks(events, lepton=tightLeptons, leptonIsolation=0.4)
        if self.isMC and do_syst:
            tracks = track_killing(self, tracks)

        # IMPORTANT: if using track_killing, do not put any selections between here and
        # storeEventVars. Put them after, and use the indices to fill the df.

        #####################################################################################
        # ---- FastJet reclustering
        # The jet clustering part. Cut on at least one ak15 cluster.
        #####################################################################################

        # make the ak15 clusters
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
            lepton=tightLeptons,
            output=output,
            out_label=out_label,
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
            tracks,
            ak_inclusive_jets,
            ak_inclusive_cluster,
            leptons=tightLeptons,
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
                "cutflow_triggerSingleMuon": processor.value_accumulator(float, 0),
                "cutflow_triggerDoubleMuon": processor.value_accumulator(float, 0),
                "cutflow_triggerEGamma": processor.value_accumulator(float, 0),
                "cutflow_allTriggers": processor.value_accumulator(float, 0),
                "cutflow_orthogonality": processor.value_accumulator(float, 0),
                "cutflow_oneLepton": processor.value_accumulator(float, 0),
                "cutflow_qualityFilters": processor.value_accumulator(float, 0),
                "cutflow_oneCluster": processor.value_accumulator(float, 0),
                "cutflow_twoTracksInCluster": processor.value_accumulator(float, 0),
                "vars": pandas_accumulator(pd.DataFrame()),
            }
        )

        # gen weights
        if self.isMC:
            output["gensumweight"] += ak.sum(events.genWeight)
        else:
            events.genWeight = np.ones(len(events)) # dummy value for data

        # run the analysis with the track systematics applied
        if self.isMC and self.do_syst:
            output.update(
                {
                    "cutflow_total_track_down": processor.value_accumulator(float, 0),
                    "cutflow_goldenJSON_track_down": processor.value_accumulator(
                        float, 0
                    ),
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
                    "cutflow_oneCluster_track_down": processor.value_accumulator(
                        float, 0
                    ),
                    "cutflow_twoTracksInCluster_track_down": processor.value_accumulator(float, 0),
                }
            )
            output = self.analysis(
                events, output, do_syst=True, out_label="_track_down"
            )

        # run the analysis
        output = self.analysis(events, output)

        return {dataset: output}

    def postprocess(self, accumulator):
        return accumulator
