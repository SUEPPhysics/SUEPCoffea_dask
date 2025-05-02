"""
SUEP_coffea_WH.py
Coffea producer for SUEP WH analysis. Uses fastjet package to recluster large jets:
https://github.com/scikit-hep/fastjet
Pietro Lugato, Chad Freer, Luca Lavezzo, Joey Reichert 2023
"""

import os
import time
import warnings
from copy import deepcopy

import awkward as ak
import numpy as np
import pandas as pd
import psutil
import vector
from coffea import processor
from hist import Hist
from numba import njit, prange, typed

warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

# Importing SUEP specific functions
import workflows.SUEP_utils as SUEP_utils
import workflows.WH_utils as WH_utils

# Importing CMS corrections
from workflows.CMS_corrections.btag_utils import btagcuts
from workflows.CMS_corrections.golden_jsons_utils import applyGoldenJSON
from workflows.CMS_corrections.HEM_utils import METHEMFilter, jetHEMFilter
from workflows.CMS_corrections.jetmet_utils import applyJECStoJets
from workflows.CMS_corrections.jetvetomap_utils import JetVetoMap
from workflows.CMS_corrections.leptonsf_utils import doWHLeptonSFs
from workflows.CMS_corrections.PartonShower_utils import GetPSWeights
from workflows.CMS_corrections.photonSF_utils import getPhotonSFs
from workflows.CMS_corrections.Prefire_utils import GetPrefireWeights
from workflows.CMS_corrections.track_killing_utils import track_killing

# IO utils
from workflows.utils.pandas_accumulator import pandas_accumulator

# Set vector behavior
vector.register_awkward()


@staticmethod
@njit
def deltaPhi_x_y(xphi, yphi):
    abs_dphi = np.zeros(len(xphi))

    for i in range(len(xphi)):
        if abs(xphi[i]) > 2 * np.pi or abs(yphi[i]) > 2 * np.pi:
            abs_dphi[i] = -999
        else:
            dphi = (xphi[i] - yphi[i] + np.pi) % (2 * np.pi) - np.pi
            abs_dphi[i] = abs(dphi)

    return abs_dphi


@njit
def find_minDeltaR(phis, etas):
    minDeltaRs = []
    for i, (iphi, ieta) in enumerate(zip(phis, etas)):
        minDeltaR = 1e6
        for j, (jphi, jeta) in enumerate(zip(phis, etas)):
            if i == j:
                continue
            dphi = (iphi - jphi + np.pi) % (2 * np.pi) - np.pi
            deta = ieta - jeta
            deltaR = np.sqrt(dphi**2 + deta**2)
            if deltaR < minDeltaR:
                minDeltaR = deltaR
        minDeltaRs.append(minDeltaR)
    return minDeltaRs


@njit
def match(
    tracks_pts,
    tracks_phis,
    tracks_etas,
    gen_pts,
    gen_phis,
    gen_etas,
    gen_pdgIds,
    minDeltaR=0.1,
    sigma_pt=0.05,
    sigmaDeltaR=0.01,
):

    sigmaDeltaR_squared = sigmaDeltaR**2
    minDeltaR_squared = minDeltaR**2

    # Sort indices by pT (descending order for efficiency)
    reco_ptorder = np.argsort(tracks_pts)[::-1]
    gen_ptorder = np.argsort(gen_pts)[::-1]

    # Boolean masks to track used indices (instead of Python sets)
    gen_used = np.zeros(gen_pts.shape[0], dtype=np.bool_)
    reco_used = np.zeros(tracks_pts.shape[0], dtype=np.bool_)
    best_dR2s = np.full(gen_pts.shape[0], 1e6, dtype=np.float64)
    best_pts = np.full(gen_pts.shape[0], 1e6, dtype=np.float64)
    reco_matched = np.full(
        tracks_pts.shape[0], -1, dtype=np.int64
    )  # Initialize output with -1 (no match)

    for iGen in gen_ptorder:

        gen_pt = gen_pts[iGen]
        gen_phi = gen_phis[iGen]
        gen_eta = gen_etas[iGen]

        best_ireco = -1
        best_chi2 = 1e6
        best_dR2 = 1e6
        best_pt = 1e6

        for iReco in reco_ptorder:
            if reco_used[iReco]:
                continue

            track_pt = tracks_pts[iReco]
            track_phi = tracks_phis[iReco]
            track_eta = tracks_etas[iReco]

            dphi = (track_phi - gen_phi + np.pi) % (2 * np.pi) - np.pi
            deta = track_eta - gen_eta
            deltaR2 = dphi**2 + deta**2
            if deltaR2 > minDeltaR_squared:
                continue  # Skip if out of range

            pt_bal_var = (track_pt - gen_pt) / (track_pt)
            chi2 = (pt_bal_var / sigma_pt) ** 2 + (deltaR2 / sigmaDeltaR_squared) ** 2
            if best_ireco == -1 or chi2 < best_chi2:
                best_ireco = iReco
                best_chi2 = chi2
                best_dR2 = deltaR2
                best_pt = pt_bal_var

        # Store match
        # reco_matched[iReco] = best_igen
        if best_ireco != -1:
            reco_used[best_ireco] = True  # Mark as used
            gen_used[iGen] = True
        best_dR2s[iGen] = best_dR2
        best_pts[iGen] = best_pt

    return gen_used, reco_matched, best_dR2s**0.5, best_pts


# @njit
# def match(tracks_pts, tracks_phis, tracks_etas, gen_pts, gen_phis, gen_etas, gen_pdgIds,
#           minDeltaR=0.05, sigma_pt=0.05, sigmaDeltaR=0.01):

#     sigmaDeltaR_squared = sigmaDeltaR ** 2
#     minDeltaR_squared = minDeltaR ** 2

#     # Sort indices by pT (descending order for efficiency)
#     reco_ptorder = np.argsort(tracks_pts)[::-1]
#     gen_ptorder = np.argsort(gen_pts)[::-1]

#     # Boolean masks to track used indices (instead of Python sets)
#     gen_used = np.zeros(gen_pts.shape[0], dtype=np.bool_)
#     best_dR2s = np.full(gen_pts.shape[0], 1e6, dtype=np.float64)
#     best_pts = np.full(gen_pts.shape[0], 1e6, dtype=np.float64)
#     reco_matched = np.full(tracks_pts.shape[0], -1, dtype=np.int64)  # Initialize output with -1 (no match)

#     # Loop over sorted reco tracks
#     for iReco in reco_ptorder:

#         track_pt = tracks_pts[iReco]
#         track_phi = tracks_phis[iReco]
#         track_eta = tracks_etas[iReco]

#         best_igen = -1
#         best_chi2 = 1e6
#         best_dR2 = 1e6
#         best_pt = 1e6

#         # Loop over sorted gen particles
#         for iGen in gen_ptorder:
#             if gen_used[iGen]:
#                 continue

#             gen_pt = gen_pts[iGen]
#             gen_phi = gen_phis[iGen]
#             gen_eta = gen_etas[iGen]

#             dphi = (track_phi - gen_phi + np.pi) % (2 * np.pi) - np.pi
#             deta = (track_eta - gen_eta)
#             deltaR2 = dphi ** 2 + deta ** 2
#             if deltaR2 > minDeltaR_squared:
#                 continue  # Skip if out of range

#             pt_bal_var = (track_pt - gen_pt) / (track_pt)
#             chi2 = ( pt_bal_var / sigma_pt) ** 2 + (deltaR2 / sigmaDeltaR_squared) **2
#             if best_igen == -1 or chi2 < best_chi2:
#                 best_igen = iGen
#                 best_chi2 = chi2
#                 best_dR2 = deltaR2
#                 best_pt = pt_bal_var

#         # Store match
#         reco_matched[iReco] = best_igen
#         if best_igen != -1:
#             gen_used[best_igen] = True  # Mark as used
#         best_dR2s[best_igen] = best_dR2
#         best_pts[best_igen] = best_pt

#     return gen_used, reco_matched, best_dR2s**0.5, best_pts


class SUEP_cluster_WH(processor.ProcessorABC):
    def __init__(
        self,
        isMC: int,
        era: str,
        sample: str,
        do_syst: bool,
        flag: bool,
        output_location=None,
        CRQCD: bool = False,
        VRGJ: bool = False,
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

    def HighestPTMethod(
        self,
        events,
        output,
        out_label=None,
        variation="",
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
            # iso_object=events.WH_lepton if not self.VRGJ else events.WH_gamma,
            # isolation_deltaR=0.4,
        )
        if self.isMC and "track_down" == variation:
            tracks = track_killing(self, tracks)
        elif self.isMC and "track_down_mod" == variation:
            tracks = modified_track_killing(
                tracks, WH_utils.getSignalDecayMode(self.sample)
            )
        events = ak.with_field(events, tracks, "WH_tracks")

        #####################################################################################
        # ---- FastJet reclustering
        # The jet clustering part.
        #####################################################################################

        # make the ak15 clusters
        ak15jets, clusters = SUEP_utils.FastJetReclustering(
            events.WH_tracks, r=1.5, minPt=5
        )
        output["leading_ak15_pt"].fill(
            ak.fill_none(ak.max(ak15jets.pt, axis=1), -999), weight=events.genWeight
        )
        ak15_60gev = ak15jets.pt > 60
        ak15jets = ak15jets[ak15_60gev]
        clusters = clusters[ak15_60gev]
        events = ak.with_field(events, ak15jets, "WH_ak15jets")
        events = ak.with_field(events, clusters, "WH_ak15clusters")

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
        highpt_jet = ak.argsort(
            events.WH_ak15jets.pt, axis=1, ascending=False, stable=True
        )
        ak15jets_pTsorted = events.WH_ak15jets[highpt_jet]
        clusters_pTsorted = events.WH_ak15clusters[highpt_jet]
        events = ak.with_field(events, ak15jets_pTsorted[:, 0], "WH_SUEP_cand")
        events = ak.with_field(
            events, clusters_pTsorted[:, 0], "WH_SUEP_cand_constituents"
        )
        events = ak.with_field(events, ak15jets_pTsorted[:, 1:], "WH_other_AK15")
        events = ak.with_field(
            events, clusters_pTsorted[:, 1:], "WH_other_AK15_constituents"
        )

        # at least 2 tracks
        singleTrackCut = ak.num(events.WH_SUEP_cand_constituents) >= 10
        indices = indices[singleTrackCut]
        events = events[singleTrackCut]
        output["cutflow_twoTracksInCluster" + out_label] += ak.sum(events.genWeight)

        # output file if no events pass selections, avoids errors later on
        if len(events) == 0:
            print("\n\nNo events pass singleTrackCut.\n\n")
            return None

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
        sphericity = 1.5 * (eigs[:, 1] + eigs[:, 0])
        events = ak.with_field(events, sphericity, "WH_SUEP_sphericity")

        # JEC corrected ak4jets inside SUEP cluster
        dR_ak4_SUEP = events.WH_jets_jec.deltaR(
            events.WH_SUEP_cand
        )  # delta R between jets (selecting events that pass the HighestPT selections) and the SUEP cluster
        ak4jets_inSUEPcluster = events.WH_jets_jec[dR_ak4_SUEP < 1.5]
        ak4jets_outsideSUEPcluster = events.WH_jets_jec[dR_ak4_SUEP > 1.5]

        ## some selections
        events = events[ak.num(ak4jets_inSUEPcluster) > 0]
        events = events[events.WH_SUEP_sphericity > 0.3]
        events = events[events.WH_W.pt / events.WH_SUEP_cand.pt < 3]
        events = events[
            deltaPhi_x_y(
                ak.to_numpy(events.WH_SUEP_cand.phi), ak.to_numpy(events.WH_MET.phi)
            )
            > 1.5
        ]
        events = events[
            deltaPhi_x_y(
                ak.to_numpy(events.WH_SUEP_cand.phi), ak.to_numpy(events.WH_W.phi)
            )
            > 1.5
        ]
        events = events[
            deltaPhi_x_y(
                ak.to_numpy(events.WH_SUEP_cand.phi), ak.to_numpy(events.WH_lepton.phi)
            )
            > 1.5
        ]

        output["SUEP_nconst"].fill(
            ak.num(events.WH_SUEP_cand_constituents), weight=events.genWeight
        )
        output["SUEP_S1"].fill(events.WH_SUEP_sphericity, weight=events.genWeight)

        return events

    def find_daughters(self, event, pdgId=25):
        """
        Find all daughters of a given particle by recursively searching through the genPartIdxMother array
        """

        daugtherIdxs = []
        for i in range(len(event.GenPart.pdgId)):
            status = event.GenPart.status[i]
            if status != 1:
                continue
            statusFlag = event.GenPart.statusFlags[i]
            if not (statusFlag & (1 << 13)):
                continue
            pt = event.GenPart.pt[i]
            if pt < 1:
                continue
            eta = event.GenPart.eta[i]
            if abs(eta) > 2.5:
                continue
            id = event.GenPart.pdgId[i]
            motherIdx = event.GenPart.genPartIdxMother[i]
            while True:
                # print(f'Particle {i} with pgdId {id} is a daughter of pdgId {event.GenPart.pdgId[motherIdx]} with index {motherIdx}')
                if motherIdx == -1:
                    break
                if event.GenPart.pdgId[motherIdx] == pdgId:
                    daugtherIdxs.append(i)
                    break
                else:
                    id = event.GenPart.pdgId[motherIdx]
                    motherIdx = event.GenPart.genPartIdxMother[motherIdx]

        return daugtherIdxs

    def process_event(self, event, output):

        events_gen_used = ak.Array([])
        try:
            event_daughters = self.find_daughters(event)

            # Directly use Awkward Arrays without conversion
            gen_pt = ak.to_numpy(event.GenPart[event_daughters].pt)
            gen_eta = ak.to_numpy(event.GenPart[event_daughters].eta)
            gen_phi = ak.to_numpy(event.GenPart[event_daughters].phi)
            gen_pdgId = ak.to_numpy(event.GenPart[event_daughters].pdgId)

            track_pt = ak.to_numpy(event.WH_tracks.pt)
            track_phi = ak.to_numpy(event.WH_tracks.phi)
            track_eta = ak.to_numpy(event.WH_tracks.eta)

            output["nGenDaughters"].fill(len(event_daughters), weight=event.genWeight)
            output["nGenDaughtersMuons"].fill(
                len(gen_pdgId[abs(gen_pdgId) == 13]), weight=event.genWeight
            )
            output["nGenDaughtersElectrons"].fill(
                len(gen_pdgId[abs(gen_pdgId) == 11]), weight=event.genWeight
            )

            minDeltaRs = find_minDeltaR(gen_phi, gen_eta)
            output["gentracks_minDeltaR"].fill(minDeltaRs, weight=event.genWeight)
            reco_minDeltaRs = find_minDeltaR(track_phi, track_eta)
            output["reco_minDeltaR"].fill(reco_minDeltaRs, weight=event.genWeight)

            gen_used, reco_matched, best_dR, best_pts = match(
                track_pt,
                track_phi,
                track_eta,
                gen_pt,
                gen_phi,
                gen_eta,
                gen_pdgId,
                minDeltaR=0.2,
                sigma_pt=0.05,
                sigmaDeltaR=0.05,
            )

            is_matched = gen_used
            is_mu = abs(gen_pdgId) == 13
            is_e = abs(gen_pdgId) == 11
            is_pi = abs(gen_pdgId) == 211
            is_kaon = abs(gen_pdgId) == 321
            is_what = ~is_mu & ~is_e & ~is_pi & ~is_kaon

            output["unexpected_pdgId"].fill(gen_pdgId[is_what], weight=event.genWeight)

            particle_types = {
                "mu": is_mu,
                "e": is_e,
                "pi": is_pi,
                "kaon": is_kaon,
            }

            for particle, mask in particle_types.items():
                matched = is_matched & mask
                unmatched = ~is_matched & mask

                matched_pt = gen_pt[matched]
                matched_phi = gen_phi[matched]
                matched_eta = gen_eta[matched]
                matched_minDeltaR = best_dR[matched]
                matched_deltaPt = best_pts[matched]

                unmatched_pt = gen_pt[unmatched]
                unmatched_phi = gen_phi[unmatched]
                unmatched_eta = gen_eta[unmatched]
                unmatched_minDeltaR = best_dR[unmatched]
                unmatched_deltaPt = best_pts[unmatched]

                # Fill histograms
                output[f"pt_{particle}_matched"].fill(
                    matched_pt, weight=event.genWeight
                )
                output[f"phi_{particle}_matched"].fill(
                    matched_phi, weight=event.genWeight
                )
                output[f"eta_{particle}_matched"].fill(
                    matched_eta, weight=event.genWeight
                )
                output[f"minDeltaR_{particle}_matched"].fill(
                    matched_minDeltaR, weight=event.genWeight
                )
                output[f"deltaPt_{particle}_matched"].fill(
                    matched_deltaPt, weight=event.genWeight
                )

                output[f"pt_{particle}_unmatched"].fill(
                    unmatched_pt, weight=event.genWeight
                )
                output[f"phi_{particle}_unmatched"].fill(
                    unmatched_phi, weight=event.genWeight
                )
                output[f"eta_{particle}_unmatched"].fill(
                    unmatched_eta, weight=event.genWeight
                )
                output[f"minDeltaR_{particle}_unmatched"].fill(
                    unmatched_minDeltaR, weight=event.genWeight
                )
                output[f"deltaPt_{particle}_unmatched"].fill(
                    unmatched_deltaPt, weight=event.genWeight
                )

        except Exception as e:
            import traceback

            print(f"Error processing event: {e}")
            traceback.print_exc()
            return {}

    def analysis(self, events, output, out_label: str = "", variation: str = ""):

        #####################################################################################
        # ---- Basic event selection
        # Define the events that we will use.
        # Apply triggers, golden JSON, quality filters, and orthogonality selections.
        #####################################################################################

        genW = WH_utils.getGenW(events)
        output["genW_pt_0"].fill(genW.pt[:, 0])

        output["cutflow_total" + out_label] += ak.sum(events.genWeight)

        if self.isMC == 0:
            events = applyGoldenJSON(self, events)
        output["cutflow_goldenJSON" + out_label] += ak.sum(events.genWeight)

        # output file if no events pass selections, avoids errors later on
        if len(events) == 0:
            print("\n\nNo events pass goldenJSON.\n\n")
            return events, output

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
            events = WH_utils.VRGJOrthogonalitySelection(
                events, era=self.era, isMC=self.isMC
            )
        else:
            events = WH_utils.orthogonalitySelection(events, isMC=self.isMC)
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
            events = WH_utils.oneTightLeptonSelection(
                events,
                era=self.era,
                isMC=self.isMC,
                variation=(
                    variation
                    if (("MuScale" in variation) or ("ElScale" in variation))
                    else ""
                ),
            )
            output["cutflow_oneTightLepton" + out_label] += ak.sum(events.genWeight)
        elif self.VRGJ:
            events = WH_utils.onePhotonSelection(events, self.isMC)
            output["cutflow_onePhoton" + out_label] += ak.sum(events.genWeight)
            events = WH_utils.prescaledGammaTriggersSelection(
                events, self.era, bool(self.isMC)
            )
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

        jets_factory = applyJECStoJets(
            self.sample, self.isMC, self.era, events, events.Jet, jer=self.isMC
        )
        jets_jec = WH_utils.getAK4Jets(
            jets_factory,
            events.run,
            iso=events.WH_lepton if not self.VRGJ else events.WH_gamma,
            isMC=self.isMC,
        )
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

        genW = WH_utils.getGenW(events)
        output["genW_pt_3"].fill(genW.pt[:, 0])

        output["muon_pt_1"].fill(
            events.WH_lepton[abs(events.WH_lepton.pdgId) == 13].pt,
            weight=events.genWeight[abs(events.WH_lepton.pdgId) == 13],
        )
        output["electron_pt_1"].fill(
            events.WH_lepton[abs(events.WH_lepton.pdgId) == 11].pt,
            weight=events.genWeight[abs(events.WH_lepton.pdgId) == 11],
        )

        events = ak.with_field(events, events.PuppiMET, "WH_MET")
        if not self.VRGJ:
            events = ak.with_field(
                events, WH_utils.make_Wt_4v(events.WH_lepton, events.WH_MET), "WH_W"
            )
            events = events[events.WH_MET.pt > 30]
            output["cutflow_MET20" + out_label] += ak.sum(events.genWeight)

        if len(events) == 0:
            print("\n\nNo events pass MET pt > 20.\n\n")
            return events, output

        #####################################################################################
        # ---- More event selections
        #####################################################################################

        genW = WH_utils.getGenW(events)
        output["genW_pt_4"].fill(genW.pt[:, 0])

        events = events[events.WH_W.pt > 60]
        events = events[events.WH_W.mt < 130]
        events = events[events.WH_W.mt > 30]
        if self.era == "2016apv":
            era_int = 2015
        else:
            era_int = int(self.era)
        nBLoose = ak.sum(
            (events.WH_jets_jec.btag >= btagcuts("Loose", era_int)), axis=1
        )[:]

        nBTight = ak.sum(
            (events.WH_jets_jec.btag >= btagcuts("Tight", era_int)), axis=1
        )[:]
        events = events[(nBLoose <= 1) & (nBTight == 0)]

        if len(events) == 0:
            return events, output

        #####################################################################################
        # ---- SUEP definition and analysis
        #####################################################################################

        genW = WH_utils.getGenW(events)
        output["genW_pt_1"].fill(genW.pt[:, 0])

        events = self.HighestPTMethod(
            events,
            output=output,
            out_label=out_label,
            variation=variation,
        )

        if type(events) == type(None):
            return events, output

        genW = WH_utils.getGenW(events)
        output["genW_pt_2"].fill(genW.pt[:, 0])

        # print()
        # print()
        # for e in events:
        #     print(e.run, e.luminosityBlock, e.event)
        # print()
        # print()

        for iEvent, event in enumerate(events):
            self.process_event(event, output)

        return events, output

    def process(self, events):
        dataset = events.metadata["dataset"]

        blank_output = processor.dict_accumulator(
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
                "vars": pandas_accumulator(pd.DataFrame()),
                "leading_ak15_pt": Hist.new.Reg(
                    400,
                    0,
                    400,
                    name="leading_ak15_pt",
                    label="Leading AK15 cluster $p_T$ [GeV]",
                ).Weight(),
                "n_ak15": Hist.new.Reg(
                    10, 0, 10, name="n_ak15", label="$n_{\mathrm{AK15}}$"
                ).Weight(),
                "n_ak15_60gev": Hist.new.Reg(
                    10, 0, 10, name="n_ak15_60gev", label="$n_{\mathrm{AK15}}$"
                ).Weight(),
                "pt_e_matched": Hist.new.Reg(
                    300, 0, 100, name="e_matched_pt", label="$p_T$ [GeV]"
                ).Weight(),
                "pt_e_unmatched": Hist.new.Reg(
                    300, 0, 100, name="e_unmatched_pt", label="$p_T$ [GeV]"
                ).Weight(),
                "pt_mu_matched": Hist.new.Reg(
                    300, 0, 100, name="mu_matched_pt", label="$p_T$ [GeV]"
                ).Weight(),
                "pt_mu_unmatched": Hist.new.Reg(
                    300, 0, 100, name="mu_unmatched_pt", label="$p_T$ [GeV]"
                ).Weight(),
                "phi_e_matched": Hist.new.Reg(
                    100, -6, 6, name="e_matched_phi", label="$\phi$ [GeV]"
                ).Weight(),
                "phi_e_unmatched": Hist.new.Reg(
                    100, -6, 6, name="e_unmatched_phi", label="$\phi$ [GeV]"
                ).Weight(),
                "phi_mu_matched": Hist.new.Reg(
                    100, -6, 6, name="mu_matched_phi", label="$\phi$ [GeV]"
                ).Weight(),
                "phi_mu_unmatched": Hist.new.Reg(
                    100, -6, 6, name="mu_unmatched_phi", label="$\phi$ [GeV]"
                ).Weight(),
                "eta_e_matched": Hist.new.Reg(
                    100, -6, 6, name="e_matched_eta", label="$\eta$ [GeV]"
                ).Weight(),
                "eta_e_unmatched": Hist.new.Reg(
                    100, -6, 6, name="e_unmatched_eta", label="$η$ [GeV]"
                ).Weight(),
                "eta_mu_matched": Hist.new.Reg(
                    100, -6, 6, name="mu_matched_eta", label="$η$ [GeV]"
                ).Weight(),
                "eta_mu_unmatched": Hist.new.Reg(
                    100, -6, 6, name="mu_unmatched_eta", label="$η$ [GeV]"
                ).Weight(),
                "minDeltaR_e_matched": Hist.new.Reg(
                    1000, 0, 0.5, name="e_matched_minDeltaR", label="min $\Delta R$"
                ).Weight(),
                "minDeltaR_e_unmatched": Hist.new.Reg(
                    1000, 0, 0.5, name="e_unmatched_minDeltaR", label="min $\Delta R$"
                ).Weight(),
                "minDeltaR_mu_matched": Hist.new.Reg(
                    1000, 0, 0.5, name="mu_matched_minDeltaR", label="min $\Delta R$"
                ).Weight(),
                "minDeltaR_mu_unmatched": Hist.new.Reg(
                    1000, 0, 0.5, name="mu_unmatched_minDeltaR", label="min $\Delta R$"
                ).Weight(),
                "deltaPt_e_matched": Hist.new.Reg(
                    1000,
                    -20,
                    20,
                    name="e_matched_deltaPt",
                    label="$p^{reco}_T - p^{gen}_T / p^{reco}_T$",
                ).Weight(),
                "deltaPt_e_unmatched": Hist.new.Reg(
                    1000,
                    -20,
                    20,
                    name="e_unmatched_deltaPt",
                    label="$p^{reco}_T - p^{gen}_T / p^{reco}_T$",
                ).Weight(),
                "deltaPt_mu_matched": Hist.new.Reg(
                    1000,
                    -20,
                    20,
                    name="mu_matched_deltaPt",
                    label="$p^{reco}_T - p^{gen}_T / p^{reco}_T$",
                ).Weight(),
                "deltaPt_mu_unmatched": Hist.new.Reg(
                    1000,
                    -20,
                    20,
                    name="mu_unmatched_deltaPt",
                    label="$p^{reco}_T - p^{gen}_T / p^{reco}_T$",
                ).Weight(),
                "pt_pi_matched": Hist.new.Reg(
                    300, 0, 100, name="pi_matched_pt", label="$p_T$ [GeV]"
                ).Weight(),
                "pt_pi_unmatched": Hist.new.Reg(
                    300, 0, 100, name="pi_unmatched_pt", label="$p_T$ [GeV]"
                ).Weight(),
                "phi_pi_matched": Hist.new.Reg(
                    100, -6, 6, name="pi_matched_phi", label="$\phi$ [GeV]"
                ).Weight(),
                "phi_pi_unmatched": Hist.new.Reg(
                    100, -6, 6, name="pi_unmatched_phi", label="$\phi$ [GeV]"
                ).Weight(),
                "eta_pi_matched": Hist.new.Reg(
                    100, -6, 6, name="pi_matched_eta", label="$η$ [GeV]"
                ).Weight(),
                "eta_pi_unmatched": Hist.new.Reg(
                    100, -6, 6, name="pi_unmatched_eta", label="$η$ [GeV]"
                ).Weight(),
                "minDeltaR_pi_matched": Hist.new.Reg(
                    1000, 0, 0.5, name="pi_matched_minDeltaR", label="min $\Delta R$"
                ).Weight(),
                "minDeltaR_pi_unmatched": Hist.new.Reg(
                    1000, 0, 0.5, name="pi_unmatched_minDeltaR", label="min $\Delta R$"
                ).Weight(),
                "deltaPt_pi_matched": Hist.new.Reg(
                    1000,
                    -20,
                    20,
                    name="pi_matched_deltaPt",
                    label="$p^{reco}_T - p^{gen}_T / p^{reco}_T$",
                ).Weight(),
                "deltaPt_pi_unmatched": Hist.new.Reg(
                    1000,
                    -20,
                    20,
                    name="pi_unmatched_deltaPt",
                    label="$p^{reco}_T - p^{gen}_T / p^{reco}_T$",
                ).Weight(),
                "pt_kaon_matched": Hist.new.Reg(
                    300, 0, 100, name="kaon_matched_pt", label="$p_T$ [GeV]"
                ).Weight(),
                "pt_kaon_unmatched": Hist.new.Reg(
                    300, 0, 100, name="kaon_unmatched_pt", label="$p_T$ [GeV]"
                ).Weight(),
                "phi_kaon_matched": Hist.new.Reg(
                    100, -6, 6, name="kaon_matched_phi", label="$\phi$ [GeV]"
                ).Weight(),
                "phi_kaon_unmatched": Hist.new.Reg(
                    100, -6, 6, name="kaon_unmatched_phi", label="$\phi$ [GeV]"
                ).Weight(),
                "eta_kaon_matched": Hist.new.Reg(
                    100, -6, 6, name="kaon_matched_eta", label="$η$ [GeV]"
                ).Weight(),
                "eta_kaon_unmatched": Hist.new.Reg(
                    100, -6, 6, name="kaon_unmatched_eta", label="$η$ [GeV]"
                ).Weight(),
                "minDeltaR_kaon_matched": Hist.new.Reg(
                    1000, 0, 0.5, name="kaon_matched_minDeltaR", label="min $\Delta R$"
                ).Weight(),
                "minDeltaR_kaon_unmatched": Hist.new.Reg(
                    1000,
                    0,
                    0.5,
                    name="kaon_unmatched_minDeltaR",
                    label="min $\Delta R$",
                ).Weight(),
                "deltaPt_kaon_matched": Hist.new.Reg(
                    1000,
                    -20,
                    20,
                    name="kaon_matched_deltaPt",
                    label="$p^{reco}_T - p^{gen}_T / p^{reco}_T$",
                ).Weight(),
                "deltaPt_kaon_unmatched": Hist.new.Reg(
                    1000,
                    -20,
                    20,
                    name="kaon_unmatched_deltaPt",
                    label="$p^{reco}_T - p^{gen}_T / p^{reco}_T$",
                ).Weight(),
                "unexpected_pdgId": Hist.new.Reg(
                    100, -50, 50, name="unexpected_pdgId", label="Unexpected pdgId"
                ).Weight(),
                "nGenDaughters": Hist.new.Reg(
                    100, 0, 100, name="nGenDaughters", label="Number of gen daughters"
                ).Weight(),
                "nGenDaughtersMuons": Hist.new.Reg(
                    100,
                    0,
                    100,
                    name="nGenDaughtersMuons",
                    label="Number of gen muon daughters",
                ).Weight(),
                "nGenDaughtersElectrons": Hist.new.Reg(
                    100,
                    0,
                    100,
                    name="nGenDaughtersElectrons",
                    label="Number of gen electron daughters",
                ).Weight(),
                "SUEP_nconst": Hist.new.Reg(
                    100, 0, 100, name="SUEP_nconst", label="Number of constituents"
                ).Weight(),
                "SUEP_S1": Hist.new.Reg(
                    100, 0, 1, name="SUEP_S1", label="Sphericity"
                ).Weight(),
                "gentracks_minDeltaR": Hist.new.Reg(
                    1000, 0, 3.0, name="gentracks_minDeltaR", label="min $\Delta R$"
                ).Weight(),
                "reco_minDeltaR": Hist.new.Reg(
                    1000, 0, 3.0, name="reco_minDeltaR", label="min $\Delta R$"
                ).Weight(),
                "genW_pt_0": Hist.new.Reg(
                    400, 0, 400, name="genW_pt_0", label="Gen W $p_T$ [GeV]"
                ).Weight(),
                "genW_pt_1": Hist.new.Reg(
                    400, 0, 400, name="genW_pt_1", label="Gen W $p_T$ [GeV]"
                ).Weight(),
                "genW_pt_2": Hist.new.Reg(
                    400, 0, 400, name="genW_pt_2", label="Gen W $p_T$ [GeV]"
                ).Weight(),
                "genW_pt_3": Hist.new.Reg(
                    400, 0, 400, name="genW_pt_3", label="Gen W $p_T$ [GeV]"
                ).Weight(),
                "genW_pt_4": Hist.new.Reg(
                    400, 0, 400, name="genW_pt_4", label="Gen W $p_T$ [GeV]"
                ).Weight(),
                "lepton_pt": Hist.new.Reg(
                    400, 0, 400, name="lepton_pt", label="Lepton $p_T$ [GeV]"
                ).Weight(),
                "muon_pt_1": Hist.new.Reg(
                    400, 0, 400, name="muon_pt_1", label="Muon $p_T$ [GeV]"
                ).Weight(),
                "electron_pt_1": Hist.new.Reg(
                    400, 0, 400, name="electron_pt_1", label="Electron $p_T$ [GeV]"
                ).Weight(),
            }
        )

        # gen weights
        if self.isMC:
            blank_output["gensumweight"] += ak.sum(events.genWeight)
        else:
            genWeight = np.ones(len(events))
            events = ak.with_field(events, genWeight, "genWeight")

        output = {}

        # run the analysis
        output_nom = deepcopy(blank_output)
        _, output_nom = self.analysis(events, output_nom)
        output["nominal"] = output_nom

        # run the analysis with the systematic variations applied
        if self.isMC and self.do_syst:

            # for these, we need to re-run the whole analysis
            variations = ["track_down", "track_down_mod"]
            for variation in variations:
                output_var = deepcopy(blank_output)
                _, output_var = self.analysis(
                    events,
                    output=output_var,
                    variation=variation,
                )
                output[variation] = output_var

        return {dataset: output}

    def postprocess(self, accumulator):
        return accumulator
