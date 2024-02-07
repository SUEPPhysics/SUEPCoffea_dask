import awkward as ak
import numpy as np
import vector
from vector._methods import LorentzMomentum, Planar

from workflows.SUEP_utils import sphericity


def selectByLeptons(self, events, extraColls=[], lepveto=False):
    ###lepton selection criteria--4momenta collection for plotting
    muons = ak.zip(
        {
            "pt": events.Muon.pt,
            "eta": events.Muon.eta,
            "phi": events.Muon.phi,
            "mass": events.Muon.mass,
            "pdgID": events.Muon.pdgId,
            "ID": (
                ak.values_astype(events.Muon.tightId, np.int32)
                + ak.values_astype(events.Muon.mediumId, np.int32)
                + ak.values_astype(events.Muon.looseId, np.int32)
            ),  # 1=loose, 2=med, 3=tight cutbased
            "IDMVA": events.Muon.mvaId,  # 1=MvaLoose, 2=MvaMedium, 3=MvaTight, 4=MvaVTight, 5=MvaVVTight
            "iso": events.Muon.pfRelIso04_all,  # events.Muon.pfIsoId <--- using the rel iso float value rather than the WPs, mainly for consistency with electrons
            "isoMVA": events.Muon.mvaTTH,  # TTH MVA lepton ID score (true leptons peak at 1)
            "miniIso": events.Muon.miniPFRelIso_all,
            "dxy": events.Muon.dxy,
            "dz": events.Muon.dz,
        },
        with_name="Momentum4D",
    )

    electrons = ak.zip(
        {
            "pt": events.Electron.pt,
            "eta": events.Electron.eta,
            "phi": events.Electron.phi,
            "mass": events.Electron.mass,
            "pdgID": events.Electron.pdgId,
            "ID": events.Electron.cutBased,  # cut-based ID Fall17 V2 (0:fail, 1:veto, 2:loose, 3:medium, 4:tight)
            "IDMVA": (
                ak.values_astype(events.Electron.mvaFall17V2Iso_WP80, np.int32)
                + ak.values_astype(events.Electron.mvaFall17V2Iso_WP90, np.int32)
                + ak.values_astype(events.Electron.mvaFall17V2Iso_WPL, np.int32)
            ),  # 1=loose WP, 2=WP90, 3=WP80 electron ID MVA (assuming they are all subsets of one another--should confirm!)
            "iso": events.Electron.pfRelIso03_all,
            "isoMVA": events.Electron.mvaTTH,  # TTH MVA lepton ID score (true leptons peak at 1)
            "miniIso": events.Electron.miniPFRelIso_all,
            "dxy": events.Electron.dxy,
            "dz": events.Electron.dz,
        },
        with_name="Momentum4D",
    )

    ###  Some very simple selections on ID ###
    ###  Muons: loose ID + dxy dz cuts mimicking the medium prompt ID https://twiki.cern.ch/twiki/bin/viewauth/CMS/SWGuideMuonIdRun2
    ###  Electrons: loose ID + dxy dz cuts for promptness https://twiki.cern.ch/twiki/bin/view/CMS/EgammaCutBasedIdentification
    if self.scouting == 1:
        cutMuons = (
            # (events.Muon.isGlobalMuon == 1 | events.Muon.isTrackerMuon == 1)
            (events.Muon.isGlobalMuon == 1)
            & (events.Muon.pt >= 10)
            & (abs(events.Muon.dxy) <= 0.02)
            & (abs(events.Muon.dz) <= 0.1)
            & (events.Muon.trkiso < 0.10)
            & (abs(events.Muon.eta) < 2.4)
        )
        cutElectrons = (
            (events.Electron.ID == 1)
            & (events.Electron.pt >= 15)
            & (
                abs(events.Electron.d0)
                < 0.05 + 0.05 * (abs(events.Electron.eta) > 1.479)
            )
            & (
                abs(events.Electron.dz)
                < 0.10 + 0.10 * (abs(events.Electron.eta) > 1.479)
            )
            & ((abs(events.Electron.eta) < 1.444) | (abs(events.Electron.eta) > 1.566))
            & (abs(events.Electron.eta) < 2.5)
        )
    else:
        # "Loose" = ZH criteria, which we must impose for orthogonality
        cutLooseMuons = (
            (events.Muon.looseId)
            & (events.Muon.pt >= 10)
            & (abs(events.Muon.dxy) <= 0.02)
            & (abs(events.Muon.dz) <= 0.1)
            & (events.Muon.pfIsoId >= 2)
            & (abs(events.Muon.eta) < 2.4)
        )
        cutLooseElectrons = (
            (events.Electron.cutBased >= 2)
            & (events.Electron.pt >= 15)
            & (events.Electron.mvaFall17V2Iso_WP90)
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

        cutMuons = (
            cutLooseMuons
            & (events.Muon.tightId)
            & (events.Muon.pfIsoId >= 5)  # PFIsoVeryTight, aka PF rel iso < 0.1
            & (abs(events.Muon.dz) <= 0.05)
        )
        cutElectrons = cutLooseElectrons & (events.Electron.mvaFall17V2Iso_WP80)

    ### Apply the cuts
    # Object selection. selMuons contain only the events that are filtered by cutMuons criteria.
    selMuons = muons[cutMuons]
    selElectrons = electrons[cutElectrons]

    cutHasOneMuon = (ak.num(selMuons, axis=1) == 1) & (
        ak.max(selMuons.pt, axis=1, mask_identity=False) >= 30
    )
    cutHasOneElec = (ak.num(selElectrons, axis=1) == 1) & (
        ak.max(selElectrons.pt, axis=1, mask_identity=False) >= 35
    )
    cutOneLep = (ak.num(selElectrons, axis=1) + ak.num(selMuons, axis=1)) < 2
    cutHasOneLep = ((cutHasOneMuon) | (cutHasOneElec)) & cutOneLep

    ### Cut the events, also return the selected leptons for operation down the line
    events = events[cutHasOneLep]
    selElectrons = selElectrons[cutHasOneLep]
    selMuons = selMuons[cutHasOneLep]

    selLeptons = ak.concatenate([selElectrons, selMuons], axis=1)

    return events, selLeptons  # , [coll[cutHasTwoLeps] for coll in extraColls]


def MET_delta_phi(x, MET):
    # define 4-vectors for MET (x already 4-vector)
    MET_4v = ak.zip(
        {
            "pt": MET.pt,
            "eta": 0,
            "phi": MET.phi,
            "mass": 0,
        },
        with_name="Momentum4D",
    )

    signed_dphi = MET_4v.deltaphi(x)
    abs_dphi = np.abs(signed_dphi)
    return abs_dphi


def W_kinematics(lepton, MET):
    # mT calculation -- m1 = m2 = 0, e.g. MT for W uses mass_lepton = mass_MET = 0
    phi = MET_delta_phi(lepton, MET)
    W_mt_2 = (
        2
        * np.abs(lepton.pt)
        * np.abs(MET.pt)
        * (1 - np.cos(phi))  # from PDG review on kinematics, eq 38.61
    )
    W_mt = np.sqrt(W_mt_2)

    # pT calculation
    W_ptx = lepton.px + MET.px
    W_pty = lepton.py + MET.py

    W_pt = np.sqrt(W_ptx**2 + W_pty**2)

    # phi calculation
    W_phi = np.arctan2(W_pty, W_ptx)

    return W_mt[:, 0], W_pt[:, 0], W_phi[:, 0]


def HighestPTMethod(
    self,
    events,
    indices,
    tracks,
    jets,
    clusters,
    output,
    out_label=None,
):
    #####################################################################################
    # ---- Highest pT Jet (PT)
    # SUEP defined as the highest pT jet
    #####################################################################################'

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
    eigs = sphericity(SUEP_tracks_b, 1.0)  # Set r=1.0 for IRC safe
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
        MET_delta_phi(SUEP_cand, events[indices].CaloMET),
    )
    output["vars"].loc(
        indices,
        "deltaPhi_SUEP_PuppiMET" + out_label,
        MET_delta_phi(SUEP_cand, events[indices].PuppiMET),
    )
    output["vars"].loc(
        indices,
        "deltaPhi_SUEP_MET" + out_label,
        MET_delta_phi(SUEP_cand, events[indices].MET),
    )
