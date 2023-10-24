import awkward as ak

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
        cutMuons = (
            (events.Muon.looseId)
            & (events.Muon.pt >= 10)
            & (abs(events.Muon.dxy) <= 0.02)
            & (abs(events.Muon.dz) <= 0.1)
            & (events.Muon.pfIsoId >= 2)
            & (abs(events.Muon.eta) < 2.4)
        )
        cutElectrons = (
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

    ### Apply the cuts
    # Object selection. selMuons contain only the events that are filtered by cutMuons criteria.
    selMuons = muons[cutMuons]
    selElectrons = electrons[cutElectrons]

    cutHasOneMuon = (ak.num(selMuons, axis=1) == 1) & (
        ak.max(selMuons.pt, axis=1, mask_identity=False) >= 25
    )
    cutHasOneElec = (ak.num(selElectrons, axis=1) == 1) & (
        ak.max(selElectrons.pt, axis=1, mask_identity=False) >= 25
    )
    cutOneLep = (
        ak.num(selElectrons, axis=1) + ak.num(selMuons, axis=1)
    ) < 2  # is this removing events with BOTH muons and electrons?
    cutHasOneLep = ((cutHasOneMuon) | (cutHasOneElec)) & cutOneLep

    ### Cut the events, also return the selected leptons for operation down the line
    events = events[cutHasOneLep]
    selElectrons = selElectrons[cutHasOneLep]
    selMuons = selMuons[cutHasOneLep]

    selLeptons = ak.concatenate([selElectrons, selMuons], axis=1)

    return events, selLeptons  # , [coll[cutHasTwoLeps] for coll in extraColls]


def TopPTMethod(
    self,
    indices,
    tracks,
    jets,
    clusters,
    output,
    out_label=None,
):
    #####################################################################################
    # ---- Top pT Jet (PT)
    # SUEP defines as the top pT jet
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
    output["vars"].loc(indices, "SUEP_nconst_TopPT" + out_label, ak.num(SUEP_tracks_b))
    output["vars"].loc(
        indices, "SUEP_pt_avg_b_TopPT" + out_label, ak.mean(SUEP_tracks_b.pt, axis=-1)
    )
    output["vars"].loc(
        indices, "SUEP_S1_TopPT" + out_label, 1.5 * (eigs[:, 1] + eigs[:, 0])
    )

    # unboost for these
    SUEP_tracks = SUEP_tracks_b.boost_p4(SUEP_cand)
    output["vars"].loc(
        indices, "SUEP_pt_avg_TopPT" + out_label, ak.mean(SUEP_tracks.pt, axis=-1)
    )
    output["vars"].loc(indices, "SUEP_pt_TopPT" + out_label, SUEP_cand.pt)
    output["vars"].loc(indices, "SUEP_eta_TopPT" + out_label, SUEP_cand.eta)
    output["vars"].loc(indices, "SUEP_phi_TopPT" + out_label, SUEP_cand.phi)
    output["vars"].loc(indices, "SUEP_mass_TopPT" + out_label, SUEP_cand.mass)

    # Calculate gen SUEP and candidate SUEP differences
    SUEP_genEta_diff_TopPT = (
        output["vars"]["SUEP_eta_TopPT" + out_label]
        - output["vars"]["SUEP_genEta" + out_label]
    )
    SUEP_genPhi_diff_TopPT = (
        output["vars"]["SUEP_phi_TopPT" + out_label]
        - output["vars"]["SUEP_genPhi" + out_label]
    )
    SUEP_genR_diff_TopPT = (
        SUEP_genEta_diff_TopPT**2 + SUEP_genPhi_diff_TopPT**2
    ) ** 0.5
    output["vars"]["SUEP_deltaEtaGen_TopPT" + out_label] = SUEP_genEta_diff_TopPT
    output["vars"]["SUEP_deltaPhiGen_TopPT" + out_label] = SUEP_genPhi_diff_TopPT
    output["vars"]["SUEP_deltaRGen_TopPT" + out_label] = SUEP_genR_diff_TopPT
    output["vars"].loc(
        indices,
        "SUEP_deltaMassGen_TopPT" + out_label,
        (SUEP_cand.mass - output["vars"]["SUEP_genMass" + out_label][indices]),
    )
    output["vars"].loc(
        indices,
        "SUEP_deltaPtGen_TopPT" + out_label,
        (SUEP_cand.pt - output["vars"]["SUEP_genPt" + out_label][indices]),
    )
