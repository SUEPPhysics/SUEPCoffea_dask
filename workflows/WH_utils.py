import awkward as ak
import numpy as np
import vector


def getAK4Jets(Jets, lepton):
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
            "btag": Jets.btagDeepFlavB,
            "jetId": Jets.jetId,
            "hadronFlavour": Jets.hadronFlavour,
            "qgl": Jets.qgl,
        },
        with_name="Momentum4D",
    )
    # jet pt cut, eta cut, and minimum separation from lepton
    jet_awk_Cut = (
        (Jets_awk.pt > 30)
        & (abs(Jets_awk.eta) < 2.4)
        & (Jets_awk.deltaR(lepton[:, 0]) >= 0.4)
    )
    Jets_correct = Jets_awk[jet_awk_Cut]

    return Jets_correct


def getGenTracks(events):
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


def getTracks(events, lepton=None, leptonIsolation=None):
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
        tracks = tracks[(tracks.deltaR(lepton[:, 0]) >= leptonIsolation)]

    return tracks, Cleaned_cands


def getLeptons(events):

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
            "charge": events.Muon.pdgId / (-13),
            "tightId": events.Muon.tightId,
            "pfIsoId": events.Muon.pfIsoId,
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
            "charge": events.Electron.pdgId / (-11),
            "mvaFall17V2Iso_WP80": events.Electron.mvaFall17V2Iso_WP80,
        },
        with_name="Momentum4D",
    )

    leptons = ak.concatenate([muons, electrons], axis=1)

    return muons, electrons, leptons


def getLooseLeptons(events):
    """
    These leptons follow EXACTLY the ZH definitions, so that we can impose
    orthogonality between the ZH, offline, and WH selections.
    """

    muons, electrons, leptons = getLeptons(events)

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
        & (abs(events.Electron.dxy) < 0.05 + 0.05 * (abs(events.Electron.eta) > 1.479))
        & (abs(events.Electron.dz) < 0.10 + 0.10 * (abs(events.Electron.eta) > 1.479))
        & ((abs(events.Electron.eta) < 1.444) | (abs(events.Electron.eta) > 1.566))
        & (abs(events.Electron.eta) < 2.5)
    )

    looseMuons = muons[cutLooseMuons]
    looseElectrons = electrons[cutLooseElectrons]
    looseLeptons = ak.concatenate([looseMuons, looseElectrons], axis=1)

    return looseMuons, looseElectrons, looseLeptons


def getTightLeptons(events):
    """
    These leptons are the ones that will be used for the WH analysis.
    """

    looseMuons, looseElectrons, _ = getLooseLeptons(events)

    # tighter lepton ID
    cutTightMuons = (
        (looseMuons.tightId)
        & (looseMuons.pfIsoId >= 5)  # PFIsoVeryTight, aka PF rel iso < 0.1
        & (abs(looseMuons.dz) <= 0.05)
        & (looseMuons.pt >= 30)
    )
    cutTightElectrons = (looseElectrons.mvaFall17V2Iso_WP80) & (looseElectrons.pt >= 35)

    tightMuons = looseMuons[cutTightMuons]
    tightElectrons = looseElectrons[cutTightElectrons]
    tightLeptons = ak.concatenate([tightMuons, tightElectrons], axis=1)

    return tightMuons, tightElectrons, tightLeptons

def getTrigObj(events):
    trigObj = ak.zip(
        {
            "pt": events.TrigObj.pt,
            "eta": events.TrigObj.eta,
            "phi": events.TrigObj.phi,
            "filterBits": events.TrigObj.filterBits,
        },
        with_name="Momentum4D",
    )
    return trigObj

def triggerSelection(events, era:str, isMC:bool, output=None, out_label=None):
        """
        Applies trigger, returns events.
        Trigger single muon and EGamma; optionally updates the cutflows.
        """

        # photon trigger
        if era == "2016" or era == "2016apv":
            triggerPhoton = events.HLT.Photon175
        elif era == "2017" or era == "2018":
            triggerPhoton = events.HLT.Photon200

        # electron trigger
        if era == "2017" and (not isMC):
            # data 2017 is special <3<3
            # https://twiki.cern.ch/twiki/bin/view/CMS/EgHLTRunIISummary#2017
            # TODO: need to implement this
            triggerElectron = (
                events.HLT.Ele115_CaloIdVT_GsfTrkIdT
            )
        else:
            triggerElectron = (
                events.HLT.Ele32_WPTight_Gsf
                | events.HLT.Ele115_CaloIdVT_GsfTrkIdT
            )

        # muon trigger
        triggerSingleMuon = events.HLT.IsoMu27 | events.HLT.Mu50

        # this is just for cutflow
        if output:
            output["cutflow_triggerSingleMuon" + out_label] += ak.sum(
                events[triggerSingleMuon].genWeight
            )
            output["cutflow_triggerEGamma" + out_label] += ak.sum(events[triggerPhoton | triggerElectron].genWeight)

        events = events[triggerElectron | triggerPhoton | triggerSingleMuon]
        return events

def orthogonalitySelection(events):
    """
    This function is used to impose orthogonality between the ZH, offline, and WH selections.
    """

    # follow ZH and offline lepton definitions
    looseMuons, looseElectrons, _ = getLooseLeptons(events)

    # offline selection
    cutAnyElecs = (ak.num(looseElectrons, axis=1) > 0) & (
        ak.max(looseElectrons.pt, axis=1, mask_identity=False) >= 25
    )
    cutAnyMuons = (ak.num(looseMuons, axis=1) > 0) & (
        ak.max(looseMuons.pt, axis=1, mask_identity=False) >= 25
    )
    cutAnyLeps = cutAnyElecs | cutAnyMuons

    # ZH selection
    cutHasTwoMuons = (
        (ak.num(looseMuons, axis=1) == 2)
        & (ak.max(looseMuons.pt, axis=1, mask_identity=False) >= 25)
        & (ak.sum(looseMuons.charge, axis=1) == 0)
    )
    cutHasTwoElecs = (
        (ak.num(looseElectrons, axis=1) == 2)
        & (ak.max(looseElectrons.pt, axis=1, mask_identity=False) >= 25)
        & (ak.sum(looseElectrons.charge, axis=1) == 0)
    )
    cutTwoLeps = (ak.num(looseElectrons, axis=1) + ak.num(looseMuons, axis=1)) < 4
    cutHasTwoLeps = ((cutHasTwoMuons) | (cutHasTwoElecs)) & cutTwoLeps

    # apply orthogonality condition
    events = events[cutAnyLeps & ~cutHasTwoLeps]

    return events


def qualityFiltersSelection(events, era: str):
    ### Apply MET filter selection (see https://twiki.cern.ch/twiki/bin/viewauth/CMS/MissingETOptionalFiltersRun2)
    if era == "2018" or era == "2017":
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
    if era == "2016" or era == "2016apv":
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
