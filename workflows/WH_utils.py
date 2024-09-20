import awkward as ak
import numpy as np
import vector
from workflows.CMS_corrections.HEM_utils import jetHEMFilter

def getGenModel(events):
    """
    Central signal samples are not split by parameters, so we need to extract the model from the input file
    and save it in the dataframe.
    The expected structure is {'SUEP_mS125.000_mPhi1.000_T0.250_modeleptonic': True} for each event.
    :param events: awkward array of events
    :return: genModel for each event, as a list.
    """
    if not hasattr(events, "GenModel"):
        raise ValueError("GenModel not found in events, please check the input file.")
    genModels = []
    for (
        genModelInfo
    ) in (
        events.GenModel
    ):  # I can't figure out a way to do this with AwkwardArrays, so I'm doing it with two for loops, may I be forgiven
        genModel = []
        genModelInfo = genModelInfo.tolist()  # this actually becomes a dictionary
        for g, v in genModelInfo.items():
            if v:
                genModel.append(g)
        if len(genModel) != 1:
            raise ValueError(f"Expected one genModel per event.")
        genModels.append(genModel[0])
    return genModels


def getAK4Jets(Jets, runs, iso=None, isMC: bool = 1):
    """
    Create awkward array of jets. Applies basic selections.
    Returns: awkward array of dimensions (events x jets x 4 momentum)
    """
    if isMC:
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
                "area": Jets.area,
            },
            with_name="Momentum4D",
        )
    else:
        Jets_awk = ak.zip(
            {
                "pt": Jets.pt,
                "eta": Jets.eta,
                "phi": Jets.phi,
                "mass": Jets.mass,
                "btag": Jets.btagDeepFlavB,
                "jetId": Jets.jetId,
                "qgl": Jets.qgl,
                "area": Jets.area,
                "raw_factor": Jets.rawFactor,
            },
            with_name="Momentum4D",
        )
    # jet pt cut, eta cut, and jet ID
    jet_awk_Cut = (
        (Jets_awk.pt > 30) & (abs(Jets_awk.eta) < 2.4) & (0 < (Jets_awk.jetId & 0b010))
    )
    # and minimum separation from some object (e.g. lepton)
    if iso is not None:
        jet_awk_Cut = jet_awk_Cut & (Jets_awk.deltaR(iso) >= 0.4)
    # and HEM filter
    jetHEMCut, _ = jetHEMFilter(None, Jets, runs) 
    jet_awk_Cut = jet_awk_Cut & jetHEMCut

    Jets_correct = Jets_awk[jet_awk_Cut]

    return Jets_correct


def getGenPart(events):
    genParts = events.GenPart
    genParts = ak.zip(
        {
            "pt": genParts.pt,
            "eta": genParts.eta,
            "phi": genParts.phi,
            "mass": genParts.mass,
            "pdgID": genParts.pdgId,
            "status": genParts.status,
            "genPartIdxMother": genParts.genPartIdxMother,
            "statusFlags": genParts.statusFlags,
        },
        with_name="Momentum4D",
    )
    return genParts


def getGenW(events):
    """
    Get the gen-level W boson, lastCopy (statusFlag 13)
    """
    genParticles = getGenPart(events)
    genW = genParticles[
        (abs(genParticles.pdgID) == 24) & (0 < (genParticles.statusFlags & (1 << 13)))
    ]
    return genW


def getGenDarkPseudoscalars(events):
    """
    Get the gen-level dark pseudoscalar particles (phi's) produced by the scalar (S).
    This depends on how you set up your signal samples. This function assumes the SUEP WH layout in e.g. https://gitlab.cern.ch/cms-exo-mci/EXO-MCsampleRequests/-/merge_requests/205/diffs
    """

    genParticles = getGenPart(events)
    darkPseudoscalarParticles = genParticles[genParticles.pdgID == 999999]

    return darkPseudoscalarParticles


def getTracks(events, iso_object=None, isolation_deltaR=0):
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
        & (events.lostTracks.pt >= 1)
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

    if isolation_deltaR > 0:
        # Sorting out the tracks that overlap with some object
        tracks = tracks[(tracks.deltaR(iso_object) >= isolation_deltaR)]
        Cleaned_cands = Cleaned_cands[(Cleaned_cands.deltaR(iso_object) >= isolation_deltaR)]
        Lost_Tracks_cands = Lost_Tracks_cands[
            (Lost_Tracks_cands.deltaR(iso_object) >= isolation_deltaR)
        ]

    return tracks, Cleaned_cands, Lost_Tracks_cands


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
            "pfIsoId": ak.ones_like(events.Electron.pt) * -1,
        },
        with_name="Momentum4D",
    )

    leptons = ak.concatenate([muons, electrons], axis=1)

    return muons, electrons, leptons


def getLooseLeptons(events):
    """
    These leptons follow EXACTLY the ZH definitions, so that we can impose
    orthogonality between the ZH, offline, and WH selections.
    WARNING: DO NOT CHANGE THIS
    """

    muons, electrons, _ = getLeptons(events)

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


def getLooseNotTightLeptons(events):

    looseMuons, looseElectrons, _ = getLooseLeptons(events)

    cutTightMuons = (
        (looseMuons.tightId)
        & (looseMuons.pfIsoId >= 5)  # PFIsoVeryTight, aka PF rel iso < 0.1
        & (abs(looseMuons.dz) <= 0.05)
        & (looseMuons.pt >= 30)
    )
    cutTightElectrons = (looseElectrons.mvaFall17V2Iso_WP80) & (looseElectrons.pt >= 35)

    looseNotTightMuons = looseMuons[~cutTightMuons]
    looseNotTightElectrons = looseElectrons[~cutTightElectrons]
    looseNotTightLeptons = ak.concatenate(
        [looseNotTightMuons, looseNotTightElectrons], axis=1
    )

    return looseNotTightMuons, looseNotTightElectrons, looseNotTightLeptons
    

def getPhotons(events, isMC: bool = 1):
    """
    Get photons.
    """
    if isMC:
        photons = ak.zip(
            {
                "pt": events.Photon.pt,
                "eta": events.Photon.eta,
                "phi": events.Photon.phi,
                "mass": events.Photon.mass,
                "pixelSeed": events.Photon.pixelSeed,
                "electronVeto": events.Photon.electronVeto,
                "hoe": events.Photon.hoe,
                "r9": events.Photon.r9,
                "mvaID": events.Photon.mvaID,
                "pfRelIso03_all": events.Photon.pfRelIso03_all,
                "cutBased": events.Photon.cutBased,
                "isScEtaEB": events.Photon.isScEtaEB,
                "isScEtaEE": events.Photon.isScEtaEE,
                "sieie": events.Photon.sieie,
                "genPartFlav ": events.Photon.genPartFlav,
            },
            with_name="Momentum4D",
        )
    else:
        photons = ak.zip(
            {
                "pt": events.Photon.pt,
                "eta": events.Photon.eta,
                "phi": events.Photon.phi,
                "mass": events.Photon.mass,
                "pixelSeed": events.Photon.pixelSeed,
                "electronVeto": events.Photon.electronVeto,
                "hoe": events.Photon.hoe,
                "r9": events.Photon.r9,
                "mvaID": events.Photon.mvaID,
                "pfRelIso03_all": events.Photon.pfRelIso03_all,
                "cutBased": events.Photon.cutBased,
                "isScEtaEB": events.Photon.isScEtaEB,
                "isScEtaEE": events.Photon.isScEtaEE,
                "sieie": events.Photon.sieie,
            },
            with_name="Momentum4D",
        )

    cutPhotons = (
        (events.Photon.mvaID_WP90)
        & (abs(events.Photon.eta) <= 2.5)
        & (events.Photon.electronVeto)
        & (events.Photon.pt >= 30)
        & (events.Photon.pfRelIso03_all < 0.15)
    )

    photons = photons[cutPhotons]

    return photons


def getGenPhotons(events):
    """
    Get final state photons in the gen-level.
    """
    parts = getGenPart(events)
    photons = parts[(abs(parts.pdgID) == 22) & (parts.statusFlags & (1 << 8) != 0) & (parts.statusFlags & (1 << 13) != 0)]
    return photons


def storeGenPhotonStuff(events, output):

    genPhotons = getGenPhotons(events)
    deltaR_genPhotons_recoPhotons = genPhotons.deltaR(events.WH_gamma)
    output["vars"]["minDeltaRGenRecoPhotons"] = ak.min(ak.fill_none(ak.pad_none(deltaR_genPhotons_recoPhotons, 1, clip=False), -999), axis=1)

    return output


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


def genSelection(events, sample: str):
    """
    Gen-level selections.
    The WJets inclusive sample needs to be cut at W gen pT of 100 GeV in order to be stitched together with the WJets pT binned samples.
    """

    if "WJetsToLNu_TuneCP5_13TeV-amcatnloFXFX-pythia8" in sample:
        pt = events.LHE.Vpt
        cut = pt < 100
        events = events[cut]

    return events


def triggerSelection(
    events, sample: str, era: str, isMC: bool, output=None, out_label=None
):
    """
    Applies trigger, returns events.
    Trigger single muon and EGamma; optionally updates the cutflows.
    """

    # muon trigger
    triggerSingleMuon = events.HLT.IsoMu27 | events.HLT.Mu50

    # photon trigger
    if era == "2016" or era == "2016apv":
        triggerPhoton = events.HLT.Photon175
    elif era == "2017" or era == "2018":
        triggerPhoton = events.HLT.Photon200

    if era == "2017" and (not isMC) and ("SingleElectron" in sample):
        # data 2017 is special <3<3
        # https://twiki.cern.ch/twiki/bin/view/CMS/EgHLTRunIISummary#2017

        # remove events in the SingleMuon dataset
        events = events[~triggerSingleMuon]
        temp_trig = events.HLT.Photon200 | events.HLT.Ele115_CaloIdVT_GsfTrkIdT

        # Grab events associated with electron trigger: https://cms-nanoaod-integration.web.cern.ch/integration/master-102X/mc102X_doc.html#TrigObj
        filts = (events.TrigObj.id == 11) & ((events.TrigObj.filterBits & 1024) == 1024)
        events = events[(ak.num(events.TrigObj[filts]) > 0) | temp_trig]
        temp_trig = events.HLT.Photon200 | events.HLT.Ele115_CaloIdVT_GsfTrkIdT

        # grab prefiltered events
        trig_obs = getTrigObj(events)
        muons, electrons, leptons = getLeptons(events)
        dR = ak.Array([[] for _ in range(len(events))])

        for i in range(
            ak.max(ak.num(trig_obs))
        ):  # only loop through the trigger objects that pass the filters
            mask = ak.mask(trig_obs, (ak.num(trig_obs) > i))
            dR_masked = ak.Array(electrons.deltaR(mask[:, i]))
            dR_masked = ak.Array(x if x is not None else [] for x in dR_masked)
            dR = ak.concatenate([dR, dR_masked], axis=-1)

        # remove events that do not have a trig object within dR of 0.1 of an electron
        dR = ak.where(ak.num(dR, axis=-1) == 0, ak.Array([[1.0]]), dR)
        events = events[(ak.min(dR, axis=-1) < 0.1) | temp_trig]

        # Do cutflow and return events
        if output:
            output["cutflow_triggerEGamma" + out_label] += ak.sum(events.genWeight)
        return events
    elif era == "2016":
        triggerElectron = (
            events.HLT.Ele27_WPTight_Gsf | events.HLT.Ele115_CaloIdVT_GsfTrkIdT
        )
    else:
        triggerElectron = (
            events.HLT.Ele32_WPTight_Gsf | events.HLT.Ele115_CaloIdVT_GsfTrkIdT
        )
    # this is just for cutflow
    if output:
        output["cutflow_triggerSingleMuon" + out_label] += ak.sum(
            events[triggerSingleMuon].genWeight
        )
        output["cutflow_triggerEGamma" + out_label] += ak.sum(
            events[triggerPhoton | triggerElectron].genWeight
        )

    # Apply selection on events
    if isMC:
        events = events[triggerElectron | triggerPhoton | triggerSingleMuon]
    else:
        if "SingleMuon" in sample:
            events = events[triggerSingleMuon]
        elif ("SingleElectron" in sample) or ("EGamma" in sample):
            events = events[(triggerElectron | triggerPhoton) & (~triggerSingleMuon)]
        else:
            events = events[triggerElectron | triggerPhoton | triggerSingleMuon]

    return events


def getGammaTriggerBits(events, era: str):
    """
    Form the photon trigger bits.
    """

    if era == '2018':
        gammaTriggerBits = (events.HLT.Photon200) + (events.HLT.Photon120_R9Id90_HE10_IsoM)*2 + (events.HLT.Photon75_R9Id90_HE10_IsoM)*4 + (events.HLT.Photon50_R9Id90_HE10_IsoM)*8
    elif era == '2017':
        gammaTriggerBits = (events.HLT.Photon200) + (events.HLT.Photon165_R9Id90_HE10_IsoM)*2 + (events.HLT.Photon120_R9Id90_HE10_IsoM)*4 + (events.HLT.Photon75_R9Id90_HE10_IsoM)*8 + (events.HLT.Photon50_R9Id90_HE10_IsoM)*16 + (events.HLT.Photon30_HoverELoose)*32 + (events.HLT.Photon20_HoverELoose)*64  
    elif era == '2016' or era == '2016apv':
        gammaTriggerBits = (events.HLT.Photon175) + (events.HLT.Photon165_HE10)*2 + (events.HLT.Photon165_R9Id90_HE10_IsoM)*4 + (events.HLT.Photon90_R9Id90_HE10_IsoM)*8 + (events.HLT.Photon75_R9Id90_HE10_IsoM)*16 + (events.HLT.Photon50_R9Id90_HE10_IsoM)*32 + (events.HLT.Photon36_R9Id90_HE10_IsoM)*64 + (events.HLT.Photon30_R9Id90_HE10_IsoM)*128 + (events.HLT.Photon22_R9Id90_HE10_IsoM)*256
    else:
        raise ValueError(f"Invalid era: {era}")
    
    return gammaTriggerBits


def prescaledGammaTriggersSelection(events, era: str, isMC: bool):
    # https://twiki.cern.ch/twiki/bin/viewauth/CMS/EgHLTRunIISummary?sortcol=2;table=2;up=0

    gammaTriggerBits = getGammaTriggerBits(events, era)
    events = ak.with_field(events, gammaTriggerBits, "WH_gammaTriggerBits")
    events = events[(events.WH_gammaTriggerBits > 0)]
    
    gammaTriggerUnprescaleWeight = ak.zeros_like(events.genWeight)
    if era == '2018':
        mask_Photon200 = ((events.HLT.Photon200 == 1) & (events.WH_gamma.pt > 230))
        mask_Photon120 = ((events.HLT.Photon120_R9Id90_HE10_IsoM == 1) & (events.WH_gamma.pt > 135) & (events.WH_gamma.pt < 230))
        mask_Photon75 = ((events.HLT.Photon75_R9Id90_HE10_IsoM == 1) & (events.WH_gamma.pt > 80) & (events.WH_gamma.pt < 135))
        mask_Photon50 = ((events.HLT.Photon50_R9Id90_HE10_IsoM == 1) & (events.WH_gamma.pt > 60) & (events.WH_gamma.pt < 80))
        mask = mask_Photon200 | mask_Photon120 | mask_Photon75 | mask_Photon50
        if not isMC:
            gammaTriggerUnprescaleWeight = ak.where(mask_Photon200, 1, gammaTriggerUnprescaleWeight)
            gammaTriggerUnprescaleWeight = ak.where(mask_Photon120, 59.96/7.44 , gammaTriggerUnprescaleWeight)
            gammaTriggerUnprescaleWeight = ak.where(mask_Photon75, 59.96/0.95, gammaTriggerUnprescaleWeight)
            gammaTriggerUnprescaleWeight = ak.where(mask_Photon50, 59.96/0.24, gammaTriggerUnprescaleWeight)
    else:
        # other eras are not supported yet
        raise ValueError(f"Invalid era: {era}")
    
    events = ak.with_field(events, gammaTriggerUnprescaleWeight, "WH_gammaTriggerUnprescaleWeight")
    events = events[mask]
    return events

def doubleCountingGenPhotonsSelection(events, sample: str):
    """
    Orthogonalize the QCD_HT samples to the GJets_HT samples.
    Should only matters for gamma+jets control region.f
    """

    if 'QCD_HT' in sample or 'QCD_Pt' in sample:
        mask = (
            (events.GenPart.status==1) & 
            ((events.GenPart.statusFlags&64) == 0) & 
            (events.GenPart.pdgId==22) & 
            (events.GenPart.pt >= 25) &  
            (abs(events.GenPart.eta) < 2.5)
        )
        doubleCountedPhotons = events.GenPart[mask]
        no_doubleCountedPhotons = (ak.num(doubleCountedPhotons) == 0)
        events = ak.with_field(events, no_doubleCountedPhotons, "WH_no_doubleCountedPhotons")

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


def VRGJOrthogonalitySelection(events, era:str):
    """
    Stay orthogonal to ZH and WH via their respective lepton selections.
    """

    # follow ZH and offline lepton definitions
    looseMuons, looseElectrons, _ = getLooseLeptons(events)

    # ZH orthogonality
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
    events = events[~cutHasTwoLeps]

    # WH orthogonality
    _, _, tightLeptons = getTightLeptons(events)
    cutTightLeptons = ak.num(tightLeptons) == 0
    events = events[cutTightLeptons]

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
    elif era == "2016" or era == "2016apv":
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
    else:
        raise ValueError(f"Invalid era: {era}")
    return events[cutAnyFilter]


def oneTightLeptonSelection(events):

    _, _, tightLeptons = getTightLeptons(events)

    # require exactly one tight lepton
    leptonSelection = ak.num(tightLeptons) == 1
    events = events[leptonSelection]
    tightLeptons = tightLeptons[leptonSelection]

    if 'WH_lepton' in events.fields:
        raise Exception("WH_lepton already in events.")
    events = ak.with_field(events, tightLeptons[:, 0], "WH_lepton")

    return events 


def CRQCDSelection(events):
    """
    Defines the Control Region for QCD.
    No tight leptons (orthogonal to SR), and exactly one loose lepton,
    with same pT requirement that we impose on the tight lepton.
    """

    # orthogonality to SR
    _, _, tightLeptons = getTightLeptons(events)
    events = events[(ak.num(tightLeptons) == 0)]

    # for muons, we take all the 'tightMuon' selection, except for the isolation id
    looseMuons, _, _ = getLooseLeptons(events)
    cutAlmostTightMuons = (
        (looseMuons.tightId)
        & (abs(looseMuons.dz) <= 0.05)
        & (looseMuons.pt >= 30)
    )
    almostTightMuons = looseMuons[cutAlmostTightMuons]

    # for electrons, we go looser than the 'looseElectron' in the isolation by using WPL
    _, electrons, _ = getLeptons(events)
    cutAlmostLooseElectrons = (
        (events.Electron.cutBased >= 2)
        & (events.Electron.pt >= 35)
        & (events.Electron.mvaFall17V2Iso_WPL)
        & (abs(events.Electron.dxy) < 0.05 + 0.05 * (abs(events.Electron.eta) > 1.479))
        & (abs(events.Electron.dz) < 0.10 + 0.10 * (abs(events.Electron.eta) > 1.479))
        & ((abs(events.Electron.eta) < 1.444) | (abs(events.Electron.eta) > 1.566))
        & (abs(events.Electron.eta) < 2.5)
    )
    almostLooseElectrons = electrons[cutAlmostLooseElectrons]

    # require exactly one lepton
    CRQCDleptons = ak.concatenate([almostTightMuons, almostLooseElectrons], axis=1)
    oneLeptonSelection = ak.num(CRQCDleptons) >= 1
    events = events[oneLeptonSelection]
    CRQCDleptons = CRQCDleptons[oneLeptonSelection]

    if 'WH_lepton' in events.fields:
        raise Exception("WH_lepton already in events.")
    events = ak.with_field(events, CRQCDleptons[:,0], "WH_lepton")
    events = ak.with_field(events, ak.num(CRQCDleptons), "nCRQCDleptons")

    return events


def onePhotonSelection(events, isMC: bool = 1):

    # TODO we need to define an orthogonal region to the SR. Via trigger or tight lepton selection or MET or what?

    photons = getPhotons(events, isMC=isMC)

    # require exactly one photon
    photonSelection = ak.num(photons) == 1
    events = events[photonSelection]
    photons = photons[photonSelection]
    events = ak.with_field(events, photons[:, 0], "WH_gamma")

    return events


def make_MET_4v(MET):
    MET_4v = ak.zip(
        {
            "pt": MET.pt,
            "eta": 0,
            "phi": MET.phi,
            "mass": 0,
        },
        with_name="Momentum4D",
    )
    return MET_4v


def make_nu_4v(MET, pz=0):
    make_nu_4v = ak.zip(
        {
            "pt": MET.pt,
            "pz": pz,
            "phi": MET.phi,
            "mass": 0,
        },
        with_name="Momentum4D",
    )
    return make_nu_4v


def MET_delta_phi(x, MET):
    MET_4v = make_MET_4v(MET)
    signed_dphi = x.deltaphi(MET_4v)
    abs_dphi = np.abs(signed_dphi)
    return abs_dphi


def projectOnTransversePlane(objects):
    """
    Project the objects onto the transverse plane.
    """
    objects_4v = ak.zip(
        {
            "pt": objects.pt,
            "eta": ak.zeros_like(objects.eta),
            "phi": objects.phi,
            "mass": objects.mass,
        },
        with_name="Momentum4D",
    )
    return objects_4v


def make_Wt_4v(lepton, MET):
    """
    Make the W boson 4-vector from lepton and MET.
    """
    W_4v = projectOnTransversePlane(lepton) + make_MET_4v(MET)
    return W_4v


def calc_W_mt(lepton, MET):
    # mT calculation -- m1 = m2 = 0, e.g. MT for W uses mass_lepton = mass_MET = 0
    _deltaPhi_lepton_MET = MET_delta_phi(lepton, MET)
    _W_mt = np.sqrt(
        2
        * np.abs(lepton.pt)
        * np.abs(MET.pt)
        * (1 - np.cos(_deltaPhi_lepton_MET))  # from PDG review on kinematics, eq 38.61
    )
    return _W_mt

def getTaus(events):

    # see https://twiki.cern.ch/twiki/bin/view/CMS/TauIDRecommendationForRun2
    taus = events.Tau
    tau_selection = (
        (taus.pt > 20) &
        (abs(taus.eta) < 2.3) &
        (abs(taus.dz) < 0.2) &
        (taus.idAntiMu >= 1) &
        ((taus.idDeepTau2017v2p1VSe >= 8) | (taus.idDeepTau2017v2p1VSmu >= 2) | (taus.idDeepTau2017v2p1VSjet >= 8)) &
        (taus.decayMode != 5) & (taus.decayMode != 6)
    )
    return taus[tau_selection]

def storeTausInfo(events, output, sample: str = ''):

    taus = getTaus(events)
    output["vars"]['nTaus'] = ak.num(taus, axis=1).to_numpy().astype(np.int8)

    if 'DYJetsToLL' in sample:
        # this is for a particular study on DY, could be dropped in future
        for i in range(2):
            output["vars"][f"tau{i}_pt"] = ak.fill_none(ak.pad_none(taus.pt, i+1, clip=True)[:,i], -999).to_numpy().astype(np.float16)
            output["vars"][f"tau{i}_eta"] = ak.fill_none(ak.pad_none(taus.eta, i+1, clip=True)[:,i], -999).to_numpy().astype(np.float16)
            output["vars"][f"tau{i}_phi"] = ak.fill_none(ak.pad_none(taus.phi, i+1, clip=True)[:,i], -999).to_numpy().astype(np.float16)

def storeGenWInfo(events, output):

    ws = events.GenPart[(abs(events.GenPart.pdgId) == 24) & (events.GenPart.statusFlags & (1 << 13) > 0)]
    output["vars"][f"genW_pt"] = ak.fill_none(ak.pad_none(ws.pt, 1, axis=1, clip=True)[:,0], -999).to_numpy().astype(np.float16)
    output["vars"][f"genW_eta"] = ak.fill_none(ak.pad_none(ws.eta, 1, axis=1, clip=True)[:,0], -999).to_numpy().astype(np.float16)
    output["vars"][f"genW_phi"] = ak.fill_none(ak.pad_none(ws.phi, 1, axis=1, clip=True)[:,0], -999).to_numpy().astype(np.float16)

def storeGenZAndDaughtersInfo(events, output):

    z_idx = ak.local_index(events.GenPart, axis=1)[(abs(events.GenPart.pdgId) == 23) & (events.GenPart.statusFlags & (1 << 13) > 0)]
    zs = events.GenPart[z_idx]
    z_idx = ak.fill_none(
                ak.pad_none(
                    z_idx, 1, axis=1, clip=True
                ),
                -999,
            )[:,0]
    daughters = events.GenPart[events.GenPart.genPartIdxMother == z_idx]

    output["vars"][f"genZ_pt"] = ak.fill_none(ak.pad_none(zs.pt, 1, axis=1, clip=True)[:,0], -999).to_numpy().astype(np.float16)
    output["vars"][f"genZ_eta"] = ak.fill_none(ak.pad_none(zs.eta, 1, axis=1, clip=True)[:,0], -999).to_numpy().astype(np.float16)
    output["vars"][f"genZ_phi"] = ak.fill_none(ak.pad_none(zs.phi, 1, axis=1, clip=True)[:,0], -999).to_numpy().astype(np.float16)
    
    for i in range(2):
        print(ak.fill_none(ak.pad_none(daughters.pt, i+1, axis=1, clip=True)[:,i], -999).to_numpy())
        output["vars"][f"genZ_daughter{i}_pt"] = ak.fill_none(ak.pad_none(daughters.pt, i+1, axis=1, clip=True)[:,i], -999).to_numpy().astype(np.float16)
        output["vars"][f"genZ_daughter{i}_eta"] = ak.fill_none(ak.pad_none(daughters.eta, i+1, axis=1, clip=True)[:,i], -999).to_numpy().astype(np.float16)
        output["vars"][f"genZ_daughter{i}_phi"] = ak.fill_none(ak.pad_none(daughters.phi, i+1, axis=1, clip=True)[:,i], -999).to_numpy().astype(np.float16)
        output["vars"][f"genZ_daughter{i}_pdgId"] = ak.fill_none(ak.pad_none(daughters.pdgId, i+1, axis=1, clip=True)[:,i], -999).to_numpy().astype(np.float16)


##########################################################################################################################
# The following functions are deprecated, but kept around for they might be useful in the future.
##########################################################################################################################


def W_kinematics(lepton, MET):
    """
    WARNING: deprecated.
    Calculate W kinematics.
    """
    W_mt = calc_W_mt(lepton, MET)

    # pT calculation
    W_ptx = lepton.px + MET.px
    W_pty = lepton.py + MET.py

    W_pt = np.sqrt(W_ptx**2 + W_pty**2)

    # phi calculation
    W_phi = np.arctan2(W_pty, W_ptx)

    return W_mt, W_pt, W_phi


def getTopMass(lepton, MET, jets):
    """
    WARNING: deprecated.
    Calculate the top mass for each event.
    """

    M_TOP = 172

    # get the W for each event (defined only in the transverse plane)
    W = make_Wt_4v(lepton, MET)

    # project the jets onto the transverse plane
    jets_T = projectOnTransversePlane(jets)

    # make an awkward array of W bosons of the same shape as the jets (the same W boson in each event is repeated N times, where N = # of jets in that event)
    Ws = ak.cartesian({"W": W, "jets_T": jets_T}).W

    # make the top hypotheses by considering each combination of W and jets_T
    topHypotheses = Ws + jets_T
    topMassHypotheses = topHypotheses.mass
    bestTopMassArg = ak.from_regular(
        ak.argmin(
            np.abs(topMassHypotheses - M_TOP), axis=1, keepdims=True, mask_identity=True
        )
    )
    bestTopMass = ak.flatten(topMassHypotheses[bestTopMassArg])
    return bestTopMass


def getNeutrinoEz(lepton, MET, MW=80.379):
    """
    WARNING: deprecated.
    Get the neutrino z component, assuming the MW.
    """
    Wt = make_Wt_4v(lepton, MET)
    A = MW**2 + Wt.pt**2 - lepton.pt**2 - MET.pt**2
    delta = np.sqrt(A**2 - 4 * (lepton.pt**2) * (MET.pt**2))
    Ez_p = (A * lepton.pz + lepton.e * delta) / (2 * (lepton.pt**2))
    Ez_m = (A * lepton.pz - lepton.e * delta) / (2 * (lepton.pt**2))
    return Ez_p, Ez_m


def make_W_4v(lepton, MET, MW=80.379):
    """
    WARNING: deprecated.
    Make the W boson 4-vector from lepton and MET, assuming the MW.
    Since the sign of the neutrino pz is not known, we have two possible W bosons.
    """
    nu_pz_p, nu_pz_m = getNeutrinoEz(lepton, MET, MW=MW)
    nu_p = make_nu_4v(MET, pz=nu_pz_p)
    nu_m = make_nu_4v(MET, pz=nu_pz_m)
    W_4v_p = lepton + nu_p
    W_4v_m = lepton + nu_m
    return W_4v_p, W_4v_m


def getCosThetaCS(lepton, MET, MW=80.379):
    """
    WARNING: deprecated.
    Get the cosine of the Collins-Soper angle. Assumes the MW.
    """

    nu_pz_p, nu_pz_m = getNeutrinoEz(lepton, MET, MW=MW)

    nu_p = make_nu_4v(MET, pz=nu_pz_p)
    nu_m = make_nu_4v(MET, pz=nu_pz_m)

    random_bits = np.random.randint(2, size=len(lepton))
    nu = ak.where(random_bits, nu_p, nu_m)
    W = lepton + nu

    Pp1 = np.sqrt(2) ** -1 * (lepton.e + lepton.pz)
    Pp2 = np.sqrt(2) ** -1 * (nu.e + nu.pz)
    Pm1 = np.sqrt(2) ** -1 * (lepton.e - lepton.pz)
    Pm2 = np.sqrt(2) ** -1 * (nu.e - nu.pz)

    return (
        (nu.pz / np.abs(nu.pz))
        * 2
        * (Pp1 * Pm2 - Pm1 * Pp2)
        / (MW * np.sqrt(MW**2 + W.pt**2))
    )


def getCosThetaCS2(lepton, MET, MW=80.379):
    """
    WARNING: deprecated.
    Alternative way to get the cosine of the Collins-Soper angle. Assumes the MW.
    """

    nu_pz_p, nu_pz_m = getNeutrinoEz(lepton, MET, MW=MW)

    nu_p = make_nu_4v(MET, pz=nu_pz_p)
    nu_m = make_nu_4v(MET, pz=nu_pz_m)

    random_bits = np.random.randint(2, size=len(lepton))
    nu = ak.where(random_bits, nu_p, nu_m)
    W = lepton + nu

    boost_W = ak.zip(
        {
            "px": -W.px,
            "py": -W.py,
            "pz": -W.pz,
            "mass": W.m,
        },
        with_name="Momentum4D",
    )

    boost_lepton = lepton.boost_p4(boost_W)
    return np.cos(boost_lepton.theta)


def savePhotonInfo(output, events, photons, jets_jec, looseLeptons):
    """
    WARNING: deprecated
    Function to save photon information in the output DataFrame.
    """

    for i in range(2):
        output["vars"]["photon" + str(i + 1) + "_pt"] = ak.fill_none(
            ak.pad_none(photons.pt, i + 1, axis=1, clip=True), -999
        )[:, i]
        output["vars"]["photon" + str(i + 1) + "_phi"] = ak.fill_none(
            ak.pad_none(photons.phi, i + 1, axis=1, clip=True), -999
        )[:, i]
        output["vars"]["photon" + str(i + 1) + "_eta"] = ak.fill_none(
            ak.pad_none(photons.eta, i + 1, axis=1, clip=True), -999
        )[:, i]
        output["vars"]["photon" + str(i + 1) + "_pixelSeed"] = ak.fill_none(
            ak.pad_none(photons.pixelSeed, i + 1, axis=1, clip=True), -999
        )[:, i]
        output["vars"]["photon" + str(i + 1) + "_mvaID"] = ak.fill_none(
            ak.pad_none(photons.mvaID, i + 1, axis=1, clip=True), -999
        )[:, i]
        output["vars"]["photon" + str(i + 1) + "_electronVeto"] = ak.fill_none(
            ak.pad_none(photons.electronVeto, i + 1, axis=1, clip=True), -999
        )[:, i]
        output["vars"]["photon" + str(i + 1) + "_hoe"] = ak.fill_none(
            ak.pad_none(photons.hoe, i + 1, axis=1, clip=True), -999
        )[:, i]
        output["vars"]["photon" + str(i + 1) + "_r9"] = ak.fill_none(
            ak.pad_none(photons.r9, i + 1, axis=1, clip=True), -999
        )[:, i]
        output["vars"]["photon" + str(i + 1) + "_cutBased"] = ak.fill_none(
            ak.pad_none(photons.cutBased, i + 1, axis=1, clip=True), -999
        )[:, i]
        output["vars"]["photon" + str(i + 1) + "_pfRelIso03_all"] = ak.fill_none(
            ak.pad_none(photons.pfRelIso03_all, i + 1, axis=1, clip=True), -999
        )[:, i]
        output["vars"]["photon" + str(i + 1) + "_isScEtaEB"] = ak.fill_none(
            ak.pad_none(photons.isScEtaEB, i + 1, axis=1, clip=True), -999
        )[:, i]
        output["vars"]["photon" + str(i + 1) + "_isScEtaEE"] = ak.fill_none(
            ak.pad_none(photons.isScEtaEE, i + 1, axis=1, clip=True), -999
        )[:, i]

        # if ith photon exist, compute deltaR with jets
        hasIthPhoton = ak.num(photons) > i
        indices_i = np.arange(len(events))[hasIthPhoton]
        photon_i = photons[hasIthPhoton][:, i]
        jets_jec_i = jets_jec[hasIthPhoton]
        looseLeptons_i = looseLeptons[hasIthPhoton]
        minDeltaR_ak4jet_photon_i = np.ones(len(events)) * -999
        minDeltaR_lepton_photon_i = np.ones(len(events)) * -999
        minDeltaR_ak4jet_photon_i[indices_i] = ak.fill_none(
            ak.min(np.abs(jets_jec_i.deltaR(photon_i)), axis=1), -999
        )
        minDeltaR_lepton_photon_i[indices_i] = ak.fill_none(
            ak.min(np.abs(looseLeptons_i.deltaR(photon_i)), axis=1), -999
        )
        output["vars"][
            "minDeltaR_ak4jet_photon" + str(i + 1)
        ] = minDeltaR_ak4jet_photon_i
        output["vars"][
            "minDeltaR_lepton_photon" + str(i + 1)
        ] = minDeltaR_lepton_photon_i
