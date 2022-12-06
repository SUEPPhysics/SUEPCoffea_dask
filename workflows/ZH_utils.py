import awkward as ak


def selectByLeptons(self, events, extraColls=[], lepveto=False):
    ###lepton selection criteria--4momenta collection for plotting

    muons = ak.zip(
        {
            "pt": events.Muon.pt,
            "eta": events.Muon.eta,
            "phi": events.Muon.phi,
            "mass": events.Muon.mass,
            "charge": events.Muon.pdgId / (-13),
        },
        with_name="Momentum4D",
    )

    electrons = ak.zip(
        {
            "pt": events.Electron.pt,
            "eta": events.Electron.eta,
            "phi": events.Electron.phi,
            "mass": events.Electron.mass,
            "charge": events.Electron.pdgId / (-11),
        },
        with_name="Momentum4D",
    )

    ###  Some very simple selections on ID ###
    ###  Muons: loose ID + dxy dz cuts mimicking the medium prompt ID https://twiki.cern.ch/twiki/bin/viewauth/CMS/SWGuideMuonIdRun2
    ###  Electrons: loose ID + dxy dz cuts for promptness https://twiki.cern.ch/twiki/bin/view/CMS/EgammaCutBasedIdentification
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
        & (abs(events.Electron.dxy) < 0.05 + 0.05 * (events.Electron.eta > 1.479))
        & (abs(events.Electron.dz) < 0.10 + 0.10 * (events.Electron.eta > 1.479))
        & ((abs(events.Electron.eta) < 1.444) | (abs(events.Electron.eta) > 1.566))
        & (abs(events.Electron.eta) < 2.5)
    )

    ### Apply the cuts
    # Object selection. selMuons contain only the events that are filtered by cutMuons criteria.
    selMuons = muons[cutMuons]
    selElectrons = electrons[cutElectrons]

    ### Now global cuts to select events. Notice this means exactly two leptons with pT >= 10, and the leading one pT >= 25

    # cutHasTwoMuons imposes three conditions:
    #  First, number of muons (axis=1 means column. Each row is an event.) in an event is 2.
    #  Second, pt of the muons is greater than 25.
    #  Third, Sum of charge of muons should be 0. (because it originates from Z)

    if self.doOF:
        # Only for the OF sideband for tt/WW/Fakes estimation
        templeps = ak.concatenate([selMuons, selElectrons], axis=1)
        cutHasOFLeps = (
            (ak.num(templeps, axis=1) == 2)
            & (ak.max(templeps.pt, axis=1, mask_identity=False) >= 25)
            & (ak.sum(templeps.charge, axis=1) == 0)
        )
        events = events[cutHasOFLeps]
        selElectrons = selElectrons[cutHasOFLeps]
        selMuons = selMuons[cutHasOFLeps]
        cutOneAndOne = (ak.num(selElectrons) == 1) & (ak.num(selMuons) == 1)
        events = events[cutOneAndOne]
        selElectrons = selElectrons[cutOneAndOne]
        selMuons = selMuons[cutOneAndOne]

    else:

        if lepveto:  # lepton veto used in the HT analyses for orthoganality

            # Cut out events with any lepton to also be orthogonal to possible WH analysis
            cutAnyElecs = (ak.num(selElectrons, axis=1) > 0) & (
                ak.max(selElectrons.pt, axis=1, mask_identity=False) >= 25
            )

            cutAnyMuons = (ak.num(selElectrons, axis=1) > 0) & (
                ak.max(selElectrons.pt, axis=1, mask_identity=False) >= 25
            )

            cutAnyLeps = cutAnyElecs | cutAnyMuons
            events = events[~cutAnyLeps]
            selElectrons = selElectrons[~cutAnyLeps]
            selMuons = selMuons[~cutAnyLeps]

        else:
            cutHasTwoMuons = (
                (ak.num(selMuons, axis=1) == 2)
                & (ak.max(selMuons.pt, axis=1, mask_identity=False) >= 25)
                & (ak.sum(selMuons.charge, axis=1) == 0)
            )
            cutHasTwoElecs = (
                (ak.num(selElectrons, axis=1) == 2)
                & (ak.max(selElectrons.pt, axis=1, mask_identity=False) >= 25)
                & (ak.sum(selElectrons.charge, axis=1) == 0)
            )
            cutTwoLeps = (ak.num(selElectrons, axis=1) + ak.num(selMuons, axis=1)) < 4
            cutHasTwoLeps = ((cutHasTwoMuons) | (cutHasTwoElecs)) & cutTwoLeps

            ### Cut the events, also return the selected leptons for operation down the line
            events = events[cutHasTwoLeps]
            selElectrons = selElectrons[cutHasTwoLeps]
            selMuons = selMuons[cutHasTwoLeps]

    return (
        events,
        selElectrons,
        selMuons,
    )  # , [coll[cutHasTwoLeps] for coll in extraColls]
