import awkward as ak
import numpy as np
from coffea import lookup_tools


def doLeptonScaleVariations(events, leptons, era):
    ## First the muons
    muonIndexes = abs(leptons.pdgId) == 13
    muons = leptons[muonIndexes]
    rochester_data = lookup_tools.txt_converters.convert_rochester_file(
        "data/MuScale/roccor.Run2.v3/RoccoR%i.txt" % (era if era != 2015 else 2016),
        loaduncs=True,
    )
    rochester = lookup_tools.rochester_lookup.rochester_lookup(rochester_data)
    murand = ak.unflatten(np.random.random(ak.sum(ak.num(muons))), ak.num(muons))

    # muSF is the correction
    muSF = rochester.kSmearMC(
        muons.charge,
        muons.pt,
        muons.eta,
        muons.phi,
        muons.aux3,
        murand,
    )
    genpt = events.GenPart.pt[
        ak.values_astype(ak.where(muons.aux1 >= 0, muons.aux1, 0), "int64")
    ]
    muSF = ak.where(
        muons.aux1 >= 0,
        rochester.kSpreadMC(muons.charge, muons.pt, muons.eta, muons.phi, genpt),
        muSF,
    )
    # muSFErr is the uncertainty
    muSFErr = rochester.kSmearMCerror(
        muons.charge,
        muons.pt,
        muons.eta,
        muons.phi,
        muons.aux3,
        murand,
    )
    muSFErr = ak.where(
        muons.aux1 >= 0,
        rochester.kSpreadMCerror(muons.charge, muons.pt, muons.eta, muons.phi, genpt),
        muSFErr,
    )

    muCentral = ak.zip(
        {
            "pt": muons.pt * muSF,
            "eta": muons.eta,
            "phi": muons.phi,
            "mass": muons.mass,
            "charge": muons.pdgId / (-13),
        },
        with_name="Momentum4D",
    )

    muUp = ak.zip(
        {
            "pt": muons.pt * (muSF + muSFErr),
            "eta": muons.eta,
            "phi": muons.phi,
            "mass": muons.mass,
            "charge": muons.pdgId / (-13),
        },
        with_name="Momentum4D",
    )
    muDn = ak.zip(
        {
            "pt": muons.pt * (muSF - muSFErr),
            "eta": muons.eta,
            "phi": muons.phi,
            "mass": muons.mass,
            "charge": muons.pdgId / (-13),
        },
        with_name="Momentum4D",
    )

    ## Now the electrons
    elecIndexes = abs(leptons.pdgId) == 11
    electrons = leptons[elecIndexes]
    elCentral = ak.zip(
        {
            "pt": electrons.pt,
            "eta": electrons.eta,
            "phi": electrons.phi,
            "mass": electrons.mass,
            "charge": electrons.pdgId / (-13),
        },
        with_name="Momentum4D",
    )

    elScaleUp = ak.zip(
        {
            "pt": electrons.pt * (1 + electrons.aux1 / electrons.E),
            "eta": electrons.eta,
            "phi": electrons.phi,
            "mass": electrons.mass,
            "charge": electrons.pdgId / (-13),
        },
        with_name="Momentum4D",
    )

    elScaleDn = ak.zip(
        {
            "pt": electrons.pt * (1 + electrons.aux2 / electrons.E),
            "eta": electrons.eta,
            "phi": electrons.phi,
            "mass": electrons.mass,
            "charge": electrons.pdgId / (-13),
        },
        with_name="Momentum4D",
    )

    elSigmaUp = ak.zip(
        {
            "pt": electrons.pt * (1 + electrons.aux3 / electrons.E),
            "eta": electrons.eta,
            "phi": electrons.phi,
            "mass": electrons.mass,
            "charge": electrons.pdgId / (-13),
        },
        with_name="Momentum4D",
    )

    elSigmaDn = ak.zip(
        {
            "pt": electrons.pt * (1 + electrons.aux4 / electrons.E),
            "eta": electrons.eta,
            "phi": electrons.phi,
            "mass": electrons.mass,
            "charge": electrons.pdgId / (-13),
        },
        with_name="Momentum4D",
    )

    # Reassign
    leptons = ak.concatenate([muCentral, elCentral], axis=1)
    leptons_ElScaleUp = ak.concatenate([muCentral, elScaleUp], axis=1)
    leptons_ElScaleDn = ak.concatenate([muCentral, elScaleDn], axis=1)
    leptons_ElSigmaUp = ak.concatenate([muCentral, elSigmaUp], axis=1)
    leptons_ElSigmaDn = ak.concatenate([muCentral, elSigmaDn], axis=1)
    leptons_MuScaleUp = ak.concatenate([muUp, elCentral], axis=1)
    leptons_MuScaleDn = ak.concatenate([muDn, elCentral], axis=1)

    # And sort
    leptons = leptons[ak.argsort(leptons.pt, axis=1, ascending=False, stable=True)]
    leptons_ElScaleUp = leptons_ElScaleUp[
        ak.argsort(leptons_ElScaleUp.pt, axis=1, ascending=False, stable=True)
    ]
    leptons_ElScaleDn = leptons_ElScaleDn[
        ak.argsort(leptons_ElScaleDn.pt, axis=1, ascending=False, stable=True)
    ]
    leptons_ElSigmaUp = leptons_ElSigmaUp[
        ak.argsort(leptons_ElSigmaUp.pt, axis=1, ascending=False, stable=True)
    ]
    leptons_ElSigmaDn = leptons_ElSigmaDn[
        ak.argsort(leptons_ElSigmaDn.pt, axis=1, ascending=False, stable=True)
    ]
    leptons_MuScaleUp = leptons_MuScaleUp[
        ak.argsort(leptons_MuScaleUp.pt, axis=1, ascending=False, stable=True)
    ]
    leptons_MuScaleDn = leptons_MuScaleDn[
        ak.argsort(leptons_MuScaleDn.pt, axis=1, ascending=False, stable=True)
    ]

    # Get the output
    lepsOut = {
        "": leptons,
        "_ElScaleUp": leptons_ElScaleUp,
        "_ElScaleDown": leptons_ElScaleDn,
        "_ElSigmaUp": leptons_ElSigmaUp,
        "_ElSigmaDown": leptons_ElSigmaDn,
        "_MuScaleUp": leptons_MuScaleUp,
        "_MuScaleDown": leptons_MuScaleDn,
    }

    return lepsOut
