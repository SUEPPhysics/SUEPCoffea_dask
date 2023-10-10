import awkward as ak
import correctionlib
import numpy as np


def doTriggerSFs(electrons, muons, era, do_syst=False):
    ceval = correctionlib.CorrectionSet.from_file(
        "data/LeptonTriggerSF/masterJSON.json"
    )  ## Assuming we always run from root dir
    year = str(era)
    SF = {}

    # Add flag indicating which Flavor was selected
    leps = ak.concatenate([electrons.pt, -muons.pt], axis=1)
    # Split into Leading and Subleading Lepton
    leps0temp, leps1temp = np.array(leps[:, 0]), np.array(leps[:, 1])
    # Convert out-of-bounds pTs to maximum bin's value
    leps0 = np.select(
        [np.abs(leps0temp) > 199, np.abs(leps0temp) <= 199],
        [np.sign(leps0temp) * 199.0, leps0temp],
    )
    leps1 = np.select(
        [np.abs(leps1temp) > 199, np.abs(leps1temp) <= 199],
        [np.sign(leps1temp) * 199.0, leps1temp],
    )
    if do_syst:
        SF["TrigSF"] = np.where(
            leps0 > 0,
            ceval["UL-PtPt-Trigger-SFs"].evaluate(
                year, "sf", "Electron", np.abs(leps1), np.abs(leps0)
            ),  # Check if Electron
            np.where(
                leps0 < 0,
                ceval["UL-PtPt-Trigger-SFs"].evaluate(
                    year, "sf", "Muon", np.abs(leps1), np.abs(leps0)
                ),
                -999,  # Check if Muon, if not insert invalid value
            ),
        )
        SF["TrigSFDn"] = np.where(
            leps0 > 0,
            ceval["UL-PtPt-Trigger-SFs"].evaluate(
                year, "sf-down", "Electron", np.abs(leps1), np.abs(leps0)
            ),  # Check if Electron
            np.where(
                leps0 < 0,
                ceval["UL-PtPt-Trigger-SFs"].evaluate(
                    year, "sf-down", "Muon", np.abs(leps1), np.abs(leps0)
                ),
                -999,  # Check if Muon, if not insert invalid value
            ),
        )
        SF["TrigSFUp"] = np.where(
            leps0 > 0,
            ceval["UL-PtPt-Trigger-SFs"].evaluate(
                year, "sf-up", "Electron", np.abs(leps1), np.abs(leps0)
            ),  # Check if Electron
            np.where(
                leps0 < 0,
                ceval["UL-PtPt-Trigger-SFs"].evaluate(
                    year, "sf-up", "Muon", np.abs(leps1), np.abs(leps0)
                ),
                -999,  # Check if Muon, if not insert invalid value
            ),
        )
    else:
        SF["TrigSF"] = np.where(
            leps0 > 0,
            ceval["UL-PtPt-Trigger-SFs"].evaluate(
                year, "sf", "Electron", np.abs(leps1), np.abs(leps0)
            ),  # Check if Electron
            np.where(
                leps0 < 0,
                ceval["UL-PtPt-Trigger-SFs"].evaluate(
                    year, "sf", "Muon", np.abs(leps1), np.abs(leps0)
                ),
                -999,  # Check if Muon, if not insert invalid value
            ),
        )
    return SF


def doLeptonSFs(electrons, muons, era):
    if era == 2015:
        tag = "16APV"
        etag = "2016preVFP"
    elif era == 2016:
        tag = "16"
        etag = "2016postVFP"
    elif era == 2017:
        tag = "17"
        etag = "2017"
    elif era == 2018:
        tag = "18"
        etag = "2018"

    elecs, nelecs = ak.flatten(electrons), np.array(ak.num(electrons))
    mus, nmus = ak.flatten(muons), np.array(ak.num(muons))

    elall = correctionlib.CorrectionSet.from_file(
        "data/EGammaUL%s/electron.json" % tag
    )["UL-Electron-ID-SF"]
    muid = correctionlib.CorrectionSet.from_file("data/MuUL%s/muon_Z.json" % tag)[
        "NUM_LooseID_DEN_TrackerMuons"
    ]
    muiso = correctionlib.CorrectionSet.from_file("data/MuUL%s/muon_Z.json" % tag)[
        "NUM_LooseRelIso_DEN_LooseID"
    ]

    elSF = np.where(
        elecs.pt > 20,
        elall.evaluate(
            etag,
            "sf",
            "RecoAbove20",
            elecs.eta,
            np.where(elecs.pt <= 20, 20.001, elecs.pt),
        ),
        elall.evaluate(
            etag,
            "sf",
            "RecoBelow20",
            elecs.eta,
            np.where(abs(elecs.pt - 15) >= 4.999, 15, elecs.pt),
        ),
    ) * elall.evaluate(etag, "sf", "wp90iso", elecs.eta, elecs.pt)
    elSFUp = np.where(
        elecs.pt > 20,
        elall.evaluate(
            etag,
            "sfup",
            "RecoAbove20",
            elecs.eta,
            np.where(elecs.pt <= 20, 20.001, elecs.pt),
        ),
        elall.evaluate(
            etag,
            "sfup",
            "RecoBelow20",
            elecs.eta,
            np.where(abs(elecs.pt - 15) >= 4.999, 15, elecs.pt),
        ),
    ) * elall.evaluate(etag, "sfup", "wp90iso", elecs.eta, elecs.pt)
    elSFDown = np.where(
        elecs.pt > 20,
        elall.evaluate(
            etag,
            "sfdown",
            "RecoAbove20",
            elecs.eta,
            np.where(elecs.pt <= 20, 20.001, elecs.pt),
        ),
        elall.evaluate(
            etag,
            "sfdown",
            "RecoBelow20",
            elecs.eta,
            np.where(abs(elecs.pt - 15) >= 4.999, 15, elecs.pt),
        ),
    ) * elall.evaluate(etag, "sfdown", "wp90iso", elecs.eta, elecs.pt)

    muSF = muid.evaluate(
        etag + "_UL", abs(mus.eta), np.where(mus.pt <= 15, 15.001, mus.pt), "sf"
    ) * muiso.evaluate(
        etag + "_UL", abs(mus.eta), np.where(mus.pt <= 15, 15.001, mus.pt), "sf"
    )
    muSFUp = muid.evaluate(
        etag + "_UL", abs(mus.eta), np.where(mus.pt <= 15, 15.001, mus.pt), "systup"
    ) * muiso.evaluate(
        etag + "_UL", abs(mus.eta), np.where(mus.pt <= 15, 15.001, mus.pt), "systup"
    )
    muSFDown = muid.evaluate(
        etag + "_UL", abs(mus.eta), np.where(mus.pt <= 15, 15.001, mus.pt), "systdown"
    ) * muiso.evaluate(
        etag + "_UL", abs(mus.eta), np.where(mus.pt <= 15, 15.001, mus.pt), "systdown"
    )

    elSF = ak.unflatten(elSF, nelecs)
    elSFUp = ak.unflatten(elSFUp, nelecs)
    elSFDown = ak.unflatten(elSFDown, nelecs)
    muSF = ak.unflatten(muSF, nmus)
    muSFUp = ak.unflatten(muSFUp, nmus)
    muSFDown = ak.unflatten(muSFDown, nmus)

    lepSF = {}
    lepSF["LepSF"] = ak.prod(ak.concatenate([elSF, muSF], axis=1), axis=1)
    lepSF["LepSFElUp"] = ak.prod(ak.concatenate([elSFUp, muSF], axis=1), axis=1)
    lepSF["LepSFElDown"] = ak.prod(ak.concatenate([elSFDown, muSF], axis=1), axis=1)
    lepSF["LepSFMuUp"] = ak.prod(ak.concatenate([elSF, muSFUp], axis=1), axis=1)
    lepSF["LepSFMuDown"] = ak.prod(ak.concatenate([elSF, muSFDown], axis=1), axis=1)
    return lepSF
