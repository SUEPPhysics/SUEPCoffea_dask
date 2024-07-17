"""
Implements the JEC and JER corrections for jets and MET in coffea.
Follow the latest recommendations: https://cms-jerc.web.cern.ch/Recommendations/

Authors: Chad Freer, Luca Lavezzo
"""

import awkward as ak
import cachetools
import numpy as np
import vector
from coffea.jetmet_tools import CorrectedJetsFactory, CorrectedMETFactory, JECStack
from coffea.lookup_tools import extractor

vector.register_awkward()


def makeJECStack(Sample: str, isMC: int, era: str, jer: bool = False, prefix: str = ""):
    """
    Define the set of weights to use for JECs and JERs based on sample, isMC, and era.
    """

    # Find the Collection we want to look at
    if int(isMC):
        if str(era) == "2016":
            jecdir = "Summer19UL16_V7_MC"
            jerdir = "Summer20UL16_JRV3_MC"
        elif str(era) == "2016apv":
            jecdir = "Summer19UL16_V7_MC"
            jerdir = "Summer20UL16APV_JRV3_MC"
        elif str(era) == "2017":
            jecdir = "Summer19UL17_V5_MC"
            jerdir = "Summer19UL17_JRV3_MC"
        elif str(era) == "2018":
            jecdir = "Summer19UL18_V5_MC"
            jerdir = "Summer19UL18_JRV2_MC"
        else:
            raise Exception("Unable to find the correct JECs for MC!")
    # Now Data
    elif not int(isMC):
        if str(era) == "2016apv":
            jecdir = "Summer19UL16APV_RunBCDEF_V7_DATA"
            jerdir = "Summer20UL16APV_JRV3_DATA"
        elif str(era) == "2016":
            jecdir = "Summer19UL16_RunFGH_V7_DATA"
            jerdir = "Summer20UL16_JRV3_DATA"
        elif str(era) == "2017":
            jerdir = "Summer19UL17_JRV3_DATA"
            if ("RunB" in Sample) or ("Run2017B" in Sample):
                jecdir = "Summer19UL17_RunB_V5_DATA"
            elif ("RunC" in Sample) or ("Run2017C" in Sample):
                jecdir = "Summer19UL17_RunC_V5_DATA"
            elif ("RunD" in Sample) or ("Run2017D" in Sample):
                jecdir = "Summer19UL17_RunD_V5_DATA"
            elif ("RunE" in Sample) or ("Run2017E" in Sample):
                jecdir = "Summer19UL17_RunE_V5_DATA"
            elif ("RunF" in Sample) or ("Run2017F" in Sample):
                jecdir = "Summer19UL17_RunF_V5_DATA"
            else:
                raise Exception(
                    "The JECs for the 2017 data era do not seem to exist! Found sample: "
                    + Sample
                )
        elif str(era) == "2018":
            jerdir = "Summer19UL18_JRV2_DATA"
            if ("RunA" in Sample) or ("Run2018A" in Sample):
                jecdir = "Summer19UL18_RunA_V5_DATA"
            elif ("RunB" in Sample) or ("Run2018B" in Sample):
                jecdir = "Summer19UL18_RunB_V5_DATA"
            elif ("RunC" in Sample) or ("Run2018C" in Sample):
                jecdir = "Summer19UL18_RunC_V5_DATA"
            elif ("RunD" in Sample) or ("Run2018D" in Sample):
                jecdir = "Summer19UL18_RunD_V5_DATA"
            else:
                raise Exception(
                    "The JECs for the 2018 data era do not seem to exist! Found sample"
                    + Sample
                )
        else:
            raise Exception(
                "Unable to find the correct JECs for Data! Found era: " + str(era)
            )
    else:
        raise Exception(
            "Unable to determine if this is MC or Data! Found value of isMC: "
            + str(isMC)
        )

    jec_path = prefix + "data/jetmet/JEC/" + jecdir + "/" + jecdir
    jer_path = prefix + "data/jetmet/JER/" + jerdir + "/" + jerdir

    ext_ak4 = extractor()
    if isMC:
        ext_ak4.add_weight_sets(
            [
                "* * " + jec_path + "_L1FastJet_AK4PFchs.jec.txt",
                "* * " + jec_path + "_L2Relative_AK4PFchs.jec.txt",
                "* * " + jec_path + "_L3Absolute_AK4PFchs.jec.txt",
                "* * " + jec_path + "_UncertaintySources_AK4PFchs.junc.txt",
                "* * " + jec_path + "_Uncertainty_AK4PFchs.junc.txt",
                "* * " + jer_path + "_PtResolution_AK4PFchs.jr.txt",
                "* * " + jer_path + "_SF_AK4PFchs.jersf.txt",
            ]
        )
    else:
        ext_ak4.add_weight_sets(
            [
                "* * " + jec_path + "_L1FastJet_AK4PFchs.jec.txt",
                "* * " + jec_path + "_L3Absolute_AK4PFchs.jec.txt",
                "* * " + jec_path + "_L2Relative_AK4PFchs.jec.txt",
                "* * " + jec_path + "_L2L3Residual_AK4PFchs.jec.txt",
            ]
        )

    ext_ak4.finalize()
    evaluator_ak4 = ext_ak4.make_evaluator()

    # these are the weights that will be used
    if int(isMC):
        jec_stack_names_ak4 = [
            jecdir + "_L1FastJet_AK4PFchs",
            jecdir + "_L2Relative_AK4PFchs",
            jecdir + "_L3Absolute_AK4PFchs",
            jecdir + "_Uncertainty_AK4PFchs",
        ]
        if jer:
            jec_stack_names_ak4 += [
                jerdir + "_PtResolution_AK4PFchs",
                jerdir + "_SF_AK4PFchs",
            ]
    else:
        jec_stack_names_ak4 = [
            jecdir + "_L1FastJet_AK4PFchs",
            jecdir + "_L3Absolute_AK4PFchs",
            jecdir + "_L2Relative_AK4PFchs",
            jecdir + "_L2L3Residual_AK4PFchs",
        ]

    jec_inputs_ak4 = {name: evaluator_ak4[name] for name in jec_stack_names_ak4}
    return JECStack(jec_inputs_ak4)


def getCorrectedJetsFactory(Sample, isMC, era, jer=False, prefix=""):

    jec_stack_ak4 = makeJECStack(Sample, isMC, era, jer=jer, prefix=prefix)

    name_map = jec_stack_ak4.blank_name_map
    name_map["JetPt"] = "pt"
    name_map["JetMass"] = "mass"
    name_map["JetEta"] = "eta"
    name_map["JetA"] = "area"
    name_map["Rho"] = "event_rho"
    name_map["massRaw"] = "mass_raw"
    name_map["ptRaw"] = "pt_raw"
    if int(isMC):
        name_map["ptGenJet"] = "pt_gen"

    return CorrectedJetsFactory(name_map, jec_stack_ak4)


def getCorrectedMETFactory(Sample, isMC, era):

    jec_stack_ak4 = makeJECStack(Sample, isMC, era)

    name_map = jec_stack_ak4.blank_name_map
    name_map["JetPt"] = "pt"
    name_map["JetMass"] = "mass"
    name_map["JetEta"] = "eta"
    name_map["JetA"] = "area"
    name_map["Rho"] = "event_rho"
    name_map["massRaw"] = "mass_raw"
    name_map["ptRaw"] = "pt_raw"
    if int(isMC):
        name_map["ptGenJet"] = "pt_gen"

    name_map["METpt"] = "pt"
    name_map["METphi"] = "phi"
    name_map["JetPhi"] = "phi"
    name_map["UnClusteredEnergyDeltaX"] = "MetUnclustEnUpDeltaX"
    name_map["UnClusteredEnergyDeltaY"] = "MetUnclustEnUpDeltaY"

    return CorrectedMETFactory(name_map)


def prepareJetsForFactory(isMC, events, jets):

    jets["pt_raw"] = (1 - jets["rawFactor"]) * jets["pt"]
    jets["mass_raw"] = (1 - jets["rawFactor"]) * jets["mass"]
    if int(isMC):
        jets["matched_gen_0p2"] = jets.nearest(events.GenJet, threshold=0.2)
        jets["pt_gen"] = ak.values_astype(
            ak.fill_none(jets.matched_gen_0p2.pt, 0), np.float32
        )
    jets["event_rho"] = ak.broadcast_arrays(events.fixedGridRhoFastjetAll, jets.pt)[0]

    return jets

def prepareScoutingJetsForFactory(isMC: int, era: str, events, jets):

    jets["pt_raw"] = jets.pt
    jets["mass_raw"] = jets.mass
    jets["pt_gen"] = ak.values_astype(
        ak.without_parameters(ak.zeros_like(jets.pt)), np.float32
    )
    jets["event_rho"] = events.rho

    return jets

def prepareMETForFactory(MET):

    met = ak.packed(MET, highlevel=True)
    met["UnClusteredEnergyDeltaX"] = met["MetUnclustEnUpDeltaX"]
    met["UnClusteredEnergyDeltaY"] = met["MetUnclustEnUpDeltaY"]

    return MET


def getJetsForMET(events):
    """
    These are the jets that are used to calculate the MET, not the ones for your analysis.
    """
    jets = events.Jet
    jets = jets[
        (
            (jets.pt * (1 - jets.muonSubtrFactor) > 15)
            & (jets.chEmEF + jets.neEmEF < 0.9)
            & (jets.jetId > 0)
        )
    ]

    # veto any jets that are in deltaR < 0.4 with any PF muon
    jets_p4 = ak.zip(
        {
            "pt": jets.pt,
            "eta": jets.eta,
            "phi": jets.phi,
            "mass": jets.mass,
        },
        with_name="Momentum4D",
    )
    muons_4p = ak.zip(
        {
            "pt": events.Muon.pt,
            "eta": events.Muon.eta,
            "phi": events.Muon.phi,
            "mass": events.Muon.mass,
        },
        with_name="Momentum4D",
    )
    product = ak.cartesian({"jet": jets_p4, "muon": muons_4p}, nested=True)
    jets = jets[ak.all(product.jet.deltaR(product.muon) >= 0.4, axis=-1)]

    return jets


def getCorrT1METJetForMET(events, isMC):

    CorrT1METJet = events.CorrT1METJet

    CorrT1METJet["pt"] = CorrT1METJet.rawPt
    CorrT1METJet["pt_raw"] = CorrT1METJet.rawPt
    CorrT1METJet["mass"] = 0 * CorrT1METJet.rawPt
    CorrT1METJet["mass_raw"] = 0 * CorrT1METJet.rawPt
    CorrT1METJet["raw_factor"] = 0 * CorrT1METJet.rawPt

    CorrT1METJet["event_rho"] = ak.broadcast_arrays(
        events.fixedGridRhoFastjetAll, CorrT1METJet.pt
    )[0]

    CorrT1METJet = CorrT1METJet[
        (CorrT1METJet.pt * (1 - CorrT1METJet["muonSubtrFactor"]) > 15)
    ]

    CorrT1METJet_p4 = ak.zip(
        {
            "pt": CorrT1METJet.pt,
            "eta": CorrT1METJet.eta,
            "phi": CorrT1METJet.phi,
            "mass": CorrT1METJet.mass,
        },
        with_name="Momentum4D",
    )

    if int(isMC):

        # veto any jets that are in deltaR < 0.4 with any PF muon
        genJet_4p = ak.zip(
            {
                "pt": events.GenJet.pt,
                "eta": events.GenJet.eta,
                "phi": events.GenJet.phi,
                "mass": events.GenJet.mass,
            },
            with_name="Momentum4D",
        )
        product = ak.cartesian(
            {"jet": CorrT1METJet_p4, "genJet": genJet_4p}, nested=True
        )
        minDeltaRJetGenJet = ak.argmin(product.genJet.deltaR(product.jet), axis=-1)
        CorrT1METJet["pt_gen"] = ak.where(
            minDeltaRJetGenJet < 0.2, genJet_4p[minDeltaRJetGenJet].pt, 0
        )

    # veto any CorrT1METJet that are in deltaR < 0.4 with any PF muon
    muons_4p = ak.zip(
        {
            "pt": events.Muon.pt,
            "eta": events.Muon.eta,
            "phi": events.Muon.phi,
            "mass": events.Muon.mass,
        },
        with_name="Momentum4D",
    )
    product = ak.cartesian({"jet": CorrT1METJet_p4, "muon": muons_4p}, nested=True)
    CorrT1METJet = CorrT1METJet[
        ak.all(product.jet.deltaR(product.muon) >= 0.4, axis=-1)
    ]

    return CorrT1METJet


def getCorrectedMET(sample, isMC, era, events):

    raise Exception("This is not working yet. Please do not use it.")

    met_factory = getCorrectedMETFactory(sample, isMC, era)
    met = prepareMETForFactory(events.MET)
    jec_cache = cachetools.Cache(np.inf)

    jets_forMET = getJetsForMET(events)
    CorrT1METJet_forMET = getCorrT1METJetForMET(events, isMC)

    corrected_jets_forMET = applyJECStoJets(sample, isMC, era, events, jets_forMET)

    alljets_forMET = ak.concatenate(
        [corrected_jets_forMET, CorrT1METJet_forMET], axis=1
    )

    alljets_forMET["pt"] = (1 - alljets_forMET["muonSubtrFactor"]) * alljets_forMET[
        "pt"
    ]
    alljets_forMET["pt_raw"] = (1 - alljets_forMET["muonSubtrFactor"]) * alljets_forMET[
        "pt_raw"
    ]

    return met_factory.build(met, alljets_forMET, lazy_cache=jec_cache)

def applyJECStoJets(sample, isMC, era, events, jets, jer: bool = False, scouting: bool = False, prefix: str = ""):

    jet_factory = getCorrectedJetsFactory(sample, isMC, era, jer=jer, prefix=prefix)
    jec_cache = cachetools.Cache(np.inf)
    if scouting: jets = prepareScoutingJetsForFactory(isMC, era, events, jets)
    else: jets = prepareJetsForFactory(isMC, events, jets)
    jets_corrected = jet_factory.build(jets, lazy_cache=jec_cache)
    return jets_corrected

def getJECCorrectedAK4Jets(sample, isMC, era, events, jer: bool = False, scouting: bool = False, prefix: str = ""):

    if scouting:
        if (isMC == 1) and (era == "2016"):
            jets = events.OffJet
        else:
            jets = events.Jet
    else:
        jets = events.Jet

    return applyJECStoJets(sample, isMC, era, events, jets, jer=jer, scouting=scouting, prefix=prefix)
