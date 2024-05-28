import awkward as ak
import cachetools
import numpy as np
from coffea.jetmet_tools import CorrectedJetsFactory, CorrectedMETFactory, JECStack
from coffea.lookup_tools import extractor


def load_jets(self, events):
    if (self.isMC == 1) and ("2016" in self.era):
        vals_jet0 = events.OffJet
        vals_jet0["pt_raw"] = events.OffJet.pt
        vals_jet0["mass_raw"] = events.OffJet.mass
        vals_jet0["pt_gen"] = ak.values_astype(
            ak.without_parameters(ak.zeros_like(events.OffJet.pt)), np.float32
        )
        vals_jet0["rho"] = events.rho
    else:
        vals_jet0 = events.Jet
        vals_jet0["pt_raw"] = events.Jet.pt
        vals_jet0["mass_raw"] = events.Jet.mass
        vals_jet0["rho"] = events.rho
        vals_jet0["pt_gen"] = ak.values_astype(
            ak.without_parameters(ak.zeros_like(events.Jet.pt)), np.float32
        )

    return vals_jet0


def apply_jecs(self, Sample, events, prefix=""):
    # Find the Collection we want to look at
    if int(self.isMC):
        if str(self.era) == "2016":
            jecdir = "Summer19UL16_V7_MC"
            jerdir = "Summer20UL16_JRV3_MC"
        elif str(self.era) == "2016apv":
            jecdir = "Summer19UL16_V7_MC"
            jerdir = "Summer20UL16APV_JRV3_MC"
        elif str(self.era) == "2017":
            jecdir = "Summer19UL17_V5_MC"
            jerdir = "Summer19UL17_JRV3_MC"
        elif str(self.era) == "2018":
            jecdir = "Summer19UL18_V5_MC"
            jerdir = "Summer19UL18_JRV2_MC"
        else:
            raise Exception("Unable to find the correct JECs for MC!")
    # Now Data
    elif not int(self.isMC):
        if str(self.era) == "2016apv":
            jecdir = "Summer19UL16APV_RunBCDEF_V7_DATA"
            jerdir = "Summer20UL16APV_JRV3_DATA"
        elif str(self.era) == "2016":
            jecdir = "Summer19UL16_RunFGH_V7_DATA"
            jerdir = "Summer20UL16_JRV3_DATA"
        elif str(self.era) == "2017":
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
        elif str(self.era) == "2018":
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
                "Unable to find the correct JECs for Data! Found era: " + str(self.era)
            )
    else:
        raise Exception(
            "Unable to determine if this is MC or Data! Found value of isMC: "
            + str(self.isMC)
        )

    # Start working here
    jec_path = prefix + "data/jetmet/JEC/" + jecdir + "/"
    jer_path = prefix + "data/jetmet/JER/" + jerdir + "/"

    # Defined the weight sets we want to use
    ext_ak4 = extractor()
    if self.isMC:
        ext_ak4.add_weight_sets(
            [  # change to correct files
                "* * "
                + jec_path
                + jecdir
                + "_L1FastJet_AK4PFchs.jec.txt",  # looks to be 0,
                #'* * ' + jec_path + jecdir +"_L1RC_AK4PFchs.jec.txt", #needs area
                #'* * ' + jec_path + jecdir +"_L2L3Residual_AK4PFchs.jec.txt",
                #'* * ' + jec_path + jecdir +"_L2Residual_AK4PFchs.jec.txt",
                "* * " + jec_path + jecdir + "_L2Relative_AK4PFchs.jec.txt",
                "* * "
                + jec_path
                + jecdir
                + "_L3Absolute_AK4PFchs.jec.txt",  # looks to be 1, no change
                "* * " + jec_path + jecdir + "_Uncertainty_AK4PFchs.junc.txt",
                "* * " + jer_path + jerdir + "_PtResolution_AK4PFchs.jr.txt",
                "* * " + jer_path + jerdir + "_SF_AK4PFchs.jersf.txt",
            ]
        )
    else:
        ext_ak4.add_weight_sets(
            [  # change to correct files
                "* * "
                + jec_path
                + jecdir
                + "_L1FastJet_AK4PFchs.jec.txt",  # looks to be 0,
                "* * " + jec_path + jecdir + "_L1RC_AK4PFchs.jec.txt",  # needs area
                "* * " + jec_path + jecdir + "_L2Relative_AK4PFchs.jec.txt",
                "* * "
                + jec_path
                + jecdir
                + "_L3Absolute_AK4PFchs.jec.txt",  # looks to be 1, no change
                "* * " + jec_path + jecdir + "_L2L3Residual_AK4PFchs.jec.txt",
                "* * " + jec_path + jecdir + "_L2Residual_AK4PFchs.jec.txt",
            ]
        )

    ext_ak4.finalize()
    evaluator_ak4 = ext_ak4.make_evaluator()

    # WARNING
    # Make sure the acorrections are applied in the right order:
    # https://twiki.cern.ch/twiki/bin/view/CMS/IntroToJEC#Mandatory_Jet_Energy_Corrections
    if self.isMC:
        jec_stack_names_ak4 = [
            jecdir + "_L1FastJet_AK4PFchs",
            # jecdir + "_L1RC_AK4PFchs",
            # jecdir + "_L2L3Residual_AK4PFchs",
            # jecdir + "_L2Residual_AK4PFchs",
            jecdir + "_L2Relative_AK4PFchs",
            jecdir + "_L3Absolute_AK4PFchs",
            jerdir + "_PtResolution_AK4PFchs",
            jerdir + "_SF_AK4PFchs",
            jecdir + "_Uncertainty_AK4PFchs",
        ]
    else:
        jec_stack_names_ak4 = [
            jecdir + "_L1FastJet_AK4PFchs",
            jecdir + "_L1RC_AK4PFchs",
            jecdir + "_L2Relative_AK4PFchs",
            jecdir + "_L3Absolute_AK4PFchs",
            jecdir + "_L2L3Residual_AK4PFchs",
            jecdir + "_L2Residual_AK4PFchs",
        ]

    jec_inputs_ak4 = {name: evaluator_ak4[name] for name in jec_stack_names_ak4}
    jec_stack_ak4 = JECStack(jec_inputs_ak4)

    # Prepare the jets from the events
    if self.scouting == 1:
        jets = load_jets(self, events)
    else:
        jets = events.Jet
        jets["pt_raw"] = (1 - jets["rawFactor"]) * jets["pt"]
        jets["mass_raw"] = (1 - jets["rawFactor"]) * jets["mass"]
        if self.isMC:
            jets["pt_gen"] = ak.values_astype(
                ak.fill_none(jets.matched_gen.pt, 0), np.float32
            )
        jets["rho"] = ak.broadcast_arrays(events.fixedGridRhoFastjetAll, jets.pt)[0]
    # Create the map (for both MC and data)
    name_map = jec_stack_ak4.blank_name_map
    name_map["JetPt"] = "pt"
    name_map["JetMass"] = "mass"
    name_map["JetEta"] = "eta"
    name_map["JetA"] = "area"
    name_map["Rho"] = "rho"
    name_map["massRaw"] = "mass_raw"
    name_map["ptRaw"] = "pt_raw"
    if self.isMC:
        name_map["ptGenJet"] = "pt_gen"

    # create and return the corrected jet collection
    jec_cache = cachetools.Cache(np.inf)
    jet_factory = CorrectedJetsFactory(name_map, jec_stack_ak4)
    corrected_jets = jet_factory.build(jets, lazy_cache=jec_cache)

    if self.scouting == 1:
        return corrected_jets, None

    else:
        name_map["METpt"] = "pt"
        name_map["METphi"] = "phi"
        name_map["JetPhi"] = "phi"
        name_map["UnClusteredEnergyDeltaX"] = "MetUnclustEnUpDeltaX"
        name_map["UnClusteredEnergyDeltaY"] = "MetUnclustEnUpDeltaY"

        met_factory = CorrectedMETFactory(name_map)
        met = ak.packed(events.MET, highlevel=True)
        met["pt"] = met["pt"]
        met["phi"] = met["phi"]
        met["UnClusteredEnergyDeltaX"] = met["MetUnclustEnUpDeltaX"]
        met["UnClusteredEnergyDeltaY"] = met["MetUnclustEnUpDeltaY"]
        corrected_met = met_factory.build(met, corrected_jets, lazy_cache=jec_cache)

        return corrected_jets, corrected_met
