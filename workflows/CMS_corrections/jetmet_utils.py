import awkward as ak
import cachetools
import numpy as np
from coffea.jetmet_tools import CorrectedJetsFactory, JECStack
from coffea.lookup_tools import extractor

def load_jets(events, isMC, era):
        if (isMC==1) and (era == "2016"):
                vals_jet0 = ak.zip({
                               'pt' : events.OffJet.pt,
                               'eta': events.OffJet.eta,
                               'phi': events.OffJet.phi,
                               'mass': events.OffJet.mass,
                               'area': events.OffJet.area,
                               'mass_raw': events.OffJet.mass, # I think there should be another factor here?
                               'pt_raw': events.OffJet.pt,
                               'passId': events.OffJet.passId,
                               'ptGen': ak.values_astype(ak.without_parameters(ak.zeros_like(events.OffJet.pt)),np.float32)
                },with_name="Momentum4D")
        else:
                vals_jet0 = ak.zip({
                               'pt' : events.Jet.pt,
                               'eta': events.Jet.eta,
                               'phi': events.Jet.phi,
                               'mass': events.Jet.mass,
                               'area': events.Jet.area,
                               'mass_raw': events.Jet.mass, # I think there should be another factor here?
                               'pt_raw': events.Jet.pt,
                               'passId': events.Jet.passId,
                               'ptGen': ak.values_astype(ak.without_parameters(ak.zeros_like(events.Jet.pt)),np.float32),
                               #'rho': ak.broadcast_arrays(events.rho, events.Jet.pt)[0]
                },with_name="Momentum4D")
        #if datatype == "Trigger":
        #       vals_jet0['rho'] = vals_jet0["pt"]/vals_jet0["area"] #ak.broadcast_arrays(arrays["rho"], vals_jet0["pt"])[0]
        #else:
        #       vals_jet0['rho'] = ak.broadcast_arrays(arrays["rho"], vals_jet0["pt"])[0]
        #vals_jet0['rho'] = vals_jet0["pt"]/vals_jet0["area"]
        vals_jet0['rho'] = ak.broadcast_arrays(events.rho, vals_jet0["pt"])[0]

        return vals_jet0

def apply_jecs(self, isMC, Sample, era, events, prefix=""):
    # Find the Collection we want to look at
    if isMC:
        if era == "2016":
            jecdir = "Summer19UL16_V7_MC"
            jerdir = "Summer20UL16_JRV3_MC"
        elif era == "2016apv":
            jecdir = "Summer19UL16_V7_MC"
            jerdir = "Summer20UL16APV_JRV3_MC"
        elif era == "2017":
            jecdir = "Summer19UL17_V5_MC"
            jerdir = "Summer19UL17_JRV3_MC"
        elif era == "2018":
            jecdir = "Summer19UL18_V5_MC"
            jerdir = "Summer19UL18_JRV2_MC"
        else:
            print("WARNING: Unable to find the correct JECs for MC!")
    # Now Data
    elif not isMC:
        if era == "2016apv":
            jecdir = "Summer19UL16APV_RunBCDEF_V7_DATA"
            jerdir = "Summer20UL16APV_JRV3_DATA"
        elif era == "2016":
            jecdir = "Summer19UL16_RunFGH_V7_DATA"
            jerdir = "Summer20UL16_JRV3_DATA"
        elif era == "2017":
            jerdir = "Summer19UL17_JRV3_DATA"
            if "RunB" or "Run2017B" in Sample:
                jecdir = "Summer19UL17_RunB_V5_DATA"
            elif "RunC" or "Run2017C" in Sample:
                jecdir = "Summer19UL17_RunC_V5_DATA"
            elif "RunD" or "Run2017D" in Sample:
                jecdir = "Summer19UL17_RunD_V5_DATA"
            elif "RunE" or "Run2017E" in Sample:
                jecdir = "Summer19UL17_RunE_V5_DATA"
            elif "RunF" or "Run2017F" in Sample:
                jecdir = "Summer19UL17_RunF_V5_DATA"
            else:
                print("WARNING: The JECs for the 2017 data era do not seem to exist!")
        elif era == "2018":
            jerdir = "Summer19UL18_JRV2_DATA"
            if "RunA" or "Run2018A" in Sample:
                jecdir = "Summer19UL18_RunA_V5_DATA"
            elif "RunB" or "Run2018B" in Sample:
                jecdir = "Summer19UL18_RunB_V5_DATA"
            elif "RunC" or "Run2018C" in Sample:
                jecdir = "Summer19UL18_RunC_V5_DATA"
            elif "RunD" or "Run2018D" in Sample:
                jecdir = "Summer19UL18_RunD_V5_DATA"
            else:
                print("WARNING: The JECs for the 2018 data era do not seem to exist!")
        else:
            print("WARNING: Unable to find the correct JECs for Data!")

    # Start working here
    jec_path = prefix + "data/jetmet/JEC/" + jecdir + "/"
    jer_path = prefix + "data/jetmet/JER/" + jerdir + "/"

    # Defined the weight sets we want to use
    ext_ak4 = extractor()
    if isMC:
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
                #'* * ' + jec_path + jecdir +"_Uncertainty_AK4PFchs.junc.txt",
                #'* * ' + jer_path + jerdir +"_PtResolution_AK4PFchs.jr.txt",
                #'* * ' + jer_path + jerdir +"_SF_AK4PFchs.jersf.txt",
            ]
        )

    ext_ak4.finalize()
    evaluator_ak4 = ext_ak4.make_evaluator()

    # WARNING
    # Make sure the acorrections are applied in the right order:
    # https://twiki.cern.ch/twiki/bin/view/CMS/IntroToJEC#Mandatory_Jet_Energy_Corrections
    if isMC:
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
            # jerdir + "_PtResolution_AK4PFchs",
            # jerdir + "_SF_AK4PFchs",
            # jecdir + "_Uncertainty_AK4PFchs",
        ]

    jec_inputs_ak4 = {name: evaluator_ak4[name] for name in jec_stack_names_ak4}
    jec_stack_ak4 = JECStack(jec_inputs_ak4)

    # Prepare the jets from the events
    #jets = events.Jet
    if self.scouting ==1:
        jets = load_jets(events, self.isMC, self.era)
    else:
        jets = events.Jet
        jets["pt_raw"] = (1 - jets["rawFactor"]) * jets["pt"]
        jets["mass_raw"] = (1 - jets["rawFactor"]) * jets["mass"]
        if isMC:
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
    if isMC:
        name_map["ptGenJet"] = "pt_gen"

    # create and return the corrected jet collection
    jec_cache = cachetools.Cache(np.inf)
    jet_factory = CorrectedJetsFactory(name_map, jec_stack_ak4)
    corrected_jets = jet_factory.build(jets, lazy_cache=jec_cache)

    return corrected_jets
