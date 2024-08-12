"""
SUEP_coffea_WH.py
Coffea producer for SUEP WH analysis, gamma+jets CR. Uses fastjet package to recluster large jets:
https://github.com/scikit-hep/fastjet
Pietro Lugato, Chad Freer, Luca Lavezzo, Joey Reichert 2023
"""

import warnings

import awkward as ak
import numpy as np
import pandas as pd
import vector
from coffea import processor
from hist import Hist

warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

# Importing SUEP specific functions
import workflows.SUEP_utils as SUEP_utils
import workflows.WH_utils as WH_utils
from workflows.SUEP_coffea_WH import SUEP_cluster_WH

# Importing CMS corrections
from workflows.CMS_corrections.btag_utils import btagcuts, doBTagWeights, getBTagEffs
from workflows.CMS_corrections.golden_jsons_utils import applyGoldenJSON
from workflows.CMS_corrections.HEM_utils import jetHEMFilter, METHEMFilter
from workflows.CMS_corrections.PartonShower_utils import GetPSWeights
from workflows.CMS_corrections.Prefire_utils import GetPrefireWeights
from workflows.CMS_corrections.track_killing_utils import track_killing
from workflows.CMS_corrections.jetvetomap_utils import JetVetoMap
from workflows.CMS_corrections.jetmet_utils import applyJECStoJets

# IO utils
from workflows.utils.pandas_accumulator import pandas_accumulator

# Set vector behavior
vector.register_awkward()


class SUEP_cluster_WH_gamma(processor.ProcessorABC):
    def __init__(
        self,
        isMC: int,
        era: str,
        sample: str,
        do_syst: bool,
        flag: bool,
        output_location=None,
    ) -> None:
        self._flag = flag
        self.do_syst = do_syst
        self.era = str(era).lower()
        self.isMC = isMC
        self.sample = sample
        self.output_location = output_location
        self.scouting = 0


    def analysis(self, events, output, out_label=""):

        #####################################################################################
        # ---- Basic event selection
        # Define the events that we will use.
        # Apply triggers, golden JSON, quality filters, and orthogonality selections.
        #####################################################################################

        output["cutflow_total" + out_label] += ak.sum(events.genWeight)

        if self.isMC == 0:
            events = applyGoldenJSON(self, events)
        output["cutflow_goldenJSON" + out_label] += ak.sum(events.genWeight)

        events = WH_utils.genSelection(events, self.sample)
        output["cutflow_genCuts" + out_label] += ak.sum(events.genWeight)

        events = WH_utils.gammaTriggerSelection(
            events, self.era
        )
        output["cutflow_allTriggers" + out_label] += ak.sum(events.genWeight)

        events = WH_utils.qualityFiltersSelection(events, self.era)
        output["cutflow_qualityFilters" + out_label] += ak.sum(events.genWeight)

        events = WH_utils.orthogonalitySelection(events)
        output["cutflow_orthogonality" + out_label] += ak.sum(events.genWeight)

        # output file if no events pass selections, avoids errors later on
        if len(events) == 0:
            print("No events pass basic event selection.")
            return output

        #####################################################################################
        # ---- Lepton selection
        # Define the lepton objects and apply single lepton selection.
        #####################################################################################

        events = WH_utils.onePhotonSelection(events, self.isMC)
        output["cutflow_onePhoton" + out_label] += ak.sum(events.genWeight)

        # TODO do we apply an electron filter here too?
        # _, eventEleHEMCut = jetHEMFilter(self, events.WH_lepton, events.run)

        # output file if no events pass selections, avoids errors later on
        if len(events) == 0:
            print("No events pass oneLepton.")
            return output

        #####################################################################################
        # ---- Jets and MET
        # Grab corrected ak4jets and MET, apply HEM filter, and require at least one ak4jet.
        #####################################################################################

        jets_factory = applyJECStoJets(self.sample, self.isMC, self.era, events, events.Jet, jer=self.isMC)  
        jets_jec = WH_utils.getAK4Jets(jets_factory, events.run, iso=events.WH_gamma, isMC=self.isMC)
        events = ak.with_field(events, jets_factory, "WH_jets_factory")
        events = ak.with_field(events, jets_jec, "WH_jets_jec")
        events = events[ak.num(events.WH_jets_jec) > 0]
        output["cutflow_oneAK4jet" + out_label] += ak.sum(events.genWeight)

        # TODO do we apply HEMcut to all jets (currently done in getAK4jets) or to the events?

        # TODO do we want this? (if so, should go in getAK4jets? or before we give the jets to the JEC corrector?)
        # _, eventJetVetoCut = JetVetoMap(events.WH_jets_jec, self.era)

        events = ak.with_field(events, events.MET, "WH_MET")

        # TODO do we want this?
        # eventMETHEMCut = METHEMFilter(self, events.WH_MET, events.run)    

        #####################################################################################
        # ---- Store event level information
        #####################################################################################

        # these only need to be saved once, as they shouldn't change even with track killing
        if out_label == "":
            events = SUEP_cluster_WH.storeEventVars(
                self,
                events,
                output=output,
            )

        #####################################################################################
        # ---- SUEP definition and analysis
        #####################################################################################

        # indices of events, used to keep track which events pass selections for each method
        # and only fill those rows of the DataFrame (e.g. track killing).
        # from now on, if any cuts are applied, the indices should be updated, and the df
        # should be filled with the updated indices.
        indices = np.arange(0, len(events))

        SUEP_cluster_WH.HighestPTMethod(
            self,
            indices,
            events,
            output=output,
            out_label=out_label,
        )

        return events, output

    def process(self, events):
        dataset = events.metadata["dataset"]

        output = processor.dict_accumulator(
            {
                "gensumweight": processor.value_accumulator(float, 0),
                "cutflow_total": processor.value_accumulator(float, 0),
                "cutflow_goldenJSON": processor.value_accumulator(float, 0),
                "cutflow_genCuts": processor.value_accumulator(float, 0),
                "cutflow_triggerSingleMuon": processor.value_accumulator(float, 0),
                "cutflow_triggerDoubleMuon": processor.value_accumulator(float, 0),
                "cutflow_triggerEGamma": processor.value_accumulator(float, 0),
                "cutflow_allTriggers": processor.value_accumulator(float, 0),
                "cutflow_orthogonality": processor.value_accumulator(float, 0),
                "cutflow_onePhoton": processor.value_accumulator(float, 0),
                "cutflow_qualityFilters": processor.value_accumulator(float, 0),
                "cutflow_jetHEMcut": processor.value_accumulator(float, 0),
                "cutflow_electronHEMcut": processor.value_accumulator(float, 0),
                "cutflow_METHEMcut": processor.value_accumulator(float, 0),
                "cutflow_JetVetoMap": processor.value_accumulator(float, 0),
                "cutflow_oneAK4jet": processor.value_accumulator(float, 0),
                "cutflow_oneCluster": processor.value_accumulator(float, 0),
                "cutflow_twoTracksInCluster": processor.value_accumulator(float, 0),
                "vars": pandas_accumulator(pd.DataFrame())
            }
        )

        # gen weights
        if self.isMC:
            output["gensumweight"] += ak.sum(events.genWeight)
        else:
            genWeight = np.ones(len(events))
            events = ak.with_field(events, genWeight, "genWeight")

        # run the analysis
        events, output = self.analysis(events, output)

        # run the analysis with the track systematics applied
        if self.isMC and self.do_syst:
            output.update(
                {
                    "cutflow_total_track_down": processor.value_accumulator(float, 0),
                    "cutflow_goldenJSON_track_down": processor.value_accumulator(
                        float, 0
                    ),
                    "cutflow_genCuts_track_down": processor.value_accumulator(float, 0),
                    "cutflow_triggerSingleMuon_track_down": processor.value_accumulator(
                        float, 0
                    ),
                    "cutflow_triggerDoubleMuon_track_down": processor.value_accumulator(
                        float, 0
                    ),
                    "cutflow_triggerEGamma_track_down": processor.value_accumulator(
                        float, 0
                    ),
                    "cutflow_allTriggers_track_down": processor.value_accumulator(
                        float, 0
                    ),
                    "cutflow_orthogonality_track_down": processor.value_accumulator(
                        float, 0
                    ),
                    "cutflow_oneLepton_track_down": processor.value_accumulator(
                        float, 0
                    ),
                    "cutflow_qualityFilters_track_down": processor.value_accumulator(
                        float, 0
                    ),
                    "cutflow_jetHEMcut_track_down": processor.value_accumulator(float, 0),
                    "cutflow_electronHEMcut_track_down": processor.value_accumulator(float, 0),
                    "cutflow_METHEMcut_track_down": processor.value_accumulator(float, 0),
                    "cutflow_JetVetoMap_track_down": processor.value_accumulator(float, 0),
                    "cutflow_oneAK4jet_track_down": processor.value_accumulator(
                        float, 0
                    ),
                    "cutflow_oneCluster_track_down": processor.value_accumulator(
                        float, 0
                    ),
                    "cutflow_twoTracksInCluster_track_down": processor.value_accumulator(
                        float, 0
                    ),
                }
            )
            #output = self.analysis(events, output, out_label="_track_down")
            indices = np.arange(0, len(events))
            SUEP_cluster_WH.HighestPTMethod(
                self,
                indices,
                events,
                output=output,
                out_label="_track_down",
            )

        return {dataset: output}

    def postprocess(self, accumulator):
        return accumulator
