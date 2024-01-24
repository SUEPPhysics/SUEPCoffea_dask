"""
SUEP_coffea.py
Coffea producer for SUEP analysis. Uses fastjet package to recluster large jets:
https://github.com/scikit-hep/fastjet
Chad Freer and Luca Lavezzo, 2021
"""
from typing import Optional

import awkward as ak
import hist
import numpy as np
import pandas as pd
import vector
from coffea import processor

# Importing SUEP specific functions
import workflows.SUEP_utils as SUEP_utils

# Importing CMS corrections
from workflows.CMS_corrections.golden_jsons_utils import applyGoldenJSON
from workflows.pandas_accumulator import pandas_accumulator

# Set vector behavior
vector.register_awkward()


class SUEP_cluster(processor.ProcessorABC):
    def __init__(
        self,
        isMC: int,
        era: int,
        sample: str,
        do_syst: bool,
        syst_var: str,
        weight_syst: bool,
        flag: bool,
        output_location: Optional[str],
        accum: Optional[bool] = None,
        trigger: Optional[str] = None,
        blind: Optional[bool] = False,
        debug: Optional[bool] = None,
    ) -> None:
        self._flag = flag
        self.output_location = output_location
        self.do_syst = do_syst
        self.gensumweight = 1.0
        self.era = int(era)
        self.isMC = bool(isMC)
        self.sample = sample
        self.syst_var, self.syst_suffix = (
            (syst_var, f"_sys_{syst_var}") if do_syst and syst_var else ("", "")
        )
        self.weight_syst = weight_syst
        self.prefixes = {"SUEP": "SUEP"}
        self.accum = accum
        self.trigger = trigger
        self.blind = blind
        self.debug = debug

    def event_selection(self, events, jets):
        # Keep events with HT > 50 GeV
        events = events[ak.sum(jets.pt, axis=-1) > 50]
        jets = jets[ak.sum(jets.pt, axis=-1) > 50]
        return events, jets

    def clean_jets(self, Jets):
        jet_Cut = (Jets.pt > 30) & (abs(Jets.eta) < 2.4)
        return Jets[jet_Cut]

    def fill_histograms(self, events, jets, output):
        dataset = events.metadata["dataset"]

        # These arrays need to be broadcasted to the per muon dims from per event dims
        weights = np.ones(len(events))
        if self.isMC:
            weights = events.genWeight

        if len(events) == 0:
            return

        # Fill the histograms
        gen_jets_clean = self.clean_jets(events.GenJet)
        output[dataset]["histograms"]["ht"].fill(
            ak.sum(jets.pt, axis=-1),
            ak.sum(gen_jets_clean.pt, axis=-1),
            weight=weights,
        )
        output[dataset]["histograms"]["nbjet"].fill(
            ak.sum(events.Jet.btagDeepFlavB > 0.049, axis=-1),
            ak.sum(events.Jet.btagDeepFlavB > 0.2783, axis=-1),
            ak.sum(events.Jet.btagDeepFlavB > 0.71, axis=-1),
            weight=weights,
        )
        output[dataset]["histograms"]["ngenjet"].fill(
            ak.sum(abs(events.GenJet.partonFlavour) == 4, axis=-1),
            ak.sum(abs(events.GenJet.partonFlavour) == 5, axis=-1),
            weight=weights,
        )
        return

    def analysis(self, events, output):
        #####################################################################################
        # ---- Trigger event selection
        # Cut based on ak4 jets to replicate the trigger
        #####################################################################################

        # golden jsons for offline data
        if not self.isMC:
            events = applyGoldenJSON(self, events)

        # event selection
        jets = self.clean_jets(events.Jet)
        events, jets = self.event_selection(events, jets)

        # fill the histograms
        self.fill_histograms(events, jets, output)

        return

    def process(self, events):
        dataset = events.metadata["dataset"]
        cutflow = hist.Hist.new.StrCategory(
            [
                "all",
                "trigger",
                "nMu==3",
            ],
            name="cutflow",
            label="cutflow",
        ).Weight()
        histograms = {
            "ht": hist.Hist.new.Regular(
                100,
                10,
                10000,
                name="ht",
                label="ht",
                transform=hist.axis.transform.log,
            )
            .Regular(
                100,
                10,
                10000,
                name="ht_gen",
                label="ht_gen",
                transform=hist.axis.transform.log,
            )
            .Weight(),
            "nbjet": hist.Hist.new.Regular(
                30,
                0,
                30,
                name="nbjet_loose",
                label="nbjet_loose",
            )
            .Regular(
                30,
                0,
                30,
                name="nbjet_medium",
                label="nbjet_medium",
            )
            .Regular(
                30,
                0,
                30,
                name="nbjet_tight",
                label="nbjet_tight",
            )
            .Weight(),
            "ngenjet": hist.Hist.new.Regular(
                30,
                0,
                30,
                name="ncgenjet",
                label="ncgenjet",
            )
            .Regular(
                30,
                0,
                30,
                name="nbgenjet",
                label="nbgenjet",
            )
            .Weight(),
        }
        output = {
            dataset: {
                "cutflow": cutflow,
                "gensumweight": processor.value_accumulator(float, 0),
                "vars": pandas_accumulator(pd.DataFrame()),
                "histograms": histograms,
            },
        }

        # gen weights
        if self.isMC:
            self.gensumweight = ak.sum(events.genWeight)
            output[dataset]["gensumweight"].add(self.gensumweight)

        # run the analysis
        self.analysis(events, output)

        return output

    def postprocess(self, accumulator):
        pass
