from coffea import processor
from typing import Optional
import awkward as ak
import pandas as pd
import numpy as np
import fastjet
import vector

vector.register_awkward()

class DummyProcessor(processor.ProcessorABC):
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
    ) -> None:
        self._flag = flag
        self.output_location = output_location
        self.do_syst = do_syst
        self.gensumweight = 1.0
        self.era = era
        self.isMC = isMC
        self.sample = sample
        self.syst_var, self.syst_suffix = (
            (syst_var, f"_sys_{syst_var}") if do_syst and syst_var else ("", "")
        )
        self.weight_syst = weight_syst
        self.prefixes = {"SUEP": "SUEP"}

        # Set up the image size and pixels
        self.eta_pix = 280
        self.phi_pix = 360
        self.eta_span = (-2.5, 2.5)
        self.phi_span = (-np.pi, np.pi)
        self.eta_scale = self.eta_pix / (self.eta_span[1] - self.eta_span[0])
        self.phi_scale = self.phi_pix / (self.phi_span[1] - self.phi_span[0])

        # Set up for the output arrays
        self._accumulator = processor.dict_accumulator({})

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, events):
        
        #####################################################################################
        # ---- Output dict definition section
        #####################################################################################
        
        dataset = events.metadata["dataset"]
        print(f"processing: {dataset}")
        output = {}
        output[dataset] = {}

        out_vars_keys = [
            "ntracks",
            "ngood_fastjets",
            "ht",
            "HLT_PFHT900",
            "HLT_PFHT1050",
            "ngood_ak4jets",
            "ngood_tracker_ak4jets",
            "n_loose_ak4jets",
            "n_tight_ak4jets",
            "ht_loose",
            "ht_tight",
            "ht_tracker",
            "Pileup_nTrueInt",
            "PV_npvs",
            "PV_npvsGood"
        ]
        columns_IRM = [
            "SUEP_nconst_IRM",
            "SUEP_ntracks_IRM",
            "SUEP_pt_avg_IRM",
            "SUEP_pt_avg_b_IRM",
            "SUEP_pt_mean_scaled",
            "SUEP_S1_IRM",
            "SUEP_rho0_IRM",
            "SUEP_rho1_IRM",
            "SUEP_pt_IRM",
            "SUEP_eta_IRM",
            "SUEP_phi_IRM",
            "SUEP_mass_IRM",
            "dphi_SUEP_ISR_IRM",
        ]
        columns_CL = [c.replace("IRM", "CL") for c in columns_IRM]
        columns_CO = [c.replace("IRM", "CO") for c in columns_IRM]
        columns = columns_IRM + columns_CL + columns_CO
        columns_CL_ISR = [
            c.replace("IRM", "CL".replace("SUEP", "ISR")) for c in columns_IRM
        ]
        columns_CO_ISR = [
            c.replace("IRM", "CO".replace("SUEP", "ISR")) for c in columns_IRM
        ]
        columns += columns_CL_ISR + columns_CO_ISR
        out_vars_keys += columns
        for key in out_vars_keys:
            output[dataset][key] = []
        
        
        #####################################################################################
        # ---- Initial processing
        #####################################################################################
        
        if self.isMC:
            self.gensumweight = ak.sum(events.genWeight)

        # cut based on ak4 jets to replicate the trigger
        Jets = ak.zip(
            {
                "pt": events.Jet.pt,
                "eta": events.Jet.eta,
                "phi": events.Jet.phi,
                "mass": events.Jet.mass,
                "jetId": events.Jet.jetId,
            }
        )
        jetCut = (Jets.pt > 30) & (abs(Jets.eta) < 4.7)
        ak4jets = Jets[jetCut]
        ht = ak.sum(ak4jets.pt, axis=-1)

        # apply trigger selection
        events = events[events.HLT.TripleMu_5_3_3_Mass3p8_DZ]
        ak4jets = ak4jets[events.HLT.TripleMu_5_3_3_Mass3p8_DZ]
               
        # output empty dataframe if no events pass trigger
        if ak.num(events, axis=0) == 0:
            print(f"[{dataset}]: No events passed trigger. Saving empty outputs.")
            return output


        #####################################################################################
        # ---- Track selection
        #####################################################################################
        
        # Prepare the clean PFCand matched to tracks collection
        Cands = ak.zip(
            {
                "pt": events.PFCands.trkPt,
                "eta": events.PFCands.trkEta,
                "phi": events.PFCands.trkPhi,
                "mass": events.PFCands.mass,
            },
            with_name="Momentum4D",
        )
        cut = (events.PFCands.trkPt >= 0.7)
        return output

    def postprocess(self, accumulator):
        return accumulator
