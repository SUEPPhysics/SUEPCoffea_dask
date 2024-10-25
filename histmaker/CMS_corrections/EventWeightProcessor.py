"""
Author: Luca Lavezzo
Date: August 2024
"""

import logging

import numpy as np
import pandas as pd
from CMS_corrections import higgs_reweight, pileup_weight, triggerSF


def apply_correctionlib(corr_file: str, corr_variable: str, input_variable: np.ndarray):
    """
    Apply a correction from a correctionlib file to a (numpy) variable.
    corr_file: path to the file containing the weights
    corr_variable: variable to use as correction
    input_variable: variable to use as input for the weights

    https://cms-nanoaod.github.io/correctionlib/correctionlib_tutorial.html
    """
    import correctionlib

    ceval = correctionlib.CorrectionSet.from_file(corr_file)
    corrections = ceval[corr_variable].evaluate(input_variable)

    return corrections


class EventWeightProcessor:
    """
    Apply CMS_corrections weights and variations to a pandas DataFrame,
    based on the options provided. Adds a column "event_weight" to the DataFrame.
    """

    def __init__(
        self,
        variation: str,
        sample: str,
        isMC: bool,
        era: str,
        channel: str,
        region: str = "",
    ):

        self.variation = variation
        self.sample = sample
        self.isMC = isMC
        self.era = era
        self.channel = channel
        self.region = region

        self.supported_channels = ["WH", "WH-VRGJ", "ggF", "ggF-scout"]
        self.supported_variations = [
            "",  # nominal
            "puweights_up",
            "puweights_down",
            "trigSF_up",
            "trigSF_down",
            "PSWeight_ISR_up",
            "PSWeight_ISR_down",
            "PSWeight_FSR_up",
            "PSWeight_FSR_down",
            "track_down",
            "JER_up",
            "JER_down",
            "JES_up",
            "JES_down",
            "higgs_weights_up",
            "higgs_weights_down",
            "prefire_up",
            "prefire_down",
            "LepSFElUp",
            "LepSFElDown",
            "LepSFMuUp",
            "LepSFMuDown",
            "bTagWeight_HFcorrelated_Up",
            "bTagWeight_HFcorrelated_Dn",
            "bTagWeight_HFuncorrelated_Up",
            "bTagWeight_HFuncorrelated_Dn",
            "bTagWeight_LFcorrelated_Up",
            "bTagWeight_LFcorrelated_Dn",
            "bTagWeight_LFuncorrelated_Up",
            "bTagWeight_LFuncorrelated_Dn",
            "photon_SF_up",
            "photon_SF_down",
        ]
        self.supported_isMC = [0, 1]
        self.supported_eras = ["2016", "2017", "2018", "2016apv"]

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run the processor on the DataFrame.
        """
        if not self.validate_options():
            raise Exception("Invalid options.")

        if not self.validate_df(df):
            raise Exception("Invalid DataFrame.")

        return self.apply_event_weights(df)

    def validate_df(self, df: pd.DataFrame) -> bool:

        if "event_weight" in df.keys():
            logging.error("DataFrame already has column 'event_weight'.")
            return False

        return True

    def validate_options(self) -> bool:

        if self.channel not in self.supported_channels:
            logging.error(f"Channel {self.channel} not supported.")
            return False

        if self.variation not in self.supported_variations:
            logging.error(f"Variation {self.variation} not supported.")
            return False

        if self.isMC not in self.supported_isMC:
            logging.error(f"isMC {self.isMC} not supported.")
            return False

        if self.era not in self.supported_eras:
            logging.error(f"Era {self.era} not supported.")
            return False

        return True

    def apply_event_weights(self, df: pd.DataFrame) -> pd.DataFrame:

        # apply event weights
        if self.isMC:

            df["event_weight"] = df["genweight"].to_numpy()

            # 1) pileup weights
            puweights, puweights_up, puweights_down = pileup_weight.pileup_weight(
                self.era
            )
            pu = pileup_weight.get_pileup_weights(
                df, self.variation, puweights, puweights_up, puweights_down
            )
            df["event_weight"] *= pu

            # 2) PS weights
            if "PSWeight" in self.variation:
                df["event_weight"] *= df[self.variation]

            # 3) prefire weights
            if self.era in ["2016apv", "2016", "2017"]:
                if "prefire" in self.variation:
                    df["event_weight"] *= df[self.variation]
                else:
                    df["event_weight"] *= df["prefire_nom"]

            # 4) TriggerSF weights
            if self.channel == "ggF":
                (
                    trig_bins,
                    trig_weights,
                    trig_weights_up,
                    trig_weights_down,
                ) = triggerSF.triggerSF(self.era)
                trigSF = triggerSF.get_trigSF_weight(
                    df,
                    self.variation,
                    trig_bins,
                    trig_weights,
                    trig_weights_up,
                    trig_weights_down,
                )
                df["event_weight"] *= trigSF
            elif self.channel == "ggF-scout":
                trigSF = triggerSF.get_scout_trigSF_weight(
                    np.array(df["ht"]).astype(int), self.variation, self.era
                )
                df["event_weight"] *= trigSF
            elif self.channel in ["WH", "WH-VRGJ"]:
                # TODO add WH triggerSF
                pass

            # 5) Higgs_pt weights
            if "mS125" in self.sample:
                (
                    higgs_bins,
                    higgs_weights,
                    higgs_weights_up,
                    higgs_weights_down,
                ) = higgs_reweight.higgs_reweight(df["SUEP_genPt"])
                higgs_weight = higgs_reweight.get_higgs_weight(
                    df,
                    self.variation,
                    higgs_bins,
                    higgs_weights,
                    higgs_weights_up,
                    higgs_weights_down,
                )
                df["event_weight"] *= higgs_weight

            # 8) b-tag weights. These have different values for each event selection
            if self.channel in ['WH', 'WH-VRGJ'] and self.isMC:
                if "btag" in self.variation.lower():
                    btag_weights = self.variation
                else:
                    btag_weights = "bTagWeight_central"
                btag_weights += "_" + self.channel.lower()
                if btag_weights not in df.keys():
                    logging.warning(
                        f"btag weights {btag_weights} not found in DataFrame. Not applying them."
                    )
                    # TODO this should not be a pass, but a raise exception, but we don't have all weights rn
                    pass
                else:
                    df["event_weight"] *= df[btag_weights]

            # 9) lepton SF
            if self.channel == 'WH':
                if 'LepSF' in self.variation:
                    df["event_weight"] *= df[self.variation]
                else:
                    df["event_weight"] *= df["LepSF"]

            # 10) photon SF
            if self.channel == 'WH-VRGJ':
                if 'photon_SF' in self.variation:
                    df["event_weight"] *= df[self.variation]
                else:
                    df["event_weight"] *= df["photon_SF"]

        # data
        else:

            df["event_weight"] = np.ones(df.shape[0])

            # un-prescaling for the gamma triggers
            if self.channel == "WH-VRGJ":
                df["event_weight"] *= df["WH_gammaTriggerUnprescaleWeight"]

            # ad hoc weights
            # print("TEMPORARY: applying ad hoc weights for data to correct suep pT. Meant only for gamma+jets testing.")
            # df["event_weight"] *= apply_correctionlib("CMS_corrections/suep_pt_corr.json", "ptweight", df["SUEP_pt_HighestPT"].to_numpy())

        return df
