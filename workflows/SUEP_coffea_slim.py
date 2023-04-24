"""
SUEP_coffea.py
Coffea producer for SUEP analysis. Uses fastjet package to recluster large jets:
https://github.com/scikit-hep/fastjet
Chad Freer and Luca Lavezzo, 2021
"""
from typing import Optional

import awkward as ak
import hist
import pandas as pd
import vector
from coffea import processor

# Importing SUEP specific functions
import workflows.SUEP_utils as SUEP_utils

# Importing CMS corrections
from workflows.CMS_corrections.golden_jsons_utils import applyGoldenJSON
from workflows.CMS_corrections.jetmet_utils import apply_jecs
from workflows.CMS_corrections.PartonShower_utils import GetPSWeights
from workflows.CMS_corrections.Prefire_utils import GetPrefireWeights
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
        self.doOF = False
        self.accum = accum
        self.trigger = trigger
        self.debug = debug

    def jet_awkward(self, Jets):
        """
        Create awkward array of jets. Applies basic selections.
        Returns: awkward array of dimensions (events x jets x 4 momentum)
        """
        Jets_awk = ak.zip(
            {
                "pt": Jets.pt,
                "eta": Jets.eta,
                "phi": Jets.phi,
                "mass": Jets.mass,
                "btagDeepB": Jets.btagDeepB,
            }
        )
        jet_awk_Cut = (Jets.pt > 30) & (abs(Jets.eta) < 2.4)
        Jets_correct = Jets_awk[jet_awk_Cut]
        return Jets_correct

    def eventSelection(self, events):
        """
        Applies trigger, returns events.
        Default is PFHT triggers. Can use selection variable for customization.
        """
        # NOTE: Might be a good idea to make this a 'match-case' statement
        # once we can move to Python 3.10 for good.
        if self.trigger == "TripleMu":
            if self.era == 2016:
                trigger = events.HLT.TripleMu_5_3_3 == 1
            elif self.era == 2017:
                trigger = events.HLT.TripleMu_5_3_3_Mass3p8to60_DZ == 1
            elif self.era == 2018:
                trigger = events.HLT.TripleMu_5_3_3_Mass3p8_DZ == 1
            else:
                raise ValueError("Invalid era")
        events = events[trigger]
        return events

    def muon_filter(self, events, nMuons=4):
        """
        Filter events after the TripleMu trigger.
        Cleans muons and electrons.
        Requires at least 6 muons with mediumId, pt, dxy, dz, and eta cuts.
        """
        muons = events.Muon
        electrons = events.Electron
        clean_muons = (
            (events.Muon.mediumId)
            & (events.Muon.pt > 3)
            & (abs(events.Muon.dxy) <= 0.02)
            & (abs(events.Muon.dz) <= 0.1)
            & (abs(events.Muon.eta) < 2.4)
        )
        clean_electrons = (
            (events.Electron.mvaFall17V2noIso_WPL)
            & (events.Electron.pt > 3)
            & (
                abs(events.Electron.dxy)
                < 0.05 + 0.05 * (abs(events.Electron.eta) > 1.479)
            )
            & (
                abs(events.Electron.dz)
                < 0.10 + 0.10 * (abs(events.Electron.eta) > 1.479)
            )
            & ((abs(events.Electron.eta) < 1.444) | (abs(events.Electron.eta) > 1.566))
            & (abs(events.Electron.eta) < 2.5)
        )
        muons = muons[clean_muons]
        electrons = electrons[clean_electrons]
        select_by_muons = ak.num(muons, axis=-1) >= nMuons
        events = events[select_by_muons]
        muons = muons[select_by_muons]
        electrons = electrons[select_by_muons]
        return events, electrons, muons

    def getGenTracks(self, events):
        genParts = events.GenPart
        genParts = ak.zip(
            {
                "pt": genParts.pt,
                "eta": genParts.eta,
                "phi": genParts.phi,
                "mass": genParts.mass,
                "pdgID": genParts.pdgId,
            },
            with_name="Momentum4D",
        )
        return genParts

    def storeEventVars(
        self,
        events,
        output,
        electrons,
        muons,
        out_label="",
    ):
        dataset = events.metadata["dataset"]

        # muon inter-isolation
        muonsCollection = ak.zip(
            {
                "pt": muons.pt,
                "eta": muons.eta,
                "phi": muons.phi,
                "mass": muons.mass,
                "charge": muons.pdgId / (-13),
            },
            with_name="Momentum4D",
        )
        # electronsCollection = ak.zip(
        #    {
        #        "pt": electrons.pt,
        #        "eta": electrons.eta,
        #        "phi": electrons.phi,
        #        "mass": electrons.mass,
        #        "charge": electrons.pdgId / (-11),
        #    },
        #    with_name="Momentum4D",
        # )

        # select out ak4jets
        ak4jets = self.jet_awkward(events.Jet)

        # work on JECs and systematics
        prefix = ""
        if self.accum:
            if "casa" in self.accum:
                prefix = "dask-worker-space/"
        jets_c = apply_jecs(
            isMC=self.isMC,
            Sample=self.sample,
            era=self.era,
            events=events,
            prefix=prefix,
        )
        jets_jec = self.jet_awkward(jets_c)
        if self.isMC:
            jets_jec_JERUp = self.jet_awkward(jets_c["JER"].up)
            jets_jec_JERDown = self.jet_awkward(jets_c["JER"].down)
            jets_jec_JESUp = self.jet_awkward(jets_c["JES_jes"].up)
            jets_jec_JESDown = self.jet_awkward(jets_c["JES_jes"].down)
        # For data set these all to nominal so we can plot without switching all of the names
        else:
            jets_jec_JERUp = jets_jec
            jets_jec_JERDown = jets_jec
            jets_jec_JESUp = jets_jec
            jets_jec_JESDown = jets_jec

        # save per event variables to a dataframe
        if out_label == "":
            output[dataset]["vars"][
                "genWeight" + out_label
            ] = events.genWeight.to_list()
            output[dataset]["vars"]["ht" + out_label] = ak.sum(
                ak4jets.pt, axis=-1
            ).to_list()
            output[dataset]["vars"]["ht_JEC" + out_label] = ak.sum(
                jets_jec.pt, axis=-1
            ).to_list()
            output[dataset]["vars"]["ht_JEC" + out_label + "_JER_up"] = ak.sum(
                jets_jec_JERUp.pt, axis=-1
            ).to_list()
            output[dataset]["vars"]["ht_JEC" + out_label + "_JER_down"] = ak.sum(
                jets_jec_JERDown.pt, axis=-1
            ).to_list()
            output[dataset]["vars"]["ht_JEC" + out_label + "_JES_up"] = ak.sum(
                jets_jec_JESUp.pt, axis=-1
            ).to_list()
            output[dataset]["vars"]["ht_JEC" + out_label + "_JES_down"] = ak.sum(
                jets_jec_JESDown.pt, axis=-1
            ).to_list()

            if self.era == 2016:
                output[dataset]["vars"]["HLT_PFHT900" + out_label] = events.HLT.PFHT900
            else:
                output[dataset]["vars"][
                    "HLT_PFHT1050" + out_label
                ] = events.HLT.PFHT1050
            output[dataset]["vars"]["ngood_ak4jets" + out_label] = ak.num(
                ak4jets
            ).to_list()
            output[dataset]["vars"]["ngood_ak4jets" + out_label] = ak.num(
                ak4jets
            ).to_list()
            if self.isMC:
                output[dataset]["vars"][
                    "Pileup_nTrueInt" + out_label
                ] = events.Pileup.nTrueInt
                GetPSWeights(events, output[dataset])  # Parton Shower weights
                GetPrefireWeights(self, events, output[dataset])  # Prefire weights
            output[dataset]["vars"]["PV_npvs" + out_label] = events.PV.npvs
            output[dataset]["vars"]["PV_npvsGood" + out_label] = events.PV.npvsGood

        # get gen SUEP kinematics
        SUEP_genMass = len(events) * [0]
        SUEP_genPt = len(events) * [0]
        SUEP_genEta = len(events) * [0]
        SUEP_genPhi = len(events) * [0]

        if self.isMC:
            genParts = self.getGenTracks(events)
            genSUEP = genParts[(abs(genParts.pdgID) == 25)]

            # we need to grab the last SUEP in the chain for each event
            SUEP_genMass = [g[-1].mass if len(g) > 0 else 0 for g in genSUEP]
            SUEP_genPt = [g[-1].pt if len(g) > 0 else 0 for g in genSUEP]
            SUEP_genPhi = [g[-1].phi if len(g) > 0 else 0 for g in genSUEP]
            SUEP_genEta = [g[-1].eta if len(g) > 0 else 0 for g in genSUEP]

        output[dataset]["vars"]["SUEP_genMass" + out_label] = SUEP_genMass
        output[dataset]["vars"]["SUEP_genPt" + out_label] = SUEP_genPt
        output[dataset]["vars"]["SUEP_genEta" + out_label] = SUEP_genEta
        output[dataset]["vars"]["SUEP_genPhi" + out_label] = SUEP_genPhi

        # MET
        output[dataset]["vars"]["MET_pt" + out_label] = events.MET.pt.to_list()
        output[dataset]["vars"]["MET_phi" + out_label] = events.MET.phi.to_list()
        output[dataset]["vars"]["MET_sumEt" + out_label] = events.MET.sumEt.to_list()

        # BTagging
        btag_scores = ak4jets.btagDeepB
        btag_scores = btag_scores[:, :4]
        btag_scores = ak.pad_none(btag_scores, target=4, axis=-1)
        btag_scores = ak.fill_none(btag_scores, value=-2)
        output[dataset]["vars"]["bTag_score_leading" + out_label] = btag_scores[
            :, 0
        ].to_list()
        output[dataset]["vars"]["bTag_score_subleading" + out_label] = btag_scores[
            :, 1
        ].to_list()
        output[dataset]["vars"]["bTag_score_third" + out_label] = btag_scores[
            :, 2
        ].to_list()
        output[dataset]["vars"]["bTag_score_fourth" + out_label] = btag_scores[
            :, 3
        ].to_list()

        # nMuons
        output[dataset]["vars"]["nMuons" + out_label] = ak.num(events.Muon).to_list()
        output[dataset]["vars"]["nMuons_highPurity" + out_label] = ak.sum(
            muons.highPurity, axis=-1
        ).to_list()
        output[dataset]["vars"]["nMuons_isPFcand" + out_label] = ak.sum(
            muons.isPFcand, axis=-1
        ).to_list()
        output[dataset]["vars"]["nMuons_looseId" + out_label] = ak.sum(
            muons.looseId, axis=-1
        ).to_list()
        output[dataset]["vars"]["nMuons_mediumId" + out_label] = ak.sum(
            muons.mediumId, axis=-1
        ).to_list()
        output[dataset]["vars"]["nMuons_tightId" + out_label] = ak.sum(
            muons.tightId, axis=-1
        ).to_list()
        output[dataset]["vars"]["nMuons_triggerIdLoose" + out_label] = ak.sum(
            muons.triggerIdLoose, axis=-1
        ).to_list()
        output[dataset]["vars"]["nMuons_isTracker" + out_label] = ak.sum(
            muons.isTracker, axis=-1
        ).to_list()
        output[dataset]["vars"]["muon_pt_mean" + out_label] = ak.mean(
            muons.pt, axis=-1
        ).to_list()
        output[dataset]["vars"]["muon_dxy_mean" + out_label] = ak.mean(
            abs(muons.dxy), axis=-1
        ).to_list()
        output[dataset]["vars"]["muon_dz_mean" + out_label] = ak.mean(
            abs(muons.dz), axis=-1
        ).to_list()
        output[dataset]["vars"]["muon_ip3d_mean" + out_label] = ak.mean(
            muons.ip3d, axis=-1
        ).to_list()
        output[dataset]["vars"]["muon_pt_leading" + out_label] = muons.pt[
            :, 0
        ].to_list()
        output[dataset]["vars"]["muon_pt_subleading" + out_label] = muons.pt[
            :, 1
        ].to_list()
        output[dataset]["vars"][
            "muon_miniPFRelIso_all_leading" + out_label
        ] = muons.miniPFRelIso_all[:, 0].to_list()
        output[dataset]["vars"][
            "muon_miniPFRelIso_all_subleading" + out_label
        ] = muons.miniPFRelIso_all[:, 1].to_list()
        output[dataset]["vars"]["muon_miniPFRelIso_all_mean" + out_label] = ak.mean(
            muons.miniPFRelIso_all, axis=-1
        ).to_list()
        output[dataset]["vars"]["muon_l_sl_deltaphi" + out_label] = abs(
            ak.materialized(muonsCollection[:, 0]).deltaphi(
                ak.materialized(muonsCollection[:, 1])
            )
        ).to_list()
        output[dataset]["vars"]["muon_l_sl_deltaeta" + out_label] = abs(
            ak.materialized(muonsCollection[:, 0]).deltaeta(
                ak.materialized(muonsCollection[:, 1])
            )
        ).to_list()
        output[dataset]["vars"]["muon_l_sl_deltaR" + out_label] = (
            ak.materialized(muonsCollection[:, 0])
            .deltaR(ak.materialized(muonsCollection[:, 1]))
            .to_list()
        )
        eigs_muons = SUEP_utils.sphericity(muons, 1.0)
        output[dataset]["vars"]["muon_S1" + out_label] = 1.5 * (
            eigs_muons[:, 1] + eigs_muons[:, 0]
        )

        """
        output[dataset]["vars"][
            "muon_interIsolation_0p2" + out_label
        ] = SUEP_utils.interIsolation(
            ak.materialized(muonsCollection[:, 0]),
            ak.materialized(electronsCollection),
            ak.materialized(muonsCollection),
            0.2,
        ).tolist()
        output[dataset]["vars"][
            "muon_interIsolation_0p4" + out_label
        ] = SUEP_utils.interIsolation(
            ak.materialized(muonsCollection[:, 0]),
            ak.materialized(electronsCollection),
            ak.materialized(muonsCollection),
            0.4,
        ).tolist()
        output[dataset]["vars"][
            "muon_interIsolation_0p8" + out_label
        ] = SUEP_utils.interIsolation(
            ak.materialized(muonsCollection[:, 0]),
            ak.materialized(electronsCollection),
            ak.materialized(muonsCollection),
            0.8,
        ).tolist()
        output[dataset]["vars"][
            "muon_interIsolation_1p6" + out_label
        ] = SUEP_utils.interIsolation(
            ak.materialized(muonsCollection[:, 0]),
            ak.materialized(electronsCollection),
            ak.materialized(muonsCollection),
            1.6,
        ).tolist()
        """

        # Eta ring variables
        output[dataset]["vars"][
            "nMuons_eta_ring_0p2" + out_label
        ] = SUEP_utils.n_eta_ring(muonsCollection, 0.2).to_list()
        output[dataset]["vars"][
            "nMuons_eta_ring_0p4" + out_label
        ] = SUEP_utils.n_eta_ring(muonsCollection, 0.4).to_list()
        output[dataset]["vars"][
            "nMuons_eta_ring_0p8" + out_label
        ] = SUEP_utils.n_eta_ring(muonsCollection, 0.8).to_list()
        output[dataset]["vars"][
            "nMuons_eta_ring_1p6" + out_label
        ] = SUEP_utils.n_eta_ring(muonsCollection, 1.6).to_list()

    def analysis(self, events, output, do_syst=False, col_label=""):
        """
        Main analyzer function.

        Parameters
        ----------
        events : NanoEvents
            NanoEvents object
        output : dict
            accumulator object
        do_syst : bool, optional
            whether to do systematics, by default False
        col_label : str, optional
            label for the collection, by default ""
        """

        # get dataset name
        dataset = events.metadata["dataset"]

        # some cutflow stuff
        output[dataset]["cutflow"].fill(
            len(events) * ["all events"], weight=events.genWeight
        )
        # output[dataset]["cutflow"].fill(
        #    ak.sum(events.HLT.PFHT430 == 1) * ["HLT_PFHT430"],
        #    weight=events[events.HLT.PFHT430 == 1].genWeight,
        # )
        output[dataset]["cutflow"].fill(
            ak.sum(events.HLT.TripleMu_5_3_3_Mass3p8_DZ == 1) * ["HLT_TripleMu_5_3_3"],
            weight=events[events.HLT.TripleMu_5_3_3_Mass3p8_DZ == 1].genWeight,
        )

        # golden jsons for offline data
        if not self.isMC:
            events = applyGoldenJSON(self, events)

        events = self.eventSelection(events)

        # filter by muons
        if self.trigger == "TripleMu":
            events, electrons, muons = self.muon_filter(events, 4)
        output[dataset]["cutflow"].fill(
            len(events) * ["nMuon_mediumId >= 4"], weight=events.genWeight
        )
        n_muons = ak.num(muons)
        output[dataset]["cutflow"].fill(
            len(events[n_muons >= 6]) * ["nMuon_mediumId >= 6"],
            weight=events[n_muons >= 6].genWeight,
        )

        # output empty dataframe if no events pass trigger
        if len(events) == 0:
            if self.debug:
                print("No events passed trigger. Saving empty outputs.")
            return output

        #####################################################################################
        # ---- Event level information
        #####################################################################################

        self.storeEventVars(
            events,
            output,
            electrons,
            muons,
            out_label=col_label,
        )

    def process(self, events):
        dataset = events.metadata["dataset"]
        cutflow = hist.Hist.new.StrCategory(
            [
                "all events",
                "HLT_TripleMu_5_3_3",
                "nMuon_mediumId >= 4",
                "nMuon_mediumId >= 6",
            ],
            name="cutflow",
            label="cutflow",
        ).Weight()
        output = {
            dataset: {
                "cutflow": cutflow,
                "gensumweight": processor.value_accumulator(float, 0),
                "vars": pandas_accumulator(pd.DataFrame()),
            },
        }

        # gen weights
        if self.isMC:
            self.gensumweight = ak.sum(events.genWeight)
            output[dataset]["gensumweight"].add(self.gensumweight)

        # run the analysis with the track systematics applied
        if self.isMC and self.do_syst:
            self.analysis(events, output, do_syst=True, col_label="_track_down")

        # run the analysis
        self.analysis(events, output)

        return output

    def postprocess(self, accumulator):
        pass
