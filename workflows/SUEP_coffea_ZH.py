"""
SUEP_coffea_ZH.py
Coffea producer for SUEP analysis. Uses fastjet package to recluster large jets:
https://github.com/scikit-hep/fastjet
Chad Freer, 2021
"""

import json
import os
import pathlib
import shutil
from typing import List, Optional

import awkward as ak
import correctionlib
import fastjet
import numpy as np
import pandas as pd
import pickle5 as pickle
import vector
from coffea import lookup_tools, processor

from workflows.CMS_corrections.btag_utils import btagcuts, doBTagWeights, getBTagEffs
from workflows.CMS_corrections.jetmet_utils import apply_jecs
from workflows.CMS_corrections.leptonscale_utils import doLeptonScaleVariations
from workflows.CMS_corrections.leptonsf_utils import doLeptonSFs, doTriggerSFs

vector.register_awkward()


class SUEP_cluster_ZH(processor.ProcessorABC):
    def __init__(
        self,
        isMC: int,
        era: int,
        sample: str,
        do_syst: bool,
        syst_var: str,
        weight_syst: bool,
        SRonly: bool,
        output_location: Optional[str],
        doOF: Optional[bool],
        isDY: Optional[bool],
    ) -> None:
        self.SRonly = SRonly
        self.output_location = output_location
        self.doOF = doOF
        self.isDY = isDY  # We need to save this to remove the overlap between the inclusive DY sample and the pT binned ones
        self.do_syst = do_syst and isMC != 0
        self.gensumweight = 1.0
        self.era = era
        self.isMC = isMC
        self.sample = sample
        self.isSignal = "ZH" in sample
        self.syst_var, self.syst_suffix = (
            (syst_var, f"_sys_{syst_var}") if do_syst and syst_var else ("", "")
        )
        self.weight_syst = weight_syst
        self.prefixes = {"SUEP": "SUEP"}
        # Set up for the histograms
        self._accumulator = processor.dict_accumulator({})

    @property
    def accumulator(self):
        return self._accumulator

    def sphericity(self, events, particles, r):
        # In principle here we already have ak.num(particles) != 0
        # Some sanity replacements just in case the boosting broke
        px = ak.nan_to_num(particles.px, 0)
        py = ak.nan_to_num(particles.py, 0)
        pz = ak.nan_to_num(particles.pz, 0)
        p = ak.nan_to_num(particles.p, 0)

        norm = np.squeeze(ak.sum(p**r, axis=1, keepdims=True))
        s = np.array(
            [
                [
                    ak.sum(px * px * p ** (r - 2.0), axis=1, keepdims=True) / norm,
                    ak.sum(px * py * p ** (r - 2.0), axis=1, keepdims=True) / norm,
                    ak.sum(px * pz * p ** (r - 2.0), axis=1, keepdims=True) / norm,
                ],
                [
                    ak.sum(py * px * p ** (r - 2.0), axis=1, keepdims=True) / norm,
                    ak.sum(py * py * p ** (r - 2.0), axis=1, keepdims=True) / norm,
                    ak.sum(py * pz * p ** (r - 2.0), axis=1, keepdims=True) / norm,
                ],
                [
                    ak.sum(pz * px * p ** (r - 2.0), axis=1, keepdims=True) / norm,
                    ak.sum(pz * py * p ** (r - 2.0), axis=1, keepdims=True) / norm,
                    ak.sum(pz * pz * p ** (r - 2.0), axis=1, keepdims=True) / norm,
                ],
            ]
        )
        s = np.squeeze(np.moveaxis(s, 2, 0), axis=3)
        s = np.nan_to_num(s, copy=False, nan=1.0, posinf=1.0, neginf=1.0)

        evals = np.sort(np.linalg.eigvals(s))
        # eval1 < eval2 < eval3
        return evals

    def rho(self, number, jet, tracks, deltaR, dr=0.05):
        r_start = number * dr
        r_end = (number + 1) * dr
        ring = (deltaR > r_start) & (deltaR < r_end)
        rho_values = ak.sum(tracks[ring].pt, axis=1) / (dr * jet.pt)
        return rho_values

    def ak_to_pandas(self, jet_collection: ak.Array) -> pd.DataFrame:
        output = pd.DataFrame()
        for field in ak.fields(jet_collection):
            prefix = self.prefixes.get(field, "")
            if len(prefix) > 0:
                for subfield in ak.fields(jet_collection[field]):
                    output[f"{prefix}_{subfield}"] = ak.to_numpy(
                        jet_collection[field][subfield]
                    )
            else:
                if not (isinstance(ak.to_numpy(jet_collection[field])[0], np.ndarray)):
                    output[field] = ak.to_numpy(jet_collection[field])
                else:
                    temp = ak.to_numpy(jet_collection[field])
                    output[field] = [[k for k in kk] for kk in temp]
        return output

    def h5store(
        self,
        store: pd.HDFStore,
        df: pd.DataFrame,
        fname: str,
        gname: str,
        **kwargs: float,
    ) -> None:
        store.put(gname, df)
        store.get_storer(gname).attrs.metadata = kwargs

    def save_dfs(self, dfs, df_names, fname=None):
        if not (fname):
            fname = "out.hdf5"
        subdirs = []
        store = pd.HDFStore(fname)
        if self.output_location is not None:
            # pandas to hdf5
            for out, gname in zip(dfs, df_names):
                if self.isMC:
                    metadata = dict(
                        gensumweight=self.gensumweight,
                        era=self.era,
                        mc=self.isMC,
                        sample=self.sample,
                    )
                    # metadata.update({"gensumweight":self.gensumweight})
                else:
                    metadata = dict(era=self.era, mc=self.isMC, sample=self.sample)

                store_fin = self.h5store(store, out, fname, gname, **metadata)

            store.close()
            self.dump_table(fname, self.output_location, subdirs)
        else:
            print("self.output_location is None")
            store.close()

    def dump_table(
        self, fname: str, location: str, subdirs: Optional[List[str]] = None
    ) -> None:
        subdirs = subdirs or []
        xrd_prefix = "root://"
        pfx_len = len(xrd_prefix)
        xrootd = False
        if xrd_prefix in location:
            try:
                import XRootD
                import XRootD.client

                xrootd = True
            except ImportError:
                raise ImportError(
                    "Install XRootD python bindings with: conda install -c conda-forge xroot"
                )
        local_file = (
            os.path.abspath(os.path.join(".", fname))
            if xrootd
            else os.path.join(".", fname)
        )
        merged_subdirs = "/".join(subdirs) if xrootd else os.path.sep.join(subdirs)
        destination = (
            location + merged_subdirs + f"/{fname}"
            if xrootd
            else os.path.join(location, os.path.join(merged_subdirs, fname))
        )
        if xrootd:
            copyproc = XRootD.client.CopyProcess()
            copyproc.add_job(local_file, destination)
            copyproc.prepare()
            copyproc.run()
            client = XRootD.client.FileSystem(
                location[: location[pfx_len:].find("/") + pfx_len]
            )
            status = client.locate(
                destination[destination[pfx_len:].find("/") + pfx_len + 1 :],
                XRootD.client.flags.OpenFlags.READ,
            )
            assert status[0].ok
            del client
            del copyproc
        else:
            dirname = os.path.dirname(destination)
            if not os.path.exists(dirname):
                pathlib.Path(dirname).mkdir(parents=True, exist_ok=True)
            if os.path.isfile(destination):
                if not os.path.samefile(local_file, destination):
                    shutil.copy2(local_file, destination)
                else:
                    fname = "condor_" + fname
                    destination = os.path.join(
                        location, os.path.join(merged_subdirs, fname)
                    )
                    shutil.copy2(local_file, destination)
            else:
                shutil.copy2(local_file, destination)
            assert os.path.isfile(destination)
        pathlib.Path(local_file).unlink()

    def selectByFilters(self, events):
        ### Apply MET filter selection (see https://twiki.cern.ch/twiki/bin/viewauth/CMS/MissingETOptionalFiltersRun2)
        if self.era == 2018 or self.era == 2017:
            cutAnyFilter = (
                (events.Flag.goodVertices)
                & (events.Flag.globalSuperTightHalo2016Filter)
                & (events.Flag.HBHENoiseFilter)
                & (events.Flag.HBHENoiseIsoFilter)
                & (events.Flag.EcalDeadCellTriggerPrimitiveFilter)
                & (events.Flag.BadPFMuonFilter)
                & (events.Flag.BadPFMuonDzFilter)
                & (events.Flag.eeBadScFilter)
                & (events.Flag.ecalBadCalibFilter)
            )
        if self.era == 2016 or self.era == 2015:  # 2015==2016APV
            cutAnyFilter = (
                (events.Flag.goodVertices)
                & (events.Flag.globalSuperTightHalo2016Filter)
                & (events.Flag.HBHENoiseFilter)
                & (events.Flag.HBHENoiseIsoFilter)
                & (events.Flag.EcalDeadCellTriggerPrimitiveFilter)
                & (events.Flag.BadPFMuonFilter)
                & (events.Flag.BadPFMuonDzFilter)
                & (events.Flag.eeBadScFilter)
            )
        return events[cutAnyFilter]

    def selectByTrigger(self, events, extraColls=[]):
        ### Apply trigger selection
        if self.era == 2018:
            cutAnyHLT = (
                (events.HLT.IsoMu24)
                | (events.HLT.Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8)
                | (events.HLT.Ele32_WPTight_Gsf)
                | (events.HLT.Ele23_Ele12_CaloIdL_TrackIdL_IsoVL)
            )  # | (events.HLT.Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ) | (events.HLT.Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL)
            return events[cutAnyHLT], [coll[cutAnyHLT] for coll in extraColls]
        if self.era == 2017:
            cutAnyHLT = (
                (events.HLT.IsoMu27)
                | (events.HLT.Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass8)
                | (events.HLT.Ele35_WPTight_Gsf)
                | (events.HLT.Ele23_Ele12_CaloIdL_TrackIdL_IsoVL)
            )  # | (events.HLT.Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ) | (events.HLT.Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ)
            return events[cutAnyHLT], [coll[cutAnyHLT] for coll in extraColls]
        if self.era == 2016 or self.era == 2015:  # 2015==2016APV
            cutAnyHLT = (
                (events.HLT.IsoMu24)
                | (events.HLT.Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ)
                | (events.HLT.Ele27_WPTight_Gsf)
                | (events.HLT.Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ)
            )  # | (events.HLT.Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ) | (events.HLT.Mu23_TrkIsoVVL_Ele8_CaloIdL_TrackIdL_IsoVL_DZ)
            return events[cutAnyHLT], [coll[cutAnyHLT] for coll in extraColls]
        return events, [coll[cutAnyHLT] for coll in extraColls]

    def selectByLeptons(self, events, extraColls=[]):
        ###lepton selection criteria--4momenta collection for plotting; aux are only needed for lepton scale corrections (and need to have 4 of them in both so we can merge the collections)
        if self.isMC and self.isSignal:
            muons = ak.zip(
                {
                    "pt": events.Muon.pt,
                    "eta": events.Muon.eta,
                    "phi": events.Muon.phi,
                    "mass": events.Muon.mass,
                    "charge": events.Muon.pdgId / (-13),
                    "pdgId": events.Muon.pdgId,
                    "aux1": events.Muon.genPartIdx,
                    "aux2": events.Muon.pt,
                    "aux3": events.Muon.nTrackerLayers,
                    "aux4": events.Muon.pt,
                },
                with_name="Momentum4D",
            )

            electrons = ak.zip(
                {
                    "pt": events.Electron.pt,
                    "eta": events.Electron.eta,
                    "phi": events.Electron.phi,
                    "mass": events.Electron.mass,
                    "charge": events.Electron.pdgId / (-11),
                    "pdgId": events.Electron.pdgId,
                    "aux1": events.Electron.dEscaleUp,
                    "aux2": events.Electron.dEscaleDown,
                    "aux3": events.Electron.dEsigmaUp,
                    "aux4": events.Electron.dEsigmaDown,
                },
                with_name="Momentum4D",
            )
        else:
            muons = ak.zip(
                {
                    "pt": events.Muon.pt,
                    "eta": events.Muon.eta,
                    "phi": events.Muon.phi,
                    "mass": events.Muon.mass,
                    "charge": events.Muon.pdgId / (-13),
                    "pdgId": events.Muon.pdgId,
                },
                with_name="Momentum4D",
            )

            electrons = ak.zip(
                {
                    "pt": events.Electron.pt,
                    "eta": events.Electron.eta,
                    "phi": events.Electron.phi,
                    "mass": events.Electron.mass,
                    "charge": events.Electron.pdgId / (-11),
                    "pdgId": events.Electron.pdgId,
                },
                with_name="Momentum4D",
            )

        ###  Some very simple selections on ID ###
        ###  Muons: loose ID + dxy dz cuts mimicking the medium prompt ID https://twiki.cern.ch/twiki/bin/viewauth/CMS/SWGuideMuonIdRun2
        ###  Electrons: loose ID + dxy dz cuts for promptness https://twiki.cern.ch/twiki/bin/view/CMS/EgammaCutBasedIdentification
        cutMuons = (
            (events.Muon.looseId)
            & (events.Muon.pt >= 10)
            & (abs(events.Muon.dxy) <= 0.02)
            & (abs(events.Muon.dz) <= 0.1)
            & (events.Muon.pfIsoId >= 2)
            & (abs(events.Muon.eta) < 2.4)
        )
        cutElectrons = (
            (events.Electron.pt >= 15)
            & (events.Electron.mvaFall17V2Iso_WP90)
            & (
                abs(events.Electron.dxy)
                < (0.05 + 0.05 * (abs(events.Electron.eta) > 1.479))
            )
            & (abs(events.Electron.dz) < (0.10 + 0.10 * (events.Electron.eta > 1.479)))
            & ((abs(events.Electron.eta) < 1.444) | (abs(events.Electron.eta) > 1.566))
            & (abs(events.Electron.eta) < 2.5)
        )

        ### Apply the cuts
        # Object selection. selMuons contain only the events that are filtered by cutMuons criteria.
        selMuons = muons[cutMuons]
        selElectrons = electrons[cutElectrons]
        ### Now global cuts to select events. Notice this means exactly two leptons with pT >= 10, and the leading one pT >= 25

        # cutHasTwoMuons imposes three conditions:
        #  First, number of muons (axis=1 means column. Each row is an event.) in an event is 2.
        #  Second, pt of the muons is greater than 25.
        #  Third, Sum of charge of muons should be 0. (because it originates from Z)
        if self.doOF:
            # Only for the OF sideband
            templeps = ak.concatenate([selMuons, selElectrons], axis=1)
            cutHasOFLeps = (
                (ak.num(templeps, axis=1) == 2)
                & (ak.max(templeps.pt, axis=1, mask_identity=False) >= 25)
                & (ak.sum(templeps.charge, axis=1) == 0)
            )
            events = events[cutHasOFLeps]
            selElectrons = selElectrons[cutHasOFLeps]
            selMuons = selMuons[cutHasOFLeps]
            cutOneAndOne = (ak.num(selElectrons) == 1) & (ak.num(selMuons) == 1)
            events = events[cutOneAndOne]
            selElectrons = selElectrons[cutOneAndOne]
            selMuons = selMuons[cutOneAndOne]

        else:
            cutHasTwoMuons = (
                (ak.num(selMuons, axis=1) == 2)
                & (ak.max(selMuons.pt, axis=1, mask_identity=False) >= 25)
                & (ak.sum(selMuons.charge, axis=1) == 0)
            )
            cutHasTwoElecs = (
                (ak.num(selElectrons, axis=1) == 2)
                & (ak.max(selElectrons.pt, axis=1, mask_identity=False) >= 25)
                & (ak.sum(selElectrons.charge, axis=1) == 0)
            )
            cutTwoLeps = (ak.num(selElectrons, axis=1) + ak.num(selMuons, axis=1)) < 3
            cutHasTwoLeps = ((cutHasTwoMuons) | (cutHasTwoElecs)) & cutTwoLeps
            ### Cut the events, also return the selected leptons for operation down the line
            events = events[cutHasTwoLeps]
            selElectrons = selElectrons[cutHasTwoLeps]
            selMuons = selMuons[cutHasTwoLeps]
        return events, selElectrons, selMuons

    def selectByJets(self, events, leptons=[], altJets=[], extraColls=[]):
        # These are just standard jets, as available in the nanoAOD
        if type(altJets) == type([]):  # I.E: build from events
            altJets = events.Jet
        if self.isMC:
            Jets = ak.zip(
                {
                    "pt": altJets.pt,
                    "eta": altJets.eta,
                    "phi": altJets.phi,
                    "mass": altJets.mass,
                    "btag": altJets.btagDeepFlavB,
                    "jetId": altJets.jetId,
                    "hadronFlavour": altJets.hadronFlavour,
                },
                with_name="Momentum4D",
            )

        else:
            Jets = ak.zip(
                {
                    "pt": altJets.pt,
                    "eta": altJets.eta,
                    "phi": altJets.phi,
                    "mass": altJets.mass,
                    "btag": altJets.btagDeepFlavB,
                    "jetId": altJets.jetId,
                },
                with_name="Momentum4D",
            )

        # Minimimum pT, eta requirements + jet-lepton recleaning. We set minimum pT at 20 here to allow for the syst variations later
        jetCut = (
            (Jets.pt > 20)
            & (abs(Jets.eta) < 2.5)
            & (Jets.deltaR(leptons[:, 0]) >= 0.4)
            & (Jets.deltaR(leptons[:, 1]) >= 0.4)
            & (Jets.jetId >= 6)
        )
        jets = Jets[jetCut]
        # The following is the collection of events and of jets
        return events, jets, [coll for coll in extraColls]

    def selectByTracks(self, events, leptons, extraColls=[]):
        ### PARTICLE FLOW CANDIDATES ###
        # Every particle in particle flow (clean PFCand matched to tracks collection)
        Cands = ak.zip(
            {
                "pt": events.PFCands.trkPt,
                "eta": events.PFCands.trkEta,
                "phi": events.PFCands.trkPhi,
                "mass": events.PFCands.mass,
                "pdgId": events.PFCands.pdgId,
            },
            with_name="Momentum4D",
        )

        cutPF = (
            (events.PFCands.fromPV > 1)
            & (events.PFCands.trkPt >= 1)
            & (abs(events.PFCands.trkEta) <= 2.5)
            & (abs(events.PFCands.dz) < 0.05)
            & (abs(events.PFCands.d0) < 0.05)
            & (events.PFCands.puppiWeight > 0.1)
        )
        Cleaned_cands = ak.packed(Cands[cutPF])

        ### LOST TRACKS ###
        # Unidentified tracks, usually SUEP Particles
        LostTracks = ak.zip(
            {
                "pt": events.lostTracks.pt,
                "eta": events.lostTracks.eta,
                "phi": events.lostTracks.phi,
                "mass": 0.0,
                "pdgId": -99,
            },
            with_name="Momentum4D",
        )

        cutLost = (
            (events.lostTracks.fromPV > 1)
            & (events.lostTracks.pt >= 1)
            & (abs(events.lostTracks.eta) <= 2.5)
            & (abs(events.lostTracks.dz) < 0.05)
            & (abs(events.lostTracks.d0) < 0.05)
            & (events.lostTracks.puppiWeight > 0.1)
        )
        Lost_Tracks_cands = ak.packed(LostTracks[cutLost])

        # dimensions of tracks = events x tracks in event x 4 momenta
        totalTracks = ak.concatenate([Cleaned_cands, Lost_Tracks_cands], axis=1)

        # Sorting out the tracks that overlap with leptons
        totalTracks = totalTracks[
            (totalTracks.deltaR(leptons[:, 0]) >= 0.4)
            & (totalTracks.deltaR(leptons[:, 1]) >= 0.4)
        ]
        nTracks = ak.num(totalTracks, axis=1)
        return events, totalTracks, nTracks, [coll for coll in extraColls]

    def clusterizeTracks(self, events, tracks):
        # anti-kt, dR=1.5 jets
        jetdef = fastjet.JetDefinition(fastjet.antikt_algorithm, 1.5)
        cluster = fastjet.ClusterSequence(tracks, jetdef)
        ak15_jets = ak.with_name(
            cluster.inclusive_jets(min_pt=0), "Momentum4D"
        )  # These are the ak15_jets
        ak15_consts = ak.with_name(
            cluster.constituents(min_pt=0), "Momentum4D"
        )  # And these are the collections of constituents of the ak15_jets
        clidx = cluster.constituent_index()
        return events, ak15_jets, ak15_consts, clidx

    def selectByGEN(self, events):
        GenParts = ak.zip(
            {
                "pt": events.GenPart.pt,
                "eta": events.GenPart.eta,
                "phi": events.GenPart.phi,
                "mass": events.GenPart.mass,
                "pdgId": events.GenPart.pdgId,
            },
            with_name="Momentum4D",
        )
        if self.isDY:
            # Build the Zpt from leptons for the DY sample bug. Somehow awkward but needed as gammastar is not saved...
            cutgenLepsNeg = (
                (events.GenPart.pdgId >= 11)
                & (events.GenPart.pdgId <= 16)
                & (abs(events.GenPart.status - 25) < 6)
            )  # Leptons from the hard scattering
            cutgenLepsPos = (
                (events.GenPart.pdgId >= -16)
                & (events.GenPart.pdgId <= -11)
                & (abs(events.GenPart.status - 25) < 6)
            )  # Antileptons from the hard scattering
            cutgenZ = (events.GenPart.pdgId == 23) & (events.GenPart.status == 22)
            genlepsPos = ak.pad_none(GenParts[cutgenLepsPos], 1, clip=True)
            genlepsNeg = ak.pad_none(GenParts[cutgenLepsNeg], 1, clip=True)
            genZfromZ = GenParts[cutgenZ]
            genZfromZpadded = ak.pad_none(genZfromZ, 1, clip=True)[:, 0]
            genZfromleps = genlepsPos[:, 0] + genlepsNeg[:, 0]
            Zpt = ak.where(ak.num(genZfromZ) >= 1, genZfromZpadded.pt, genZfromleps.pt)
            return events, Zpt
        else:
            cutgenZ = (events.GenPart.pdgId == 23) & (
                (events.GenPart.status == 62) | (events.GenPart.status == 22)
            )
            cutgenH = (events.GenPart.pdgId == 25) & (
                (events.GenPart.status == 62) | (events.GenPart.status == 22)
            )
            cutgenSUEP = (events.GenPart.pdgId == 999999) & (events.GenPart.status == 2)
            return events, GenParts[cutgenZ], GenParts[cutgenH], GenParts[cutgenSUEP]

    def shouldContinueAfterCut(self, events, out):
        # if debug: print("Conversion to pandas...")
        if len(events) == 0:
            outdfs = []
            outcols = []
            print("No events pass cut, stopping...")
            for channel in out.keys():
                outcols.append(channel)
                if len(out[channel][0]) == 0:
                    print("Create empty frame")
                    outdfs = pd.DataFrame(["empty"], columns=["empty"])
                else:
                    if self.isMC:
                        out[channel][0]["genweight"] = out[channel][1].genWeight[:]

                if not isinstance(out[channel][0], pd.DataFrame):
                    out[channel][0] = self.ak_to_pandas(out[channel][0])
            self.save_dfs(
                [out[key][0] for key in out], [key for key in out], self.chunkTag
            )

            return False
        else:
            return True

    def process(self, events):
        np.random.seed(
            max(0, min(events.event[0], 2**31))
        )  # This ensures reproducibility of results (i.e. for the random track dropping), while also getting different random numbers per file to avoid biases (like always dropping the first track, etc.)
        debug = True  # If we want some prints in the middle
        self.chunkTag = "out_%i_%i_%i.hdf5" % (
            events.event[0],
            events.luminosityBlock[0],
            events.run[0],
        )  # Unique tag to get different outputs per tag
        fullFile = self.output_location + "/" + self.chunkTag
        if debug:
            print("Will check file %s" % fullFile)
        if os.path.isfile(fullFile):
            print("SKIP")
            return self.accumulator.identity()

        # ------------------------------- CONFIGURABLES -------------------------------------
        self.doTracks = (
            True  # Make it false, and it will speed things up but not run the tracks
        )
        self.doClusters = (
            True  # Make it false, and it will speed things up but not run the clusters
        )
        self.doGen = (
            False if not (self.isDY) else True
        )  # In case we want info on the gen level, we do need it for the buggy DY samples (to get proper stitching)

        # Main processor code
        # ------------------------------------------------------------------------------------
        # ------------------------------- DEFINE CHANNELS ------------------------------------
        # ------------------------------------------------------------------------------------

        accumulator = self.accumulator.identity()
        # Each entry is one selection level
        outputs = {
            "twoleptons": [{}, []],  # Has Two Leptons, pT and Trigger requirements
            "onecluster": [{}, []],  # At least one cluster is found
            "SR": [{}, []],  # SR selection
        }

        # Data dependent stuff
        dataset = events.metadata["dataset"]
        if self.isMC:
            self.gensumweight = ak.sum(events.genWeight)
        if not (self.isMC):
            self.doGen = False

        # ------------------------------------------------------------------------------------
        # ----------------------- OBJECT LOADING AND SELECTION -------------------------------
        # ------------------------------------------------------------------------------------
        # MET filters
        if debug:
            print("Applying MET requirements.... %i events in" % len(events))
        self.events = self.selectByFilters(events)
        if not (self.shouldContinueAfterCut(self.events, outputs)):
            return accumulator  # If we have no events, we simply stop
        if debug:
            print(
                "%i events pass METFilter cuts. Applying lepton requirements...."
                % len(self.events)
            )

        # Lepton selection
        self.events, self.electrons, self.muons = self.selectByLeptons(self.events)[:3]
        if not (self.shouldContinueAfterCut(self.events, outputs)):
            return accumulator  # If we have no events, we simply stop
        if debug:
            print(
                "%i events pass lepton cuts. Applying trigger requirements...."
                % len(self.events)
            )

        # Trigger selection
        self.events, [self.electrons, self.muons] = self.selectByTrigger(
            self.events, [self.electrons, self.muons]
        )
        # Here we join muons and electrons into leptons and sort them by pT
        self.leptons = ak.concatenate([self.electrons, self.muons], axis=1)
        highpt_leptons = ak.argsort(
            self.leptons.pt, axis=1, ascending=False, stable=True
        )
        self.leptons = self.leptons[highpt_leptons]
        self.electrons = self.electrons[
            ak.argsort(self.electrons.pt, axis=1, ascending=False, stable=True)
        ]
        self.muons = self.muons[
            ak.argsort(self.muons.pt, axis=1, ascending=False, stable=True)
        ]
        if not (self.shouldContinueAfterCut(self.events, outputs)):
            return accumulator
        if debug:
            print("%i events pass trigger cuts. Selecting jets..." % len(self.events))

        # Jet selection
        self.events, self.jets = self.selectByJets(self.events, self.leptons)[
            :2
        ]  # Leptons are needed to do jet-lepton cleaning
        # Sorting jets by pt.
        highpt_jets = ak.argsort(self.jets.pt, axis=1, ascending=False, stable=True)
        self.jets = self.jets[highpt_jets]
        if not (self.shouldContinueAfterCut(self.events, outputs)):
            return accumulator
        if debug:
            print("%i events pass jet cuts. Selecting tracks..." % len(self.events))

        if self.doTracks:
            # Select tracks
            # Right now no track-based event-level cuts, only selecting tracks
            self.events, self.tracks = self.selectByTracks(self.events, self.leptons)[
                :2
            ]  # Again, we need leptons to clean the tracks
            if not (self.shouldContinueAfterCut(self.events, outputs)):
                return accumulator
            if debug:
                print(
                    "%i events pass track cuts. Doing track clustering..."
                    % len(self.events)
                )
            if self.doClusters:
                # Clusterize to get SUEP candidates
                (
                    self.events,
                    self.clusters,
                    self.constituents,
                    clidx,
                ) = self.clusterizeTracks(self.events, self.tracks)[:4]
                highpt_clusters = ak.argsort(
                    self.clusters.pt, axis=1, ascending=False, stable=True
                )
                self.clusters = self.clusters[highpt_clusters]
                self.constituents = self.constituents[highpt_clusters]
                clidx = clidx[highpt_clusters]
                self.constituents = ak.zip(
                    {
                        "pt": self.constituents.pt,
                        "eta": self.constituents.eta,
                        "phi": self.constituents.phi,
                        "mass": self.constituents.mass,
                        "pdgId": ak.unflatten(
                            self.tracks.pdgId[ak.flatten(clidx, axis=2)],
                            ak.flatten(ak.num(self.constituents, axis=2)),
                            axis=1,
                        ),
                    },
                    with_name="Momentum4D",
                )

        if self.doGen:
            if debug:
                print("Will run gen-level variables as well")
            if self.isDY:
                self.events, self.Zpt = self.selectByGEN(self.events)[:2]
            else:
                self.events, self.genZ, self.genH, self.genSUEP = self.selectByGEN(
                    self.events
                )[:4]
            if not (self.shouldContinueAfterCut(self.events, outputs)):
                return accumulator
            if debug:
                print("%i events pass gen cuts. Doing more stuff..." % len(self.events))

        ##### Finally, build additional composite objects
        # First the Z candidates
        self.Zcands = self.leptons[:, 0] + self.leptons[:, 1]
        if debug:
            print("Now build systematics and weights")
        if self.isMC:
            ### All the weights ###
            self.btagweights = doBTagWeights(
                self.events, self.jets, self.era, "L", do_syst=self.do_syst
            )  # Does not change selection
            self.puweights = self.doPUWeights(self.events)  # Does not change selection
            self.l1preweights = self.doPrefireWeights(self.events)
            self.triggerSFs = doTriggerSFs(
                self.electrons, self.muons, self.era, self.do_syst
            )
            self.leptonSFs = doLeptonSFs(self.electrons, self.muons, self.era)

        # ------------------------------------------------------------------------------
        # ------------------------------- UNCERTAINTIES --------------------------------
        # ------------------------------------------------------------------------------
        self.jetsVar = {
            "": self.jets
        }  # If not activated, always central jet collection
        self.tracksVar = {
            "": self.tracks
        }  # If not activated, always central track collection
        self.lepsVar = {
            "": self.leptons
        }  # If not activated, always central lepton collection
        self.varsToDo = [""]  # If not activated just do nominal yields
        if self.do_syst:
            self.isrweights = self.doISRWeights(
                self.events
            )  # Does not change selection
            if self.isSignal:
                # JECS and lepton scales are heavy, so we only run them for signal #
                self.jetsVar = self.doJECJERVariations(
                    self.events, self.jets
                )  # Does change selection, entry "" is central jets, entry "JECX" is JECUp/JECDown, entry "JERX" is JERUp/JerDown
                self.lepsVar = doLeptonScaleVariations(
                    self.events, self.leptons, self.era
                )

            if self.doTracks:
                if debug:
                    print("Start tracks dropping")
                self.tracksVar = self.doTracksDropping(
                    self.events, self.tracks
                )  # Does change selection, entry "" is central, "TRACKUP" is tracks modified
            outputsnew = {}
            for channel in outputs:
                outputsnew[channel] = outputs[channel]
                if self.isSignal:  # Only do JEC/JER/LeptonScale for signal
                    outputsnew[channel + "_JECUP"] = outputs[channel]
                    outputsnew[channel + "_JECDOWN"] = outputs[channel]
                    outputsnew[channel + "_JERUP"] = outputs[channel]
                    outputsnew[channel + "_JERDOWN"] = outputs[channel]
                    outputsnew[channel + "_ElScaleUp"] = outputs[channel]
                    outputsnew[channel + "_ElScaleDown"] = outputs[channel]
                    outputsnew[channel + "_ElSigmaUp"] = outputs[channel]
                    outputsnew[channel + "_ElSigmaDown"] = outputs[channel]
                    outputsnew[channel + "_MuScaleUp"] = outputs[channel]
                    outputsnew[channel + "_MuScaleDown"] = outputs[channel]

                outputsnew[channel + "_TRACKUP"] = outputs[
                    channel
                ]  # Tracks done for all as they change shapes significantly
            outputs = outputsnew
            self.varsToDo = (
                [
                    "",
                    "_JECUP",
                    "_JECDOWN",
                    "_JERUP",
                    "_JERDOWN",
                    "_TRACKUP",
                    "_ElScaleUp",
                    "_ElScaleDown",
                    "_ElSigmaUp",
                    "_ElSigmaDown",
                    "_MuScaleUp",
                    "_MuScaleDown",
                ]
                if self.isSignal
                else ["", "_TRACKUP"]
            )

        # ------------------------------------------------------------------------------
        # ------------------------------- SELECTION + PLOTTING -------------------------
        # ------------------------------------------------------------------------------
        self.var = ""  # To keep track of the systematic variation: "" == nominal
        if len(self.varsToDo) > 1:
            self.saveAllCollections()
        for var in self.varsToDo:
            self.var = var
            # Reset collections for syst variation
            if len(self.varsToDo) == 1:
                highpt_jets = ak.argsort(
                    self.jets.pt, axis=1, ascending=False, stable=True
                )
                self.jets = self.jets[highpt_jets]
                self.jets = self.jets[(self.jets.pt >= 30)]

            if len(self.varsToDo) > 1:
                self.resetAllCollections()
                self.jets = self.jetsVar[""]
                self.leptons = self.lepsVar[""]
                self.tracks = self.tracksVar[""]
                if var in self.jetsVar:
                    if debug:
                        print("Replacing jets for var %s" % var)
                    self.jets = self.jetsVar[var]
                if var in self.tracksVar:
                    if debug:
                        print("Replacing tracks for var %s" % var)
                    self.tracks = self.tracksVar[var]
                if var in self.lepsVar:
                    if debug:
                        print("Replacing leptons for var %s" % var)
                    self.leptons = self.lepsVar[var]
                self.Zcands = self.leptons[:, 0] + self.leptons[:, 1]
                highpt_jets = ak.argsort(
                    self.jets.pt, axis=1, ascending=False, stable=True
                )
                self.jets = self.jets[highpt_jets]
                self.jets = self.jets[(self.jets.pt >= 30)]

            self.isSpherable = (
                False  # So we don't do sphericity plots until we have clusters
            )
            self.isClusterable = (
                False  # So we don't try to compute sphericity if clusters are empty
            )
            outputs["twoleptons" + var] = [
                self.doAllPlots("twoleptons" + var, debug),
                self.events,
            ]
            if not (self.shouldContinueAfterCut(self.events, outputs)):
                return accumulator
            if debug:
                print(
                    "%i events pass twoleptons cuts. Doing more stuff..."
                    % len(self.events)
                )

            if self.doTracks:
                cutOneTrack = ak.num(self.tracks) != 0
                self.applyCutToAllCollections(cutOneTrack)
                self.isSpherable = True  # So we do sphericity plots
                if not (self.shouldContinueAfterCut(self.events, outputs)):
                    return accumulator
                if debug:
                    print(
                        "%i events pass onetrack cuts. Doing more stuff..."
                        % len(self.events)
                    )
                if self.doClusters:
                    cutOneCluster = ak.num(self.clusters) != 0
                    self.applyCutToAllCollections(cutOneCluster)
                    self.isClusterable = True  # So we do cluster plots
                    if (
                        self.doClusters and self.var != ""
                    ):  # As tracks might have changed, need to redo this step
                        (
                            self.events,
                            self.clusters,
                            self.constituents,
                            clidx,
                        ) = self.clusterizeTracks(self.events, self.tracks)[:4]
                        highpt_clusters = ak.argsort(
                            self.clusters.pt, axis=1, ascending=False, stable=True
                        )
                        self.clusters = self.clusters[highpt_clusters]
                        self.constituents = self.constituents[highpt_clusters]
                        clidx = clidx[highpt_clusters]
                        self.constituents = ak.zip(
                            {
                                "pt": self.constituents.pt,
                                "eta": self.constituents.eta,
                                "phi": self.constituents.phi,
                                "mass": self.constituents.mass,
                                "pdgId": ak.unflatten(
                                    self.tracks.pdgId[ak.flatten(clidx, axis=2)],
                                    ak.flatten(ak.num(self.constituents, axis=2)),
                                    axis=1,
                                ),
                            },
                            with_name="Momentum4D",
                        )
                    # Finally, the analysis cutflow #
                    outputs["onecluster" + var] = [
                        self.doAllPlots("onecluster" + var, debug),
                        self.events,
                    ]
                    if not (self.shouldContinueAfterCut(self.events, outputs)):
                        return accumulator
                    if debug:
                        print(
                            "%i events pass onecluster cuts. Doing more stuff..."
                            % len(self.events)
                        )
                    cutZm = abs(self.Zcands.mass - 90) < 30
                    self.applyCutToAllCollections(cutZm)
                    if not (self.shouldContinueAfterCut(self.events, outputs)):
                        return accumulator
                    if debug:
                        print(
                            "%i events pass Zm cuts. Doing more stuff..."
                            % len(self.events)
                        )
                    cutZpt = self.Zcands.pt > 25
                    self.applyCutToAllCollections(cutZpt)
                    if not (self.shouldContinueAfterCut(self.events, outputs)):
                        return accumulator
                    if debug:
                        print(
                            "%i events pass Zpt cuts. Doing more stuff..."
                            % len(self.events)
                        )
                    cut0tag = (
                        ak.sum(
                            self.jets.btag >= self.btagcuts("Loose", self.era), axis=1
                        )
                        == 0
                    )
                    self.applyCutToAllCollections(cut0tag)
                    if not (self.shouldContinueAfterCut(self.events, outputs)):
                        return accumulator
                    if debug:
                        print(
                            "%i events pass 1tag cuts. Doing more stuff..."
                            % len(self.events)
                        )
                    cutclusterpt60 = self.clusters.pt[:, 0] >= 60
                    self.applyCutToAllCollections(cutclusterpt60)
                    if not (self.shouldContinueAfterCut(self.events, outputs)):
                        return accumulator
                    if debug:
                        print(
                            "%i events pass clusterpt cuts. Doing more stuff..."
                            % len(self.events)
                        )
                    outputs["SR" + var] = [
                        self.doAllPlots("SR" + var, debug),
                        self.events,
                    ]

        # ------------------------------------------------------------------------------
        # -------------------------------- SAVING --------------------------------------
        # ------------------------------------------------------------------------------
        todel = []
        if self.SRonly:  # Lightweight, save only SR stuff
            for out in outputs:
                if not ("SR" in out):
                    todel.append(out)
            for t in todel:
                del outputs[t]

        for out in outputs:
            if out in todel:
                continue
            if self.isMC:
                outputs[out][0]["genweight"] = outputs[out][1].genWeight[:]
            if debug:
                print("Conversion to pandas...")
            if not isinstance(outputs[out][0], pd.DataFrame):
                if debug:
                    print("......%s" % out)
                outputs[out][0] = self.ak_to_pandas(outputs[out][0])

        if debug:
            print("DFS saving....")

        self.save_dfs(
            [outputs[key][0] for key in outputs],
            [key for key in outputs],
            self.chunkTag,
        )

        return accumulator

    def applyCutToAllCollections(
        self, cut
    ):  # Cut has to by a selection applicable across all collections, i.e. something defined per event
        self.events = self.events[cut]
        self.electrons = self.electrons[cut]
        self.muons = self.muons[cut]
        self.leptons = self.leptons[cut]
        self.jets = self.jets[cut]
        self.Zcands = self.Zcands[cut]
        if self.isMC:
            for var in self.btagweights:
                self.btagweights[var] = self.btagweights[var][cut]

            for var in self.puweights:
                self.puweights[var] = self.puweights[var][cut]

            for var in self.l1preweights:
                self.l1preweights[var] = self.l1preweights[var][cut]

            for var in self.triggerSFs:
                self.triggerSFs[var] = self.triggerSFs[var][cut]

            for var in self.leptonSFs:
                self.leptonSFs[var] = self.leptonSFs[var][cut]
        if self.doTracks:
            self.tracks = self.tracks[cut]
            if self.doClusters:
                self.clusters = self.clusters[cut]
                self.constituents = self.constituents[cut]

        if self.doGen:
            if self.isDY:
                self.Zpt = self.Zpt[cut]
            else:
                self.genZ = self.genZ[cut]
                self.genH = self.genH[cut]
                self.genSUEP = self.genSUEP[cut]
        if self.do_syst and self.var == "":
            for var in self.isrweights:
                self.isrweights[var] = self.isrweights[var][cut]

    def saveAllCollections(
        self,
    ):  # Save collections before cutting, to reset when dealing with systematics
        self.safeevents = self.events
        self.safeelectrons = self.electrons
        self.safemuons = self.muons
        self.safeleptons = self.leptons
        self.safejets = self.jets
        self.safeZcands = self.Zcands
        if self.isMC:
            self.safebtagweights = {}
            for var in self.btagweights:
                self.safebtagweights[var] = self.btagweights[var]
            self.safepuweights = {}
            for var in self.puweights:
                self.safepuweights[var] = self.puweights[var]
            self.safel1preweights = {}
            for var in self.l1preweights:
                self.safel1preweights[var] = self.l1preweights[var]
            self.safetriggerSFs = {}
            for var in self.triggerSFs:
                self.safetriggerSFs[var] = self.triggerSFs[var]
            self.safeleptonSFs = {}
            for var in self.leptonSFs:
                self.safeleptonSFs[var] = self.leptonSFs[var]

        if self.doTracks:
            self.safetracks = self.tracks
            if self.doClusters:
                self.safeclusters = self.clusters
                self.safeconstituents = self.constituents

        if self.doGen:
            if self.isDY:
                self.safeZpt = self.Zpt
            else:
                self.safegenZ = self.genZ
                self.safegenH = self.genH
                self.safegenSUEP = self.genSUEP

    def resetAllCollections(
        self,
    ):  # Reset collections to before cutting, useful when dealing with systematics
        self.events = self.safeevents
        self.electrons = self.safeelectrons
        self.muons = self.safemuons
        self.leptons = self.safeleptons
        self.jets = self.safejets
        self.Zcands = self.safeZcands
        if self.isMC:
            for var in self.btagweights:
                self.btagweights[var] = self.safebtagweights[var]
            for var in self.puweights:
                self.puweights[var] = self.safepuweights[var]
            for var in self.l1preweights:
                self.l1preweights[var] = self.safel1preweights[var]
            for var in self.triggerSFs:
                self.triggerSFs[var] = self.safetriggerSFs[var]
            for var in self.leptonSFs:
                self.leptonSFs[var] = self.safeleptonSFs[var]

        if self.doTracks:
            self.tracks = self.safetracks
            if self.doClusters:
                self.clusters = self.safeclusters
                self.constituents = self.safeconstituents

        if self.doGen:
            if self.isDY:
                self.Zpt = self.safeZpt
            else:
                self.genZ = self.safegenZ
                self.genH = self.safegenH
                self.genSUEP = self.safegenSUEP

    def doISRWeights(self, events):
        out = {}
        if len(events.PSWeight[0, :]) > 3:
            out["PSWeight_ISRUP"] = events.PSWeight[:, 0]
            out["PSWeight_FSRUP"] = events.PSWeight[:, 1]
            out["PSWeight_ISRDOWN"] = events.PSWeight[:, 2]
            out["PSWeight_FSRDOWN"] = events.PSWeight[:, 3]
        else:
            out["PSWeight_ISRUP"] = ak.Array([1.0] * len(events))
            out["PSWeight_FSRUP"] = ak.Array([1.0] * len(events))
            out["PSWeight_ISRDOWN"] = ak.Array([1.0] * len(events))
            out["PSWeight_FSRDOWN"] = ak.Array([1.0] * len(events))
        return out

    def doPrefireWeights(self, events):
        out = {}
        if self.era == 2016 or self.era == 2017 or self.era == 2015:  # 2015 == 2016APV
            out["L1prefire_nom"] = events.L1PreFiringWeight.Nom
            out["L1prefire_up"] = events.L1PreFiringWeight.Up
            out["L1prefire_down"] = events.L1PreFiringWeight.Dn
        else:
            out["L1prefire_nom"] = ak.Array([1.0] * len(events))
            out["L1prefire_up"] = ak.Array([1.0] * len(events))
            out["L1prefire_down"] = ak.Array([1.0] * len(events))
        return out

    def doPUWeights(self, events):
        if self.era == 2016 or self.era == 2015:
            weightsNom = [
                0.26018077,
                0.36628044,
                0.88641703,
                0.8539137,
                0.9363881,
                0.79212093,
                0.7889529,
                0.7477801,
                0.70349234,
                0.6906215,
                0.7222935,
                0.7636166,
                0.80246115,
                0.8335345,
                0.86026233,
                0.8914726,
                0.92116714,
                0.9473544,
                0.96650285,
                0.97886634,
                0.98783386,
                0.99871737,
                1.013512,
                1.0294081,
                1.0416297,
                1.0514929,
                1.0588624,
                1.0660558,
                1.0753927,
                1.0872909,
                1.099803,
                1.1150165,
                1.1363182,
                1.1628817,
                1.1909355,
                1.2174218,
                1.2473173,
                1.2743632,
                1.3033608,
                1.3298254,
                1.3512429,
                1.3726808,
                1.4003977,
                1.4352003,
                1.4816226,
                1.4986303,
                1.5163825,
                1.5327102,
                1.4839698,
                1.3851563,
                1.2323056,
                1.078277,
                0.87726563,
                0.7286223,
                0.57999974,
                0.41314062,
                0.33262798,
                0.22232734,
                0.21020901,
                0.17870684,
                0.14504391,
                0.13231093,
                0.14085712,
                0.12380952,
                0.073247686,
                0.05144765,
                0.08378049,
                0.061047662,
                0.042274423,
                0.1293808,
                0.20961441,
                0.020201342,
                0.02054449,
                0.12478866,
                0.0447785,
                0.0037996888,
                0.024160748,
                0.008747019,
                1.0,
                0.0015699057,
                0.00015667515,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
            ]
            weightsUp = [
                0.22796525,
                0.3130057,
                0.8050016,
                0.7574693,
                0.81391674,
                0.6178775,
                0.5898711,
                0.578412,
                0.54505455,
                0.5216448,
                0.54068935,
                0.5790399,
                0.62146866,
                0.6597525,
                0.6952342,
                0.7410549,
                0.7937431,
                0.8439945,
                0.8842998,
                0.91400003,
                0.93553334,
                0.9558551,
                0.98172486,
                1.0135264,
                1.0456519,
                1.0769331,
                1.105947,
                1.1351252,
                1.1674252,
                1.2039505,
                1.2433747,
                1.2893019,
                1.3474729,
                1.4191604,
                1.5020808,
                1.594486,
                1.7053531,
                1.8293147,
                1.9766083,
                2.144738,
                2.3333235,
                2.554826,
                2.8267605,
                3.1591506,
                3.5723693,
                3.9708588,
                4.42342,
                4.922977,
                5.2384787,
                5.3511305,
                5.17285,
                4.866292,
                4.194612,
                3.6213768,
                2.9274712,
                2.0628958,
                1.6006144,
                1.0088502,
                0.88807166,
                0.7020976,
                0.5351205,
                0.46630746,
                0.48375195,
                0.42228788,
                0.25211716,
                0.1809277,
                0.30376983,
                0.22966583,
                0.16573903,
                0.5301713,
                0.8995535,
                0.09091304,
                0.09704618,
                0.6191172,
                0.23344412,
                0.020822363,
                0.13921489,
                0.053007763,
                1.0,
                0.010531503,
                0.0011062882,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
            ]
            weightsDn = [
                0.29942793,
                0.44098234,
                0.97288865,
                0.96896553,
                1.1037534,
                1.055607,
                1.0567397,
                0.9758199,
                0.9264562,
                0.9344218,
                0.97321296,
                1.0085037,
                1.0343684,
                1.0503454,
                1.0548458,
                1.0524924,
                1.0470169,
                1.0442938,
                1.0408691,
                1.0372617,
                1.0345886,
                1.032288,
                1.0288227,
                1.0228273,
                1.0125073,
                1.0004466,
                0.9864025,
                0.97216034,
                0.9593289,
                0.94749963,
                0.933951,
                0.9194741,
                0.90587735,
                0.89158195,
                0.8730761,
                0.84793437,
                0.8196417,
                0.7841922,
                0.7452712,
                0.70118177,
                0.6522694,
                0.6027846,
                0.55654156,
                0.5142841,
                0.47776642,
                0.4349034,
                0.39703262,
                0.36407393,
                0.32272804,
                0.27954924,
                0.2351147,
                0.19918405,
                0.16142318,
                0.1376844,
                0.11573233,
                0.08875457,
                0.07742832,
                0.055668335,
                0.055628337,
                0.048876766,
                0.04010227,
                0.036289115,
                0.037768353,
                0.032112747,
                0.018243475,
                0.012244026,
                0.018989949,
                0.01315021,
                0.008641727,
                0.02507445,
                0.0384884,
                0.0035125734,
                0.0033815438,
                0.019437198,
                0.006598452,
                0.000529551,
                0.0031836047,
                0.0010893325,
                1.0,
                0.000174414,
                1.6427193e-05,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
            ]
        if self.era == 2017:
            weightsNom = [
                0.56062746,
                0.7356768,
                0.53791887,
                1.2182341,
                0.80225897,
                0.9162976,
                1.0035353,
                0.925153,
                0.6846759,
                0.7131643,
                0.7505122,
                0.81651765,
                0.83766913,
                0.84635574,
                0.84402835,
                0.84390676,
                0.8666289,
                0.8929616,
                0.91802394,
                0.9357903,
                0.9556427,
                0.9721437,
                0.981183,
                0.9832299,
                0.9780843,
                0.97595227,
                0.9765533,
                0.98268455,
                0.9903925,
                0.9981199,
                1.0078238,
                1.0190171,
                1.0302624,
                1.0435067,
                1.0528812,
                1.0608327,
                1.067958,
                1.0711179,
                1.0701725,
                1.0608593,
                1.0426793,
                1.0292999,
                1.0110271,
                0.9897588,
                0.9667128,
                0.95965326,
                0.9605182,
                0.97262126,
                0.98598033,
                1.0203637,
                1.0798059,
                1.1309571,
                1.1752437,
                1.2064251,
                1.1893482,
                1.175364,
                1.1530457,
                1.1393427,
                1.1491059,
                1.1861941,
                1.229691,
                1.2843318,
                1.3475113,
                1.4150772,
                1.4889932,
                1.5639929,
                1.7605078,
                2.4973722,
                3.6048465,
                3.5103703,
                5.5461583,
                16.473486,
                30.208288,
                159.5513,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
            ]
            weightsUp = [
                0.52825576,
                0.6443592,
                0.43563217,
                1.2328962,
                0.681312,
                0.8173814,
                0.9699813,
                0.84744084,
                0.5507151,
                0.46106908,
                0.5342559,
                0.5669717,
                0.6342342,
                0.6532066,
                0.64575315,
                0.63797486,
                0.6526295,
                0.68949956,
                0.7374308,
                0.77661604,
                0.8149431,
                0.8510646,
                0.8780491,
                0.8933845,
                0.8956852,
                0.89616364,
                0.9014064,
                0.9177949,
                0.94078916,
                0.9670744,
                0.99642795,
                1.0270404,
                1.0584179,
                1.0940686,
                1.1266212,
                1.1550058,
                1.1773461,
                1.1900263,
                1.1926558,
                1.1774322,
                1.138953,
                1.0896544,
                1.0225472,
                0.9495554,
                0.8836903,
                0.84982145,
                0.8451007,
                0.87502867,
                0.9322783,
                1.0379951,
                1.2038503,
                1.3998994,
                1.6277765,
                1.876774,
                2.0781448,
                2.2990248,
                2.509696,
                2.7378345,
                3.0210533,
                3.3794851,
                3.76058,
                4.177537,
                4.621196,
                5.0730963,
                5.5325174,
                5.9695816,
                6.8397713,
                9.786043,
                14.126716,
                13.661171,
                21.332146,
                62.4961,
                113.12327,
                591.6532,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
            ]
            weightsDn = [
                0.59782815,
                0.807688,
                0.7068454,
                1.1679783,
                0.93490654,
                1.0152472,
                1.047385,
                1.0350986,
                0.9933875,
                1.0624125,
                1.0984166,
                1.1376686,
                1.0986158,
                1.1164582,
                1.126787,
                1.137055,
                1.1452283,
                1.130928,
                1.1201867,
                1.1079938,
                1.0993465,
                1.091717,
                1.0839876,
                1.0777133,
                1.0688987,
                1.0591336,
                1.0446264,
                1.0307156,
                1.01609,
                1.0019246,
                0.9908169,
                0.9804498,
                0.96907,
                0.961301,
                0.9547044,
                0.9521451,
                0.95375824,
                0.95871556,
                0.97194415,
                0.9945388,
                1.0261062,
                1.0728791,
                1.1109722,
                1.1248587,
                1.1034995,
                1.0646065,
                1.004128,
                0.9337611,
                0.852428,
                0.78408086,
                0.7321661,
                0.67489123,
                0.6182854,
                0.5625684,
                0.4955873,
                0.44199666,
                0.39557,
                0.3605184,
                0.33896416,
                0.32952267,
                0.3248907,
                0.3259218,
                0.33180428,
                0.34167248,
                0.35624802,
                0.37438077,
                0.42493474,
                0.61087203,
                0.89509976,
                0.8834194,
                1.4085646,
                4.1966577,
                7.666078,
                40.053448,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
            ]
        if self.era == 2018:
            weightsNom = [
                4.7595778,
                1.080637,
                1.2167456,
                0.88091785,
                0.76623887,
                1.0115454,
                1.3231441,
                1.3395604,
                1.1045256,
                0.9123554,
                0.82408345,
                0.80039644,
                0.7907753,
                0.8067813,
                0.8323937,
                0.8564064,
                0.8725103,
                0.8821015,
                0.8946645,
                0.91871005,
                0.9464174,
                0.96640754,
                0.9801638,
                0.99084246,
                0.99259967,
                0.9873948,
                0.97978294,
                0.9812892,
                0.9868872,
                0.99286324,
                0.9982276,
                1.0017577,
                1.0033858,
                1.0051285,
                1.0066245,
                1.008838,
                1.0116601,
                1.0160315,
                1.0208794,
                1.0281874,
                1.03718,
                1.047111,
                1.0584829,
                1.0710881,
                1.0818367,
                1.0950999,
                1.1098212,
                1.1259873,
                1.142475,
                1.1547482,
                1.1680748,
                1.1769618,
                1.1830877,
                1.1996388,
                1.1972251,
                1.1950814,
                1.1915771,
                1.2094747,
                1.2186052,
                1.2395588,
                1.252892,
                1.2392353,
                1.1711651,
                1.0816797,
                1.0067116,
                0.9124977,
                0.86009574,
                0.7820993,
                0.65758467,
                0.63088226,
                0.59884727,
                0.714113,
                0.71516764,
                0.5360348,
                0.40905133,
                0.4283063,
                0.44173172,
                0.44235674,
                1.0353466,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
            ]
            weightsUp = [
                4.4270334,
                0.94696635,
                1.0174417,
                0.75674623,
                0.660584,
                0.87549543,
                1.1313093,
                1.132636,
                0.92944854,
                0.75887454,
                0.6724943,
                0.643334,
                0.6300619,
                0.6410678,
                0.66177934,
                0.6823179,
                0.6978642,
                0.7102508,
                0.72788054,
                0.7584186,
                0.7958842,
                0.83003813,
                0.8601236,
                0.8863557,
                0.9013146,
                0.9056365,
                0.90373224,
                0.9073797,
                0.9133702,
                0.91952753,
                0.9258928,
                0.9319249,
                0.9378536,
                0.94576323,
                0.9556233,
                0.9688274,
                0.9859416,
                1.0086592,
                1.0367004,
                1.072807,
                1.1168977,
                1.1687762,
                1.2295333,
                1.2994728,
                1.3751776,
                1.4623153,
                1.559881,
                1.6679225,
                1.7843952,
                1.9008608,
                2.0239122,
                2.1420274,
                2.2552886,
                2.387414,
                2.4787638,
                2.5656967,
                2.6453657,
                2.7715604,
                2.8805547,
                3.0244043,
                3.161072,
                3.2420754,
                3.1878612,
                3.074373,
                2.9981654,
                2.8564253,
                2.837117,
                2.723575,
                2.4203987,
                2.4557285,
                2.4652593,
                3.1080935,
                3.2893617,
                2.604268,
                2.0987904,
                2.3212414,
                2.5304043,
                2.6816385,
                6.6539707,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
            ]
            weightsDn = [
                5.1427293,
                1.252341,
                1.460148,
                1.0309888,
                0.89403486,
                1.1795671,
                1.5657326,
                1.6002822,
                1.3279383,
                1.1164327,
                1.028542,
                1.0120672,
                1.0054909,
                1.0262952,
                1.0569706,
                1.0835164,
                1.096756,
                1.0971861,
                1.0959034,
                1.1031847,
                1.1105756,
                1.1076767,
                1.1002375,
                1.0946506,
                1.0853548,
                1.0738025,
                1.0631845,
                1.0639712,
                1.0690187,
                1.0731227,
                1.0745729,
                1.0717589,
                1.0643873,
                1.0542315,
                1.0403752,
                1.023244,
                1.0023034,
                0.9783363,
                0.95047414,
                0.9209969,
                0.88969696,
                0.8565766,
                0.8227402,
                0.78869885,
                0.75298655,
                0.719494,
                0.6880412,
                0.6591464,
                0.63260126,
                0.60641074,
                0.5837689,
                0.56197304,
                0.5418327,
                0.5288385,
                0.5093433,
                0.49132562,
                0.4733166,
                0.4634092,
                0.44906828,
                0.4377016,
                0.42218664,
                0.3968928,
                0.3552229,
                0.3097935,
                0.2716718,
                0.23172171,
                0.20541322,
                0.1756697,
                0.13896713,
                0.1255068,
                0.112196885,
                0.12601438,
                0.1188145,
                0.083754934,
                0.060007818,
                0.05885399,
                0.05669032,
                0.052844524,
                0.1147116,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
            ]

        weightsNom = ak.Array(weightsNom)
        weightsUp = ak.Array(weightsUp)
        weightsDn = ak.Array(weightsDn)
        if hasattr(events, "Pileup"):
            weightsMask = (np.array(events.Pileup.nTrueInt)).astype(int)
            out = {}
            out["PUWeight"] = weightsNom[weightsMask]
            out["PUWeightUp"] = weightsUp[weightsMask]
            out["PUWeightDn"] = weightsDn[weightsMask]
        else:
            out = {}
            out["PUWeight"] = ak.ones_like(events.genWeight)
            out["PUWeightUp"] = ak.ones_like(events.genWeight)
            out["PUWeightDn"] = ak.ones_like(events.genWeight)
        return out

    def doTracksDropping(self, events, tracks):
        probsLowPt = {2015: 0.027, 2016: 0.027, 2017: 0.022, 2018: 0.021}
        probsHighPt = {2015: 0.01, 2016: 0.01, 2017: 0.01, 2018: 0.01}
        cutLowPtTracks = (tracks.pt < 20) & (tracks.pt >= 1)
        lowPtTracks = tracks[cutLowPtTracks]
        cutHighPtTracks = tracks.pt >= 20
        highPtTracks = tracks[cutHighPtTracks]
        # Generate random numbers
        randomLow = ak.unflatten(
            np.random.random(ak.sum(ak.num(lowPtTracks))), ak.num(lowPtTracks)
        )
        randomHigh = ak.unflatten(
            np.random.random(ak.sum(ak.num(highPtTracks))), ak.num(highPtTracks)
        )
        # Now get the tracks
        passLowPtTracks = lowPtTracks[(randomLow > probsLowPt[self.era])]
        passHighPtTracks = highPtTracks[(randomHigh > probsHighPt[self.era])]
        # And then broadcast them together again
        return {
            "": tracks,
            "_TRACKUP": ak.concatenate([passLowPtTracks, passHighPtTracks], axis=1),
        }

    def doJECJERVariations(self, events, jets):
        jets_corrected, met_c = apply_jecs(
            isMC=self.isMC,
            Sample=self.sample,
            era=self.era,
            events=events,
            doStoc=False,
        )
        # print(jets_corrected["JES_jes"].__dict__)
        jetsOut = {
            "": self.selectByJets(events, self.leptons, jets_corrected)[1],
            "_JECUP": self.selectByJets(
                events, self.leptons, jets_corrected["JES_jes"].up
            )[1],
            "_JECDOWN": self.selectByJets(
                events, self.leptons, jets_corrected["JES_jes"].down
            )[1],
            "_JERUP": self.selectByJets(events, self.leptons, jets_corrected["JER"].up)[
                1
            ],
            "_JERDOWN": self.selectByJets(
                events, self.leptons, jets_corrected["JER"].down
            )[1],
        }

        return jetsOut

    def doAllPlots(self, channel, debug=True):
        # ------------------------------------------------------------------------------
        # ------------------------------- PLOTTING -------------------------------------
        # ------------------------------------------------------------------------------
        out = {}
        # Define outputs for plotting
        if debug:
            print("Saving reco variables for channel %s" % channel)
        out["run"] = self.events.run[:]
        # Object: leptons
        out["leadlep_pt"] = self.leptons.pt[:, 0]
        out["subleadlep_pt"] = self.leptons.pt[:, 1]
        out["leadlep_eta"] = self.leptons.eta[:, 0]
        out["subleadlep_eta"] = self.leptons.eta[:, 1]
        out["leadlep_phi"] = self.leptons.phi[:, 0]
        out["subleadlep_phi"] = self.leptons.phi[:, 1]
        out["nleptons"] = ak.num(self.leptons, axis=1)[:]
        out["nmuons"] = ak.num(self.muons)
        out["nelectrons"] = ak.num(self.electrons)

        # Object: reconstructed Z
        out["Z_pt"] = self.Zcands.pt[:]
        out["Z_eta"] = self.Zcands.eta[:]
        out["Z_phi"] = self.Zcands.phi[:]
        out["Z_m"] = self.Zcands.mass[:]

        # Object: jets, a bit tricky as number varies per event!
        out["njets"] = ak.num(self.jets, axis=1)[:]
        out["nBLoose"] = ak.sum(
            (self.jets.btag >= btagcuts("Loose", self.era)), axis=1
        )[:]
        out["nBMedium"] = ak.sum(
            (self.jets.btag >= btagcuts("Medium", self.era)), axis=1
        )[:]
        out["nBTight"] = ak.sum(
            (self.jets.btag >= btagcuts("Tight", self.era)), axis=1
        )[:]
        out["leadjet_pt"] = ak.fill_none(
            ak.pad_none(self.jets.pt, 1, axis=1, clip=True), 0.0
        )[
            :, 0
        ]  # So take all events, if there is no jet_pt fill it with none, then replace none with 0
        out["leadjet_eta"] = ak.fill_none(
            ak.pad_none(self.jets.eta, 1, axis=1, clip=True), -999
        )[
            :, 0
        ]  # So take all events, if there is no jet_pt fill it with none, then replace none with -999
        out["leadjet_phi"] = ak.fill_none(
            ak.pad_none(self.jets.phi, 1, axis=1, clip=True), -999
        )[
            :, 0
        ]  # So take all events, if there is no jet_pt fill it with none, then replace none with -999
        out["subleadjet_pt"] = ak.fill_none(
            ak.pad_none(self.jets.pt, 2, axis=1, clip=True), 0.0
        )[
            :, 1
        ]  # So take all events, if there is no jet_pt fill it with none, then replace none with 0
        out["subleadjet_eta"] = ak.fill_none(
            ak.pad_none(self.jets.eta, 2, axis=1, clip=True), -999
        )[
            :, 1
        ]  # So take all events, if there is no jet_pt fill it with none, then replace none with -999
        out["subleadjet_phi"] = ak.fill_none(
            ak.pad_none(self.jets.phi, 2, axis=1, clip=True), -999
        )[
            :, 1
        ]  # So take all events, if there is no jet_pt fill it with none, then replace none with -999
        out["trailjet_pt"] = ak.fill_none(
            ak.pad_none(self.jets.pt, 3, axis=1, clip=True), 0.0
        )[
            :, 2
        ]  # So take all events, if there is no jet_pt fill it with none, then replace none with 0
        out["trailjet_eta"] = ak.fill_none(
            ak.pad_none(self.jets.eta, 3, axis=1, clip=True), -999
        )[
            :, 2
        ]  # So take all events, if there is no jet_pt fill it with none, then replace none with -999
        out["trailjet_phi"] = ak.fill_none(
            ak.pad_none(self.jets.phi, 3, axis=1, clip=True), -999
        )[
            :, 2
        ]  # So take all events, if there is no jet_pt fill it with none, then replace none with -999
        out["H_T"] = ak.sum(self.jets.pt, axis=1)[:]
        out["L_T"] = ak.sum(self.leptons.pt, axis=1)[:]
        out["MET_pt"] = self.events.MET.pt[:]  # Not used, but hust in case

        # Corrections
        if self.isMC:
            out["bTagWeight"] = self.btagweights["central"][:]
            out["PUWeight"] = self.puweights["PUWeight"][:]
            out["L1prefireWeight"] = self.l1preweights["L1prefire_nom"][:]
            out["TrigSF"] = self.triggerSFs["TrigSF"][:]
            out["LepSF"] = self.leptonSFs["LepSF"][:]

        if self.var == "" and self.do_syst:
            out["bTagWeight_HFCorr_Up"] = self.btagweights["HFcorrelated_Up"][:]
            out["bTagWeight_HFCorr_Dn"] = self.btagweights["HFcorrelated_Dn"][:]
            out["bTagWeight_LFCorr_Up"] = self.btagweights["LFcorrelated_Up"][:]
            out["bTagWeight_LFCorr_Dn"] = self.btagweights["LFcorrelated_Dn"][:]
            out["bTagWeight_HFUnCorr_Up"] = self.btagweights["HFuncorrelated_Up"][:]
            out["bTagWeight_HFUnCorr_Dn"] = self.btagweights["HFuncorrelated_Dn"][:]
            out["bTagWeight_LFUnCorr_Up"] = self.btagweights["LFuncorrelated_Up"][:]
            out["bTagWeight_LFUnCorr_Dn"] = self.btagweights["LFuncorrelated_Dn"][:]

            out["L1prefireWeight_Up"] = self.l1preweights["L1prefire_up"][:]
            out["L1prefireWeight_Dn"] = self.l1preweights["L1prefire_down"][:]

            out["PUWeight_Up"] = self.puweights["PUWeightUp"][:]
            out["PUWeight_Dn"] = self.puweights["PUWeightDn"][:]

            out["ISRWeight_Up"] = self.isrweights["PSWeight_ISRUP"][:]
            out["ISRWeight_Dn"] = self.isrweights["PSWeight_ISRDOWN"][:]
            out["FSRWeight_Up"] = self.isrweights["PSWeight_FSRUP"][:]
            out["FSRWeight_Dn"] = self.isrweights["PSWeight_FSRDOWN"][:]

            out["TrigSF_Up"] = self.triggerSFs["TrigSFUp"][:]
            out["TrigSF_Dn"] = self.triggerSFs["TrigSFDn"][:]

            out["LepSF_ElUp"] = self.leptonSFs["LepSFElUp"][:]
            out["LepSF_ElDn"] = self.leptonSFs["LepSFElDown"][:]
            out["LepSF_MuUp"] = self.leptonSFs["LepSFMuUp"][:]
            out["LepSF_MuDn"] = self.leptonSFs["LepSFMuDown"][:]

        if self.doTracks:
            out["ntracks"] = ak.num(self.tracks, axis=1)[:]
            if self.isSpherable:
                if self.doClusters and self.isClusterable:
                    out["nclusters"] = ak.num(self.clusters, axis=1)[:]
                    out["leadcluster_pt"] = self.clusters.pt[:, 0]
                    out["leadcluster_eta"] = self.clusters.eta[:, 0]
                    out["leadcluster_phi"] = self.clusters.phi[:, 0]
                    out["leadcluster_ntracks"] = ak.num(self.constituents[:, 0], axis=1)
                    out["leadcluster_nmuon"] = ak.sum(
                        abs(self.constituents.pdgId[:, 0]) == 13, axis=1
                    )
                    out["leadcluster_nelectron"] = ak.sum(
                        abs(self.constituents.pdgId[:, 0]) == 11, axis=1
                    )
                    out["leadcluster_npion"] = ak.sum(
                        abs(self.constituents.pdgId[:, 0, :]) == 211, axis=1
                    )
                    out["leadcluster_nkaon"] = ak.sum(
                        (abs(self.constituents.pdgId[:, 0]) - 310) < 15, axis=1
                    )

                    out["leadcluster_m"] = self.clusters.mass[:, 0]

                    boost_leading = ak.zip(
                        {
                            "px": self.clusters[:, 0].px * -1,
                            "py": self.clusters[:, 0].py * -1,
                            "pz": self.clusters[:, 0].pz * -1,
                            "mass": self.clusters[:, 0].mass,
                        },
                        with_name="Momentum4D",
                    )

                    leadingclustertracks = self.constituents[:, 0]
                    leadingclustertracks_boostedagainstSUEP = (
                        leadingclustertracks.boost_p4(boost_leading)
                    )

                    evalsL = self.sphericity(self.events, leadingclustertracks, 2)
                    evalsC = self.sphericity(
                        self.events, leadingclustertracks_boostedagainstSUEP, 2
                    )

                    out["leadclusterSpher_L"] = np.real(
                        1.5 * (evalsL[:, 0] + evalsL[:, 1])
                    )
                    out["leadclusterSpher_C"] = np.real(
                        1.5 * (evalsC[:, 0] + evalsC[:, 1])
                    )

        if self.doGen:
            if debug:
                print("Saving gen variables")
            if self.isDY:
                out["genZpt"] = self.Zpt

            else:
                out["genZpt"] = self.genZ.pt[:, 0]
                out["genZeta"] = self.genZ.eta[:, 0]
                out["genZphi"] = self.genZ.phi[:, 0]
                out["genHpt"] = self.genH.pt[:, 0]
                out["genHeta"] = self.genH.eta[:, 0]
                out["genHphi"] = self.genH.phi[:, 0]
        # out["nPU"] = self.getNPU()[:]
        return out

    def getNPU(self):
        if self.isMC:
            if hasattr(self.events, "Pileup"):
                return self.events.Pileup.nTrueInt
            else:
                return ak.ones_like(self.events.genWeight)
        else:
            if self.era == 2015 or self.era == 2016:
                lumifile = "data/Lumi/16.json"
            if self.era == 2017:
                lumifile = "data/Lumi/17.json"
            if self.era == 2018:
                lumifile = "data/Lumi/18.json"
            with open(lumifile) as lf:
                runsAndLumis = json.loads(lf.read())
            PU = []
            for iev in range(len(self.events.run)):
                if not (str(self.events.run[iev])) in runsAndLumis:
                    PU.append(-1)
                    print(self.events.run[iev])
                    continue
                if not (
                    str(self.events.luminosityBlock[iev])
                    in runsAndLumis[str(self.events.run[iev])]
                ):
                    PU.append(-1)
                    continue
                PU.append(
                    round(
                        runsAndLumis[str(self.events.run[iev])][
                            str(self.events.luminosityBlock[iev])
                        ]
                    )
                )
            PU = ak.Array(PU)
            return PU

    def postprocess(self, accumulator):
        return accumulator
